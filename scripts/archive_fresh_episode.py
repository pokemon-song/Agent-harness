"""Validate, render and package independent live A/B episodes using recorded physics only."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np

from go2_demo.occluded_video import _sha


MODES = ("empty", "ledger", "query")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(condition, reason):
    if not condition:
        raise ValueError(reason)


def write(path, obj):
    partial = path.with_suffix(".json.tmp")
    partial.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(path)


def pack(path, entries, compression):
    partial = path.with_suffix(".zip.tmp")
    with ZipFile(partial, "w", compression=compression, allowZip64=True) as archive:
        for filename, name in entries:
            archive.write(filename, name)
    with ZipFile(partial) as archive:
        verify(archive.testzip() is None, f"CRC mismatch in {path}")
    partial.replace(path)


def publish_video(src, dst, capture, clip):
    name, manifest = ("return.mp4", "return-video.json") if clip else ("full.mp4", "video.json")
    if not (src / manifest).exists():
        args = [sys.executable, "-m", "go2_demo.occluded_video", str(src), "--fast"]
        if clip:
            args.append("--return-only")
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
    video = read(src / manifest)
    start = capture["handoff_s"] if clip else capture["start_s"]
    expected = max(1 if clip else 2, math.ceil((capture["end_s"] - start) * video["fps"] - 1e-8) + 1)
    verify(video["source"] == capture and video["file"] == name, f"Wrong capture for {src / name}")
    verify(video["frame_count"] == expected and video["decoded_all_frames"] and
           video["initial_and_final_states_included"], f"Incomplete recording {src / name}")
    verify(_sha(src / name) == video["sha256"], f"MP4 digest differs in {src}")
    partial = dst / (name + ".tmp")
    shutil.copyfile(src / name, partial)
    verify(_sha(partial) == video["sha256"], f"MP4 copy differs for {dst}")
    partial.replace(dst / name)
    write(dst / manifest, video)
    return {"file": name, "fps": video["fps"], "frames": expected, "duration_s": video["duration_s"],
            "sha256": video["sha256"]}


def audit_stage(src, stage, recorded, events):
    inputs = [r for r in events if r["event"] == "planner_input" and r.get("stage") == stage]
    reads = [r["detail"] for r in events if r["event"] == "memory_result" and r.get("stage") == stage]
    actions = [r["detail"] for r in events if r["event"] == "native_action" and r.get("stage") == stage]
    verify(len(inputs) == recorded["planner_requests"] and
           sum(r["requests"] for r in reads) == recorded["reader_requests"], f"{src}: {stage} request mismatch")
    verify(Counter(r["tool"] for r in actions) == Counter(recorded["native_attempts"]), f"{src}: {stage} actions mismatch")
    first = next((r for r in inputs if any(t["task_label"] == stage for t in r["current_targets"])), None)
    first_index = first["index"] if first else None
    reported_first = recorded["first_planner_target_candidate"]
    verify(first_index == (reported_first["planner_decision"] if reported_first else None), f"{src}: {stage} first target differs")
    nav = [r["exploration"] for r in inputs if isinstance(r.get("exploration"), dict)]
    selected_feedback = [e["detail"] for e in events if e["event"] == "planner_feedback" and e["stage"] == stage
                         and "chosen_waypoint" in e["detail"]]
    goals = [e["detail"]["task_state"] for e in events if e["event"] == "subgoal_set" and e["stage"] == stage]
    checks = [e["task_state"] for e in events if e["event"] == "subgoal_verification" and e["stage"] == stage]
    states = [r["task_state"] for r in inputs if isinstance(r.get("task_state"), dict)]
    goal_states = [s["active"] for s in [*goals, *checks] if s.get("active")]
    completed = {g["id"]: g for g in goal_states if g["status"] == "completed"}
    choices = [e for e in events if e["event"] == "model_choice" and e.get("stage") == stage]
    feedback = [e["detail"] for e in events if e["event"] == "planner_feedback" and e.get("stage") == stage]
    return {"planner_requests": len(inputs), "reader_requests": recorded["reader_requests"],
            "total_requests": recorded["total_requests"], "native_actions": dict(recorded["native_attempts"]),
            "planner_tools": recorded["planner_tools"], "first_seen_decision": first_index,
            "first_seen_sim_time_s": reported_first["sim_time_s"] if reported_first else None,
            "stage_start_sim_s": recorded["start_sim_s"], "stage_end_sim_s": recorded["end_sim_s"],
            "path_m": recorded["path_m"],
            "boundary_refusals": recorded["boundary_refusals"], "depth_refusals": recorded["depth_refusals"],
            "known_usage": recorded["known_usage"], "unknown_usage_requests": recorded["unknown_usage_requests"],
            "memory_lookups": len(reads), "fully_cached_lookups": sum(bool(r.get("cache_hit")) for r in reads),
            "memory_result_status": dict(Counter(r["status"] for r in reads)),
            "stalled_inputs": sum(r["stalled"] for r in nav),
            "inputs_with_proposed_viewpoints": sum(bool(r["viewpoints"]) for r in nav),
            "selected_viewpoints": len(selected_feedback),
            "viewpoints_with_at_least_019m_measured_displacement":
                sum(r["measured_displacement_m"] >= .19 for r in selected_feedback),
            "viewpoint_outcomes": dict(Counter(r.get("error_code") or r["status"] for r in selected_feedback)),
            "subgoals": {"created": len(goals), "completed": len(completed),
                         "criteria": dict(Counter(s["active"]["criterion"] for s in goals)),
                         "completion_sources": dict(Counter(g["verification"].get("source") for g in completed.values())),
                         "verification_events": len(checks),
                         "recovery_required_inputs": sum(bool(s.get("recovery_required")) for s in states),
                         "recovery_reasons": dict(Counter(s.get("recovery_reason") for s in states if s.get("recovery_required"))),
                         "recovery_routes": sum(e["tool"] == "follow_waypoints" and
                                                e["arguments"].get("route_role") == "recovery" for e in choices),
                         "route_continuations": sum(e["tool"] == "continue_subgoal" for e in choices),
                         "execution_slices": sum(f.get("error_code") == "execution_slice_complete" for f in feedback),
                         "final_state": recorded.get("task_state")},
            "planner_errors": dict(Counter(e["code"] for e in events if e["event"] == "planner_error" and e.get("stage") == stage))}


def audit_subgoal_state(src, events, *, handoff_context="reset", outbound_state=None):
    """Distinguish a clean start from retained public history at the A/B boundary."""
    verify(handoff_context in ("reset", "retain"), f"{src}: unknown handoff context policy")
    for stage in ("A", "B"):
        inputs = [e for e in events if e["event"] == "planner_input" and e.get("stage") == stage]
        if not inputs:
            continue
        for row in inputs:
            state = row.get("task_state")
            verify(isinstance(state, dict) and state.get("stage") == stage,
                   f"{src}: {stage} missing or cross-stage subgoal state")
            verify(row.get("exploration") is None, f"{src}: subgoal episode exposed viewpoint proposals")
        first = inputs[0]
        state = first["task_state"]
        verify(state.get("active") is None and state.get("recent_goals") == [] and
               state.get("last_action") is None and state.get("action_count") == 0 and
               state.get("no_progress_actions") == 0 and state.get("recovery_required") is False and
               state.get("recovery_reason") is None,
               f"{src}: {stage} execution state was not reset")
        previous = state.get("previous_stage_state")
        if stage == "B" and handoff_context == "retain":
            verify(isinstance(previous, dict) and previous.get("stage") == "A" and
                   outbound_state is not None and previous == outbound_state,
                   f"{src}: B did not retain the exact completed A execution state")
        else:
            verify(previous is None, f"{src}: unexpected prior-stage execution state")
        sighting = state.get("last_target_sighting")
        if sighting is not None:
            verify(sighting.get("target_label") == stage and
                   any(c.get("task_label") == stage for c in first["current_targets"]) and
                   sighting.get("observer_pose") is not None and
                   sighting["observer_pose"]["x"] == first["current_odometry"]["translation_m"][0] and
                   sighting["observer_pose"]["y"] == first["current_odometry"]["translation_m"][1] and
                   sighting["observer_pose"]["yaw"] == first["current_odometry"]["yaw_rad"],
                   f"{src}: {stage} inherited a target sighting without a current observation")
    verify(not any(e["event"] == "model_choice" and e.get("tool") == "explore_viewpoint" for e in events),
           f"{src}: subgoal episode called a viewpoint selection tool")


def audit_handoff_context(src, events, manifest):
    """Verify the retained conversation counters and memory card at first B input."""
    b_inputs = [e for e in events if e["event"] == "planner_input" and e.get("stage") == "B"]
    if not b_inputs:
        return
    first_b = b_inputs[0]
    if manifest.get("handoff_context", "reset") == "reset":
        verify(first_b["history_turns"] == 0 and first_b["memory_card"] is None and
               first_b.get("latest_memory_feedback") is None,
               f"{src}: nonempty history/card at first return decision")
        return
    verify(manifest.get("handoff_context") == "retain" and
           manifest.get("context_retention") == "full_public_dialogue" and
           manifest.get("reset_model_history_at_A") is False,
           f"{src}: inconsistent retained-context manifest")
    a_inputs = [e for e in events if e["event"] == "planner_input" and e.get("stage") == "A"]
    a_choices = [e for e in events if e["event"] == "model_choice" and e.get("stage") == "A"]
    verify(bool(a_inputs) and bool(a_choices), f"{src}: missing outbound conversation")
    turns, messages = first_b.get("retained_context_turns"), first_b.get("retained_context_messages")
    verify(type(turns) is int and turns >= len(a_choices) and
           type(messages) is int and messages >= 2 * turns,
           f"{src}: outbound conversation was cleared or truncated before B")
    verify(first_b.get("retained_context_images") == turns,
           f"{src}: retained conversation lost outbound camera observations")
    for key in ("retained_context_turns", "retained_context_messages", "retained_context_images"):
        counts = [e.get(key) for e in [*a_inputs, *b_inputs]]
        verify(all(type(value) is int and value >= 0 for value in counts) and
               all(new >= old for old, new in zip(counts, counts[1:])),
               f"{src}: retained context counter {key} decreased")
    last_a = a_inputs[-1]
    verify(first_b["memory_card"] == last_a["memory_card"] and
           first_b.get("latest_memory_feedback") == last_a.get("latest_memory_feedback"),
           f"{src}: A memory card or query feedback was cleared at handoff")


def audit_native_action_ids(src, events, *, retain_context=False):
    """Reject executor-cache collisions, including replayed successful replies.

    Legacy reset episodes clear the harness cache at the A/B boundary, allowing
    the counter to restart there. Retained episodes must use globally unique IDs.
    """
    seen = set()
    count = 0
    for event in events:
        if event["event"] != "native_action":
            continue
        detail = event["detail"]
        call_id, stage = detail.get("call_id"), event.get("stage")
        verify(isinstance(call_id, str) and bool(call_id) and stage in ("A", "B"),
               f"{src}: native action lacks a valid call ID or stage")
        reply = detail.get("reply", {})
        verify(reply.get("error_code") != "CALL_ID_REUSED",
               f"{src}: {stage} native action {call_id} failed with CALL_ID_REUSED")
        verify(reply.get("call_id", call_id) == call_id,
               f"{src}: native reply does not match its action call ID")
        key = call_id if retain_context else (stage, call_id)
        verify(key not in seen, f"{src}: duplicate native action call ID {call_id} in "
               f"{'episode' if retain_context else stage + ' stage'}")
        seen.add(key)
        count += 1
    return {"scope": "episode" if retain_context else "stage", "actions_checked": count,
            "unique_call_ids": len(seen), "call_id_reuse_errors": 0}


def audit_collision_score(src, manifest, metrics):
    """Keep completion independent of contact penalties, with legacy compatibility."""
    fields = ("goal_points", "collision_count", "collision_penalty", "score")
    if "collision_policy" not in manifest and not any(key in metrics for key in fields):
        return {}
    requirements = manifest.get("completion_requirements", {})
    points, cost = manifest.get("points_per_checkpoint"), manifest.get("collision_penalty_points")
    verify(manifest.get("collision_policy") == "record_and_penalize" and
           manifest.get("first_contact_terminates") is False and
           all(requirements.get(key) is False for key in (
               "collision_free_required", "b_contact_ends_episode", "other_scene_contacts_end_episode")) and
           type(points) is int and points > 0 and type(cost) is int and cost >= 0 and
           requirements.get("collision_penalty_points") == cost,
           f"{src}: inconsistent collision scoring policy")
    verify(all(type(metrics.get(key)) is int and metrics[key] >= 0 for key in fields),
           f"{src}: missing or invalid collision score fields")
    evaluation = metrics["independent_evaluation"]
    count = len(evaluation["collision_events"])
    verify(metrics["collision_count"] == evaluation.get("collision_count") == count,
           f"{src}: collision count differs from recorded contacts")
    # B points require full episode success, including finish and the final stop.
    # A geometric B checkpoint alone must not grant completion points.
    goals = points * (int(evaluation["checkpoints"]["A"]["passed"]) + int(metrics["success"]))
    penalty = cost * count
    verify(metrics["goal_points"] == goals and metrics["collision_penalty"] == penalty and
           metrics["score"] == max(0, goals - penalty),
           f"{src}: collision score arithmetic differs from evaluation")
    return {key: metrics[key] for key in fields}


def audit_arm(raw, out, mode):
    src, dst = raw / mode, out / mode
    dst.mkdir(parents=True, exist_ok=True)
    m, man, cap = read(src / "metrics.json"), read(src / "manifest.json"), read(src / "video/capture.json")
    score = audit_collision_score(src, man, m)
    verify(m["episode_id"] == man["episode_id"] and m["mode"] == mode == man["mode"], f"{src}: identity mismatch")
    verify(man["archive_inputs"] == [] and man["initial_memory"] == "empty" and
           man["outbound"] == "fresh_live_planner", f"{src}: unexpected initial state")
    verify(cap["outbound_label"] == f"LIVE SEARCH FOR A | {mode.upper()}", f"{src}: wrong recording")
    events = [json.loads(row) for row in (src / "events.jsonl").read_text().splitlines()]
    call_ids = audit_native_action_ids(src, events, retain_context=man.get("handoff_context") == "retain")
    if man.get("navigation") == "persistent_subgoals_v1":
        audit_subgoal_state(src, events, handoff_context=man.get("handoff_context", "reset"),
                            outbound_state=m["stages"]["A"].get("task_state"))
    first = next(e for e in events if e["event"] == "planner_input")
    verify(first["stage"] == "A" and first["index"] == 1 and first["history_turns"] == 0 and first["memory_card"] is None
           and first["current_odometry"] == man["initial_odometry"], f"{src}: nonempty initial planner context")
    for name, key in (("model.mjb", "model_sha256"), ("poses.npz", "poses_sha256")):
        verify(_sha(src / "video" / name) == cap[key], f"{src}: wrong source recording {name}")
    b_entered = "B" in m["stages"]
    verify((cap["handoff_s"] is not None) == b_entered and
           sum(e["event"] == "A_handoff" for e in events) == int(b_entered), f"{src}: inconsistent A handoff")
    if b_entered:
        audit_handoff_context(src, events, man)
    ev = m["independent_evaluation"]
    checkpoints = ev["checkpoints"]
    with np.load(src / "video/poses.npz", allow_pickle=False) as sample:
        times, poses = sample["time"], sample["qpos"]
    verify(len(times) == cap["sample_count"] == len(poses) and np.all(np.diff(times) > 0), f"{src}: invalid timebase")
    verify(abs(times[0] - cap["start_s"]) < 1e-8 and abs(times[-1] - m["final_sim_s"]) < 1e-8, f"{src}: recording incomplete")
    trajectory = [json.loads(row) for row in (src / "trajectory.jsonl").read_text().splitlines()]
    verify(len(times) == len(trajectory) + 1 and np.allclose(times[1:], [r["sim_time_s"] for r in trajectory],
                                                             rtol=0, atol=1e-7), f"{src}: control samples missing")
    # The scoring frame uses the base body, whose xy differs slightly from qpos.
    # Recover its initial position from a synchronized odometry and world pose.
    by_time = {r["sim_time_s"]: r for r in trajectory}
    by_time_xy = {t: np.asarray(r["base_xyz_m"][:2]) for t, r in by_time.items()}
    for event in events:
        if event["event"] != "native_action":
            continue
        action = event["detail"]
        after = by_time.get(action["time_after"])
        if after is None:
            continue
        odom = action["odom_after"]
        angle = after["yaw_rad"] - odom["yaw_rad"]
        c, s = math.cos(angle), math.sin(angle)
        dx, dy = odom["translation_m"]
        by_time_xy[times[0]] = np.asarray(after["base_xyz_m"][:2]) - [c * dx - s * dy, s * dx + c * dy]
        break
    stages = {}
    for stage, report in m["stages"].items():
        facts = audit_stage(src, stage, report, events)
        start, end = report["start_sim_s"], report["end_sim_s"]
        stage_times = times[(times >= start) & (times <= end)]
        if all(t in by_time_xy for t in stage_times):
            xy = np.asarray([by_time_xy[t] for t in stage_times])
            goal = np.asarray(checkpoints[stage]["target_xy_m"])
            facts["min_base_distance_to_goal_m"] = float(np.linalg.norm(xy - goal, axis=1).min())
            facts["net_displacement_m"] = float(np.linalg.norm(xy[-1] - xy[0]))
            path = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            verify(abs(path.sum() - facts["path_m"]) < 1e-6, f"{src}: {stage} measured path mismatch")
            if facts["first_seen_sim_time_s"] is not None:
                steps = sum(stage_times[1:] <= facts["first_seen_sim_time_s"])
                facts["path_to_first_seen_m"] = float(path[:steps].sum())
        else:
            verify(start == end, f"{src}: initial scoring pose unavailable in a moving stage")
            facts["min_base_distance_to_goal_m"] = None
            facts["net_displacement_m"] = 0.0
        stages[stage] = facts
    videos = [publish_video(src / "video", dst, cap, False)]
    if b_entered:
        videos.append(publish_video(src / "video", dst, cap, True))
    for name in ("metrics.json", "manifest.json"):
        shutil.copyfile(src / name, dst / name)
    write(dst / "capture.json", cap)
    result = {"episode_id": m["episode_id"], "A_confirmed_at_checkpoint": checkpoints["A"]["achieved_at_checkpoint"],
              "B_entered": b_entered, "B_passed_if_entered": checkpoints["B"]["passed"] if b_entered else None,
              "passed": m["success"], "termination": m["termination"], "termination_detail": m["detail"],
              "collision": ev["collision"], "collision_events": ev["collision_events"], "fallen": ev["fallen"],
              "initial_image_sha256": _sha(src / first["image"]), "total_requests": m["total_requests"], "stages": stages,
              "native_call_ids": call_ids, "videos": videos, **score}
    write(dst / "audit.json", result)
    print(f"{mode}: A={result['A_confirmed_at_checkpoint']} B entered={b_entered} pass={result['passed']}; videos verified", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory containing independent memory-mode subdirectories")
    parser.add_argument("--output", type=Path, help="Default: results/<run-directory-name>.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--mode", choices=("all", *MODES), default="all", help="Single mode can render as soon as its run finishes.")
    selection.add_argument("--modes", nargs="+", choices=MODES, help="Audit and package only these modes, e.g. empty query.")
    args = parser.parse_args()
    raw, out = args.run.resolve(), (args.output or Path("results") / args.run.name).resolve()
    if raw == out:
        parser.error("Publish into a separate directory.")
    if args.mode != "all":
        audit_arm(raw, out, args.mode)
        return
    modes = tuple(args.modes or MODES)
    if len(set(modes)) != len(modes):
        parser.error("Choose each memory mode only once.")
    with ThreadPoolExecutor(max_workers=2) as pool:
        arms = dict(zip(modes, pool.map(lambda mode: audit_arm(raw, out, mode), modes), strict=True))
    verify(len({r["episode_id"] for r in arms.values()}) == len(arms), "Episode IDs must be distinct")
    verify(len({r["initial_image_sha256"] for r in arms.values()}) == 1, "Initial observations differ")
    manifests = [read(out / mode / "manifest.json") for mode in modes]
    keys = ("task", "seed", "floor_size_m", "model", "api_limit_per_stage", "max_native_per_stage",
            "max_sim_s_per_stage", "max_wall_s_per_stage", "completion_requirements")
    keys += tuple(key for key in ("navigation", "memory_queries", "fresh_driver_sha256", "verifier",
                                  "native_actions_per_slice", "history_unit", "navigation_reset_at_A",
                                  "handoff_context", "context_retention", "reset_model_history_at_A",
                                  "retain_memory_card_and_cache_at_A", "new_active_target_at_A",
                                  "collision_policy", "collision_penalty_points", "points_per_checkpoint",
                                  "first_contact_terminates") if any(key in man for man in manifests))
    verify(all(all(key in man for key in keys) for man in manifests), "Arm settings differ")
    fixed = {key: manifests[0][key] for key in keys}
    verify(all(all(man[key] == value for key, value in fixed.items()) for man in manifests), "Arm settings differ")
    summary = {"experiment": raw.name, "settings": fixed, "modes": list(modes), "replicates_per_arm": 1, "arms": arms,
               "A_confirmed": sum(r["A_confirmed_at_checkpoint"] for r in arms.values()),
               "B_entered": sum(r["B_entered"] for r in arms.values()),
               "B_passed_among_entered": sum(r["B_passed_if_entered"] is True for r in arms.values()),
               "full_successes": sum(r["passed"] for r in arms.values())}
    write(out / "summary.json", summary)
    video_bundle = {1: "single-full-video.zip", 2: "paired-full-videos.zip", 3: "three-full-videos.zip"}[len(modes)]
    pack(out / video_bundle, [(out / mode / "full.mp4", f"{mode}-full.mp4") for mode in modes], ZIP_STORED)
    data = [(out / "summary.json", "summary.json")]
    for mode in modes:
        for name in ("manifest.json", "metrics.json", "events.jsonl", "trajectory.jsonl", "episode_documents.json",
                     "video/capture.json", "video/poses.npz"):
            path = raw / mode / name
            if path.exists():
                data.append((path, f"{mode}/{name}"))
        data.append((out / mode / "audit.json", f"{mode}/audit.json"))
    pack(out / "episode-data.zip", data, ZIP_DEFLATED)
    print(out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
