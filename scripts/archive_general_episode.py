"""Offline audit, complete labelled replay and portable archive of a general-agent run.

A/B labels and evaluator timestamps are used only in this export. This module
never instantiates a controller, advances physics or calls a model API.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from go2_demo.occluded_video import _ffmpeg
from scripts.export_full_labeled_episode_video import project


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    """Allow resuming matching files, but never overwrite a different artifact."""
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        require(path.read_text() == text, f"Existing artifact differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(text)


def copy_new(source, destination):
    source, destination = Path(source), Path(destination)
    require(source.is_file() and not source.is_symlink(), f"Not a regular source file: {source}")
    if destination.exists():
        require(sha(source) == sha(destination), f"Existing artifact differs: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def audit(run, out):
    metrics, protocol = read(run / "metrics.json"), read(run / "protocol.json")
    require(metrics.get("termination") not in (None, "initializing"), "Run is not complete")
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    started = [e for e in events if e["event"] == "request_started"]
    ended = [e for e in events if e["event"] == "request_ended"]
    key = lambda e: (e["role"], e["ordinal"])
    start_keys, end_keys = [key(e) for e in started], [key(e) for e in ended]
    require(len(start_keys) == len(set(start_keys)), "Duplicate request-start ID")
    require(len(end_keys) == len(set(end_keys)), "Duplicate request-end ID")
    require(set(start_keys) == set(end_keys), "Request starts and ends do not match")
    counts, usage, usage_by_role, unknown = Counter(), Counter(), {}, 0
    for e in started:
        counts[e["role"]] += 1
    for e in ended:
        usage.update(e["usage"])
        usage_by_role.setdefault(e["role"], Counter()).update(e["usage"])
        unknown += int(not {"prompt_tokens", "completion_tokens"} <= set(e["usage"]))
    require(dict(counts) == metrics["requests"], "Request totals differ from metrics")
    require(dict(usage) == metrics["usage"], "Usage totals differ from metrics")
    require(unknown == metrics["unknown_usage_requests"], "Unknown-usage count differs")
    choices = Counter(e["tool"] for e in events if e["event"] == "choice")
    require(dict(choices) == metrics["tools_used"], "Tool choice totals differ")
    images = []
    for e in started:
        if e["role"] != "planner":
            continue
        image = e["image"]
        relative = Path(image["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Image path escapes run directory")
        require(sha(run / relative) == image["sha256"], f"Image hash differs: {relative}")
        images.append({**image, "request_ordinal": e["ordinal"]})
    frozen = read(out / "frozen-source.json")
    for name, expected in frozen.items():
        require(sha(out / "source_snapshot" / name) == expected, f"Frozen source differs: {name}")
    for name, expected in protocol["source_sha256"].items():
        require(frozen.get("go2_demo/" + name) == expected, f"Runtime protocol/source freeze differs: {name}")
    if (run / "memory.json").exists():
        memory = read(run / "memory.json")
        require(memory["event_count"] == len(memory["events"]), "Memory event count differs")
        ids = [e["event_id"] for e in memory["events"]]
        require(len(set(ids)) == len(ids), "Memory has duplicate event IDs")
    return {"requests": dict(counts), "requests_total": len(started), "usage": dict(usage),
            "usage_by_role": {role: dict(value) for role, value in usage_by_role.items()},
            "unknown_usage_requests": unknown, "tools_used": dict(choices),
            "native_actions": sum(e["event"] == "native_action" for e in events),
            "request_pairing_verified": True, "request_image_hashes_verified": len(images),
            "images": images, "frozen_source_files_verified": len(frozen),
            "success": metrics["success"], "termination": metrics["termination"],
            "score": metrics.get("score"), "metrics_are_original": True}


def load_recording(run):
    capture = read(run / "capture/capture.json")
    require(capture["mujoco_version"] == mujoco.__version__, "MuJoCo version differs from recording")
    for name, key in (("model.mjb", "model_sha256"), ("poses.npz", "poses_sha256")):
        require(sha(run / "capture" / name) == capture[key], f"Capture hash differs: {name}")
    model = mujoco.MjModel.from_binary_path(str(run / "capture/model.mjb"))
    with np.load(run / "capture/poses.npz", allow_pickle=False) as data:
        times, poses = data["time"].copy(), data["qpos"].copy()
    require(times.shape == (capture["sample_count"],) and poses.shape == (len(times), model.nq), "Pose shapes differ")
    require(len(times) and np.isfinite(times).all() and np.isfinite(poses).all(), "Invalid recording values")
    require(np.all(np.diff(times) > 0) and times[0] == capture["start_s"] and times[-1] == capture["end_s"],
            "Invalid recording timeline")
    evaluation = read(run / "evaluation.evaluator-only.json")
    a = evaluation.get("checkpoints", {}).get("A", {})
    handoff = float(a["sim_time_s"]) if a.get("achieved_at_checkpoint") else None
    handoff_source = "evaluation.checkpoints.A.sim_time_s" if handoff is not None else None
    if handoff is not None:
        require(times[0] <= handoff <= times[-1], "A acceptance is outside recording")
        require(np.min(np.abs(times - handoff)) < 1e-7, "A acceptance has no recorded pose")
    if capture.get("handoff_s") is not None:
        require(handoff is not None and abs(capture["handoff_s"] - handoff) < 1e-7,
                "Recorder/evaluator handoff timestamps differ")
    return capture, model, times, poses, handoff, handoff_source


def render(run, out, *, fps=12.5, preview_only=False):
    output = out / "full-labeled.mp4"
    metadata = out / "full-labeled.json"
    if output.exists() and not preview_only:
        previous = read(metadata)
        require(previous["sha256"] == sha(output), "Existing video hash differs")
        require(previous["source_capture"] == read(run / "capture/capture.json"), "Existing video has different source")
        require(previous["fps"] == fps and previous["decoded_all_frames"], "Existing video has different settings")
        return previous
    capture, model, times, poses, handoff, handoff_source = load_recording(run)
    count = max(2, math.ceil((times[-1] - times[0]) * fps - 1e-8) + 1)
    sources = np.searchsorted(times, times[0] + np.arange(count) / fps + 1e-9, side="right") - 1
    sources = np.clip(sources, 0, len(times) - 1)
    sources[0], sources[-1] = 0, len(times) - 1
    width, view_height, header, footer = 1280, 864, 64, 32
    height = view_height + header + footer
    model.vis.global_.offwidth, model.vis.global_.offheight = width, view_height
    model.vis.quality.offsamples = 0
    state = mujoco.MjData(model)
    state.qpos[:] = poses[0]
    mujoco.mj_forward(model, state)
    # Go2 base is its root free body. Verify this before using the recorded xyz.
    require(model.body("base").parentid == 0, "Replay expects a root base body")
    require(np.allclose(state.body("base").xpos, poses[0, :3], atol=1e-8), "Base translation is not first qpos")
    robot = poses[:, :3].copy()
    targets = {label: state.geom_xpos[model.geom(name).id].copy()
               for label, name in (("A", "scenario_target"), ("B", "memory_goal_b"))}
    bounds = np.vstack([robot[:, :2], *[point[:2] for point in targets.values()]])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:2] = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
    camera.lookat[2] = .2
    camera.azimuth, camera.elevation, camera.distance = 90, -82, 14.5
    regular = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    bold = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    font, small = ImageFont.truetype(regular, 22), ImageFont.truetype(regular, 18)
    title, target_font = ImageFont.truetype(bold, 27), ImageFont.truetype(bold, 28)
    memory_name = {"empty": "无外部记忆", "query": "按需记忆"}[read(run / "protocol.json")["mode"]]
    display_title = f"通用规划 Agent · {memory_name}"
    handoff_index = int(np.argmin(np.abs(times - handoff))) if handoff is not None else len(times) - 1
    out.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, width=width, height=view_height) as renderer:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        footprint = np.vstack([robot + [dx, dy, dz] for dx in (-.65, .65)
                               for dy in (-.55, .55) for dz in (-.3, .4)])
        for _ in range(35):
            model.vis.map.zfar = max(model.vis.map.zfar,
                (camera.distance + np.linalg.norm(np.ptp(bounds, axis=0)) + 5) / model.stat.extent)
            renderer.update_scene(state, camera=camera)
            gl_camera = mujoco.mjv_averageCamera(renderer.scene.camera[0], renderer.scene.camera[1])
            p, depth = project(footprint, gl_camera, width, view_height)
            t, td = project(np.array(list(targets.values())), gl_camera, width, view_height)
            if (p[:, 0].min() >= 12 and p[:, 0].max() < width - 12
                    and p[:, 1].min() >= 12 and p[:, 1].max() < view_height - 70
                    and depth.min() > gl_camera.frustum_near and depth.max() < gl_camera.frustum_far
                    and t[:, 0].min() > 45 and t[:, 0].max() < width - 190
                    and t[:, 1].min() > 75 and t[:, 1].max() < view_height - 45
                    and td.min() > gl_camera.frustum_near and td.max() < gl_camera.frustum_far):
                break
            camera.distance *= 1.1
        else:
            raise ValueError("Cannot fit both targets and full robot trajectory")
        screen = project(robot, gl_camera, width, view_height)[0]
        screen[:, 1] += header
        target_screen = {name: project(point, gl_camera, width, view_height)[0][0] + [0, header]
                         for name, point in targets.items()}
        camera_info = {"lookat": camera.lookat.tolist(), "distance": camera.distance,
                       "azimuth": camera.azimuth, "elevation": camera.elevation,
                       "target_pixels": {key: value.tolist() for key, value in target_screen.items()},
                       "whole_robot_trajectory_in_frame": True, "both_targets_and_labels_in_frame": True}

        def frame(index):
            state.qpos[:], state.time = poses[index], times[index]
            mujoco.mj_forward(model, state)
            renderer.update_scene(state, camera=camera)
            image = Image.new("RGB", (width, height), (18, 27, 42))
            image.paste(Image.fromarray(renderer.render()), (0, header))
            draw = ImageDraw.Draw(image)
            split = min(index, handoff_index)
            trace = screen[np.r_[np.arange(0, split, 16), split]]
            if len(trace) > 1:
                draw.line([tuple(p) for p in trace], fill=(91, 147, 244), width=3)
            if handoff is not None and index > handoff_index:
                trace = screen[np.r_[np.arange(handoff_index, index, 16), index]]
                draw.line([tuple(p) for p in trace], fill=(40, 185, 108), width=3)
            for label, color, name in (("A", (28, 87, 218), "A  蓝球"), ("B", (13, 133, 62), "B  绿方块")):
                x, y = target_screen[label]
                draw.ellipse((x-24, y-24, x+24, y+24), outline="white", width=5)
                draw.ellipse((x-22, y-22, x+22, y+22), outline=color, width=3)
                tx, ty = x+42, y-64
                box = draw.textbbox((0, 0), name, font=target_font)
                draw.line((x+17, y-17, tx, ty+24), fill="white", width=6)
                draw.line((x+17, y-17, tx, ty+24), fill=color, width=3)
                draw.rounded_rectangle((tx-10, ty-3, tx+box[2]-box[0]+13, ty+43), radius=9,
                                       fill=color, outline="white", width=2)
                draw.text((tx, ty), name, font=target_font, fill="white")
            x, y = screen[index]
            draw.rounded_rectangle((x-32, y+29, x+40, y+57), radius=5, fill=(34, 42, 52))
            draw.text((x-25, y+29), "机器人", font=small, fill=(255, 226, 94))
            returning = handoff is not None and times[index] >= handoff
            stage = "A 已通过 · 返回 B" if returning else "寻找并停靠 A"
            draw.text((18, 8), display_title, font=title, fill="white")
            draw.text((530, 13), stage, font=font, fill=(120, 236, 172) if returning else (149, 191, 255))
            draw.text((936, 14), f"仿真时间 {times[index]:.2f} s", font=font, fill="white")
            draw.text((18, height-29), "蓝线：A 验收前   绿线：A 验收后   |   全部仿真时间原速保留   |   A/B 标记和轨迹仅用于回放",
                      font=small, fill=(218, 228, 240))
            return image

        previews = [("start", 0), ("final", len(times)-1)]
        if handoff is not None:
            previews.insert(1, ("handoff", handoff_index))
        for name, index in previews:
            path = out / f"preview-{name}.png"
            if not path.exists():
                frame(index).save(path)
        if preview_only:
            print(json.dumps({"previews": [name for name, _ in previews], "camera": camera_info}, ensure_ascii=False), flush=True)
            return None
        part = output.with_suffix(".partial.mp4")
        require(not part.exists(), f"Partial output already exists: {part}")
        binary = _ffmpeg()
        process = subprocess.Popen([binary, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(part)], stdin=subprocess.PIPE)
        try:
            for number, index in enumerate(sources):
                process.stdin.write(frame(index).tobytes())
                if number % 500 == 0:
                    print(f"rendered {number}/{count}", flush=True)
        finally:
            try:
                process.stdin.close()
            finally:
                status = process.wait()
        require(status == 0, f"Video encoder failed: {status}")
    decoded = subprocess.run([binary, "-hide_banner", "-loglevel", "error", "-xerror", "-i", str(part),
        "-map", "0:v:0", "-f", "null", "-", "-progress", "pipe:1", "-nostats"],
        check=True, capture_output=True, text=True)
    frames = re.findall(r"^frame=(\d+)$", decoded.stdout, re.MULTILINE)
    require(frames and int(frames[-1]) == count, "Decoded frame count differs")
    part.replace(output)
    result = {"file": output.name, "sha256": sha(output), "frame_count": count, "fps": fps,
              "duration_s": count/fps, "width": width, "height": height, "source_capture": capture,
              "simulation_duration_s": float(times[-1]-times[0]), "initial_and_final_states_included": True,
              "entire_simulation_time_included": True, "decoded_all_frames": True,
              "annotations_are_observer_only": True, "physics_or_api_calls": False,
              "handoff_s": handoff, "handoff_source": handoff_source,
              "return_phase_exists": bool(handoff is not None and times[-1] > handoff),
              "camera": camera_info, "display_title": display_title,
              "source_exporter_sha256": sha(Path(__file__)),
              "projection_helper_sha256": sha(Path(__file__).with_name("export_full_labeled_episode_video.py"))}
    write_new(metadata, result)
    return result


def preserve(run, out):
    names = ["metrics.json", "protocol.json", "task-config.json", "events.jsonl", "initial_state.npz",
             "trajectory.jsonl", "evaluation.evaluator-only.json", "memory.json", "memory.md"]
    required = set(names) - {"memory.json", "memory.md"}
    selected = []
    for name in names:
        path = run / name
        require(path.exists() or name not in required, f"Missing final run artifact: {name}")
        if path.exists():
            selected.append(path)
    selected += [run / "capture" / name for name in ("capture.json", "model.mjb", "poses.npz")]
    selected += sorted((run / "images").glob("*.png"))
    for source in selected:
        copy_new(source, out / "episode" / source.relative_to(run))
    sources = [(path, "episode/" + path.relative_to(run).as_posix()) for path in selected]
    sources += [(path, path.relative_to(out).as_posix()) for path in sorted((out / "source_snapshot").rglob("*")) if path.is_file()]
    sources += [(out/name, name) for name in ("experiment.json", "frozen-source.json") if (out/name).exists()]
    archive = out / "full-replay-data.zip"
    if not archive.exists():
        part = archive.with_suffix(".partial.zip")
        require(not part.exists(), "Partial replay archive already exists")
        with ZipFile(part, "x", compression=ZIP_DEFLATED, compresslevel=5) as zipped:
            for source, name in sources:
                require(not source.is_symlink(), f"Symlink refused: {source}")
                zipped.write(source, name)
        part.replace(archive)
    with ZipFile(archive) as zipped:
        require(zipped.testzip() is None, "Replay archive CRC failure")
        require(set(zipped.namelist()) == {name for _, name in sources}, "Replay archive membership differs")
        for source, name in sources:
            with zipped.open(name) as stream:
                require(hashlib.file_digest(stream, "sha256").hexdigest() == sha(source), f"Replay zip hash differs: {name}")
    return {"file": archive.name, "sha256": sha(archive), "bytes": archive.stat().st_size,
            "members": len(sources), "crc_and_member_hashes_verified": True}


def finalize(out):
    result = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "archive-integrity.json":
            require(not path.is_symlink(), f"Symlink refused: {path}")
            require(".partial." not in path.name, f"Incomplete artifact remains: {path}")
            result.append({"path": path.relative_to(out).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
    write_new(out / "archive-integrity.json", result)
    return {"files_hashed": len(result), "total_bytes": sum(row["bytes"] for row in result)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fps", type=float, default=12.5)
    p.add_argument("--preview-only", action="store_true")
    p.add_argument("--pack-only", action="store_true", help="Audit and copy raw data, without rendering or final integrity")
    p.add_argument("--finalize-only", action="store_true", help="Hash final artifacts after adding the report and independent audit")
    args = p.parse_args()
    require(math.isfinite(args.fps) and 0 < args.fps <= 30, "Invalid output fps")
    require(sum((args.preview_only, args.pack_only, args.finalize_only)) <= 1, "Choose at most one stage")
    if args.finalize_only:
        print(json.dumps(finalize(args.output), ensure_ascii=False), flush=True)
        return
    verification = audit(args.run, args.output)
    if args.preview_only:
        render(args.run, args.output, fps=args.fps, preview_only=True)
        return
    write_new(args.output / "archive-audit.json", verification)
    exporter = args.output / "exporter_snapshot/archive_general_episode.py"
    if exporter.exists() and sha(exporter) != sha(Path(__file__)):
        # Keep the exact renderer used by an existing video when only archive
        # metadata handling is subsequently repaired.
        exporter = exporter.with_name("archive_general_episode-" + sha(Path(__file__))[:12] + ".py")
    copy_new(Path(__file__), exporter)
    copy_new(Path(__file__).with_name("export_full_labeled_episode_video.py"),
             args.output / "exporter_snapshot/export_full_labeled_episode_video.py")
    bundle = preserve(args.run, args.output)
    write_new(args.output / "replay-package.json", bundle)
    if args.pack_only:
        print(json.dumps(bundle, ensure_ascii=False), flush=True)
        return
    video = render(args.run, args.output, fps=args.fps)
    write_new(args.output / "summary.json", {"complete": True, "audit": verification, "video": video, "replay_package": bundle})
    print(json.dumps({"output": str(args.output), "requests": verification["requests_total"],
                      "video_duration_s": video["duration_s"], "zip_bytes": bundle["bytes"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
