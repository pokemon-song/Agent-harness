"""Offline report and optional continuous-state video for one memory round trip."""
from __future__ import annotations

import argparse
from datetime import datetime
import html
import json
import os
from pathlib import Path, PurePosixPath
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_complex_memory_mission import local_file, number, read_json, sha256, verdict
from scripts.report_md_complex_memory import count, event_rows, known_usage, role_stats, snapshot_info

PHASES = ("outbound", "return")
LABELS = {"outbound": "去程：绕柜、过门、到 B", "return": "返程：读取去程记忆并原路返回"}
OUTPUTS = ("REPORT.zh-CN.md", "analysis.json", "watch.html")
RETURN_FIELDS = (
    "success", "geometry_success", "returned_to_start", "state_valid",
    "return_distance_to_start_m", "return_tolerance_m", "retrace_tolerance_m",
    "outbound_path_length_m", "return_path_length_m", "retrace_max_deviation_m",
    "retrace_mean_deviation_m", "retrace_p95_deviation_m", "outbound_coverage",
    "return_corridor_fraction", "outbound_stages_in_order", "return_stages_in_order",
    "same_cabinet_side", "retrace_corridor_passed", "boundaries_continuous", "invalid_path",
    "invalid_samples", "heading_required", "stopped", "speed_mps", "yaw_rate_radps",
    "fallen", "collision", "collision_physics_steps", "source", "events",
)


def render_continuous(directory):
    """Reconstruct the single archive; never execute physics or insert reset cards."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from scripts import render_memory_mission_video as video

    previous_camera, previous_labels = video.fixed_camera, video.LABELS

    def camera(mujoco, phase):
        value = mujoco.MjvCamera()
        value.type = mujoco.mjtCamera.mjCAMERA_FREE
        value.lookat[:] = [2.65, 0., .25]
        value.distance, value.azimuth, value.elevation = 9., 45., -65.
        return value

    try:
        video.fixed_camera = camera
        video.LABELS = {**previous_labels, "roundtrip": "SINGLE SHIFT: OUTBOUND + MEMORY-GUIDED RETURN"}
        return video.render_phase(Path(directory) / "third_person", "roundtrip", video.ffmpeg_path(),
                                  timebases=("sim",), overwrite=False)
    finally:
        video.fixed_camera, video.LABELS = previous_camera, previous_labels


def verified_video(directory, warnings):
    manifest = directory / "third_person/video-manifest.json"
    if not manifest.exists():
        return None
    value = read_json(local_file(directory, manifest.relative_to(directory)))
    hashes = value.get("input_sha256") or {}
    for filename in ("model.mjb", "states.npz", "metadata.json"):
        source = local_file(directory, Path("third_person") / filename)
        if hashes.get(filename) != sha256(source):
            raise ValueError("Continuous video does not match its source archive")
    item = (value.get("videos") or {}).get("sim")
    if not isinstance(item, dict):
        warnings.append("录像清单未包含仿真时间视频")
        return None
    if (item.get("decoded_all_frames") is not True
            or item.get("first_and_final_states_included") is not True):
        warnings.append("录像未记录完整解码与起末状态覆盖，不作为完整录像展示")
        return None
    if item.get("file") != "third-person-sim.mp4":
        raise ValueError("Unexpected continuous video filename")
    path = local_file(directory, "third_person/third-person-sim.mp4")
    if sha256(path) != item.get("sha256"):
        raise ValueError("Continuous video checksum mismatch")
    return {**item, "src": path.relative_to(directory).as_posix(),
            "manifest": manifest.relative_to(directory).as_posix(), "timebase": "sim",
            "state_reconstruction": value.get("state_reconstruction"), "reset_cards": False}


def role_accounting(stats, events, request_event, *, enabled=True):
    result = role_stats(stats, events, request_event, enabled=enabled)
    # A transport failure can have both model_error and worker failure records.
    # Report their roles separately instead of counting both as API failures.
    result.pop("errors")
    seen, request_errors, lifecycle = set(), [], []
    for event in events:
        if event.get("event") == "model_error":
            key = (event.get("turn"), event.get("attempt"))
            if key not in seen:
                request_errors.append({"turn": key[0], "attempt": key[1],
                    "code": (event.get("error") or {}).get("code"), "latency_s": event.get("latency_s")})
                seen.add(key)
        elif event.get("event") == "failure":
            lifecycle.append(event.get("error"))
    result.update(request_errors=request_errors, worker_lifecycle_errors=lifecycle)
    if isinstance(stats, dict):
        for key in ("preempted_batches", "superseded_errors", "event_bytes_original", "event_bytes_projected"):
            if key in stats:
                result[key] = stats[key]
    return result


def memory_versions(directory):
    versions = []
    for path in sorted((directory / "memory/revisions").glob("*/manifest.json")):
        manifest = read_json(local_file(directory, path.relative_to(directory)))
        changed = []
        for item in manifest.get("files", []):
            if item.get("revision") != manifest.get("revision"):
                continue
            relative = PurePosixPath(item["path"])
            if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".md":
                raise ValueError("Nonlocal memory document path")
            document = local_file(directory, path.parent.relative_to(directory) / relative)
            if not document.is_relative_to(path.parent.resolve()) or sha256(document) != item.get("sha256"):
                raise ValueError("Published memory document checksum/path mismatch")
            changed.append({"path": item["path"], "href": document.relative_to(directory).as_posix(),
                "sha256": item["sha256"], "revision": item.get("revision"), "status": item.get("status"),
                "evidence_count": len(item.get("evidence_ids", []))})
        versions.append({"revision": manifest.get("revision"),
            "processed_event_seq": manifest.get("processed_event_seq"),
            "acknowledged_event_seqs": manifest.get("acknowledged_event_seqs", []),
            "manifest": path.relative_to(directory).as_posix(), "changed_documents": changed})
    return versions


def inspect_phase(directory, row):
    phase = row.get("phase")
    if phase not in PHASES:
        raise ValueError("Unknown round-trip phase")
    base, warnings = directory / phase, []
    metrics = row.get("metrics") or {}
    archived = base / "metrics.json"
    if archived.exists():
        if read_json(local_file(directory, archived.relative_to(directory))) != metrics:
            raise ValueError("Summary and archived phase metrics disagree")
    else:
        warnings.append("没有归档 metrics.json；保留汇总记录的有限结果")
    events = event_rows(base / "events.jsonl", warnings)
    memory_events = event_rows(base / "memory-agent.jsonl", warnings)
    planner = role_accounting({"api_requests": metrics.get("model_calls"),
        "completed_responses": metrics.get("completed_responses"), "usage": metrics.get("usage")}, events, "model_input")
    memory = role_accounting(metrics.get("memory_agent"), memory_events, "model_request",
                             enabled=metrics.get("memory_mode") != "read_only")
    reads, actions, errors, frames = [], [], [], []
    for role, log in (("planner", events), ("memory", memory_events)):
        for event in log:
            if event.get("event") == "tool_results":
                for result in event.get("results", []):
                    if result.get("status") != "succeeded":
                        errors.append({"role": role, "turn": event.get("turn"), "attempt": event.get("attempt"),
                                       "tool": result.get("tool"), "code": result.get("error_code")})
                    if role == "planner" and result.get("tool") == "memory_read" and result.get("status") == "succeeded":
                        document = result.get("document") or {}
                        reads.append({"turn": event.get("turn"), **{key: document.get(key)
                            for key in ("path", "revision", "snapshot_revision", "sha256")}})
                    if role == "planner" and result.get("tool") in {"move_relative", "turn"} and result.get("status") == "succeeded":
                        actions.append(event.get("turn"))
            if role == "planner" and event.get("event") == "model_input" and isinstance(event.get("image_path"), str):
                relative = Path(phase) / event["image_path"]
                try:
                    local_file(directory, relative)
                    frames.append({"src": relative.as_posix(), "phase": phase, "turn": event.get("turn")})
                except ValueError:
                    warnings.append("观测图片缺失或越界，未加入播放页面")
    worker = row.get("worker") or {}
    strict = (metrics.get("success") is True and not row.get("error") and not metrics.get("cleanup_errors")
              and not worker.get("error") and not worker.get("cleanup_errors")
              and worker.get("readonly_memory_unchanged") is not False
              and metrics.get("memory_agent_stopped") is not False and row.get("returncode", 0) == 0)
    evaluation = metrics.get("evaluation") or {}
    first_motion = min(actions) if actions and all(type(t) is int for t in actions) else None
    read_turns = [r["turn"] for r in reads if type(r["turn"]) is int]
    return {"phase": phase, "label": LABELS[phase], "success": strict,
        "termination": metrics.get("termination"), "error": metrics.get("error"),
        "wall_elapsed_s": metrics.get("wall_elapsed_s"), "sim_elapsed_s": metrics.get("sim_elapsed_s"),
        "path_length_m": metrics.get("path_length_m"), "model_wait_s": metrics.get("model_wait_s"),
        "memory_wait_s": metrics.get("memory_wait_s"), "action_execution_wall_s": metrics.get("action_execution_wall_s"),
        "planner": planner, "memory_agent": memory, "memory_mode": metrics.get("memory_mode"),
        "snapshots": {label: snapshot_info(base, label) for label in ("initial", "final")},
        "memory_reads": reads, "first_successful_motion_turn": first_motion,
        "read_before_first_motion": min(read_turns) < first_motion if read_turns and first_motion is not None else None,
        "tool_errors": errors, "evaluation": evaluation,
        "return_evaluation": {key: evaluation[key] for key in RETURN_FIELDS if key in evaluation} if phase == "return" else None,
        "artifact_warnings": warnings, "frames": frames}


def build_report(directory, *, render=False):
    directory = Path(directory).resolve()
    outputs = [directory / name for name in OUTPUTS]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Round-trip reports already exist; refusing to overwrite")
    before = {path.relative_to(directory).as_posix(): sha256(path)
              for path in sorted(directory.rglob("*")) if path.is_file()}
    summary = read_json(local_file(directory, "summary.json"))
    config = read_json(local_file(directory, "experiment.json"))
    marker = summary.get("experiment_kind", config.get("experiment_kind", ""))
    if not isinstance(marker, str) or "roundtrip" not in marker:
        raise ValueError("This report requires roundtrip experiment artifacts")
    phases = [inspect_phase(directory, row) for row in summary.get("phases", [])]
    by_phase = {phase["phase"]: phase for phase in phases}
    if len(phases) != len(by_phase):
        raise ValueError("Duplicate round-trip phase")
    warnings = []
    render_error = None
    if render:
        try:
            render_continuous(directory)
        except FileExistsError:
            raise  # Never overwrite an earlier render by accident.
        except Exception as exc:
            render_error = {"type": type(exc).__name__}
            warnings.append("连续录像渲染未完成：" + type(exc).__name__ + "；保留失败实验报告")
    try:
        clip = verified_video(directory, warnings)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        clip = None
        warnings.append("录像验证失败：" + type(exc).__name__ + "；未将该视频作为已验证录像展示")
    versions = memory_versions(directory)
    document_links = {(item["path"], item["sha256"]): item["href"]
                      for revision in versions for item in revision["changed_documents"]}
    for phase in phases:
        for read in phase["memory_reads"]:
            read["href"] = document_links.get((read["path"], read["sha256"]))
    roles = [phase[key] for phase in phases for key in ("planner", "memory_agent")]
    attempts = sum(role["attempts"] for role in roles) if roles and all(role["attempts"] is not None for role in roles) else None
    responses = sum(role["completed_responses"] for role in roles) if roles and all(
        role["completed_responses"] is not None for role in roles) else None
    known = {key: sum(role["known_usage"].get(key, 0) for role in roles)
             for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    unknown = sum(role["unknown_usage_attempts"] for role in roles) if roles and all(role["unknown_usage_attempts"] is not None for role in roles) else None
    usage_complete = bool(roles) and all(role["usage_complete"] for role in roles)
    for key, computed in (("api_requests", attempts), ("complete_responses", responses),
                          ("known_usage", known), ("unknown_usage_attempts", unknown)):
        if key in summary and summary[key] != computed:
            warnings.append(f"汇总 {key} 与逐角色日志统计不一致，保留两者")
    checks = summary.get("checks") or {}
    return_phase = by_phase.get("return", {})
    success = (summary.get("mission_success") is True and all(by_phase.get(p, {}).get("success") is True for p in PHASES)
               and not summary.get("error") and not summary.get("cleanup_errors")
               and not summary.get("accounting_errors") and not any(value is False for value in checks.values())
               and bool(return_phase.get("memory_reads")) and return_phase.get("read_before_first_motion") is True)
    if summary.get("mission_success") is True and not success:
        warnings.append("汇总通过标志未满足逐段成功或完整性条件，不报告通过")
    not_run = [phase for phase in PHASES if phase not in by_phase]
    headline = "去程到 B 并原路返回通过。" if success else "本轮往返任务未通过；失败与未运行项见逐段结果。"
    is_llm_result = summary.get("is_llm_result", config.get("is_llm_result")) is True
    analysis = {"experiment_kind": marker, "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "experiment": directory.name, "mission_success": success, "reported_mission_success": summary.get("mission_success"),
        "is_llm_result": is_llm_result,
        "simulator_used": summary.get("simulator_used", config.get("simulator_used")),
        "git": config.get("git"), "model": config.get("model"), "phases": phases, "not_run": not_run,
        "api_requests": attempts if is_llm_result else 0, "model_request_attempts": attempts,
        "reported_api_requests": summary.get("api_requests"),
        "complete_responses": responses, "reported_complete_responses": summary.get("complete_responses"),
        "known_usage": known, "reported_known_usage": summary.get("known_usage"),
        "unknown_usage_attempts": unknown, "reported_unknown_usage_attempts": summary.get("unknown_usage_attempts"),
        "usage_complete": usage_complete, "checks": checks, "return_not_run_reason": summary.get("return_not_run_reason"),
        "error": summary.get("error"), "cleanup_errors": summary.get("cleanup_errors"),
        "accounting_errors": summary.get("accounting_errors"),
        "coordinate_frame_id": summary.get("coordinate_frame_id", config.get("coordinate_frame_id")),
        "wall_elapsed_s": summary.get("wall_elapsed_s"), "sim_elapsed_s": summary.get("sim_elapsed_s"),
        "path_length_m": summary.get("path_length_m"),
        "memory_versions": versions, "video": clip, "render_error": render_error, "warnings": warnings,
        "source_artifact_sha256": before, "report_generation_api_requests": 0,
        "causal_memory_benefit_established": False}
    table = ["| 阶段 | 结果 | 终止/错误 | 墙钟秒 | 仿真秒 | 路径米 | 规划/记忆请求 |",
             "|---|---|---|---:|---:|---:|---:|"]
    for phase in phases:
        table.append(f"| {phase['label']} | {verdict(phase['success'])} | "
            f"`{(phase['error'] or {}).get('code') or phase['termination']}` | {number(phase['wall_elapsed_s'])} | "
            f"{number(phase['sim_elapsed_s'])} | {number(phase['path_length_m'], digits=3)} | "
            f"{phase['planner']['attempts']}/{phase['memory_agent']['attempts']} |")
    for phase in not_run:
        table.append(f"| {LABELS[phase]} | 未运行 | — | — | — | — | — |")
    report = ["# 单班导航记忆：去 B 后原路返回", "", headline, "",
        f"真实模型结果：{analysis['is_llm_result']}；模型：`{config.get('model')}`；冻结提交：`{(config.get('git') or {}).get('commit')}`。", "",
        "本实验设计为同一个机器人世界中的连续往返：交接时清空规划模型上下文，保留机器人状态和同一个起点坐标系。"
        "返程从实际发布的去程 MD 检索路线，不由程序填充返程答案。实际连续性和只读检查见下方 checks；模型读取文件不等于已证明记忆的因果收益。", "",
        "[播放页面](watch.html) · [原始汇总](summary.json) · [冻结配置](experiment.json) · [结构化分析](analysis.json)", "",
        "声明预算：`" + json.dumps({key: config.get(key) for key in (
            "budgets", "memory_calls", "api_timeout_s", "max_output_tokens", "history_turns")}, ensure_ascii=False) + "`。", "",
        *table, "", f"请求合计：{attempts}，完整回复：{responses}；已知用量：`{json.dumps(known, ensure_ascii=False)}`；"
        f"用量未知尝试：{unknown}；用量完整：{usage_complete}。失败请求缺失的用量不填零，不按输出上限推算费用。", ""]
    if not is_llm_result:
        report += ["本轮标记为 fixture：上述是脚本模型请求与合成用量，真实模型 API 调用为0，不作为大模型成绩或真实费用。", ""]
    if clip:
        report += [f"[完整连续第三视角录像]({clip['src']})：{number(clip['duration_s'], '秒')}，"
            f"{clip.get('frame_count')}帧。按仿真时间播放，推理暂停从时轴中移除；没有 RESET 卡，不重新执行物理仿真。", ""]
    else:
        report += ["尚无经过源哈希和完整解码记录核验的视频；播放页仅显示已保存的真实关键帧，不能将其称为连续录像。", ""]
    for phase in phases:
        report += ["## " + phase["label"], ""]
        for label, role in (("规划 Agent", phase["planner"]), ("记忆 Agent", phase["memory_agent"])):
            report += [f"- {label}：请求{role['attempts']}，完整回复{role['completed_responses']}，未知用量尝试{role['unknown_usage_attempts']}；"
                f"已知用量 `{json.dumps(role['known_usage'], ensure_ascii=False)}`。",
                f"  请求错误：`{json.dumps(role['request_errors'], ensure_ascii=False)}`；"
                f"worker 生命周期错误：`{json.dumps(role['worker_lifecycle_errors'], ensure_ascii=False)}`。"]
        report += ["", f"模型等待：{number(phase['model_wait_s'], '秒')}；记忆等待：{number(phase['memory_wait_s'], '秒')}；"
            f"工具执行墙钟：{number(phase['action_execution_wall_s'], '秒')}。", "",
            f"实际成功读取 MD {len(phase['memory_reads'])}次；首个成功运动前已有读取：{phase['read_before_first_motion']}。"
            "这核对读工具记录，不自动证明路线语义正确或规划确实采用了每一条记忆。", ""]
        for read in phase["memory_reads"]:
            label = f"{read['path']}（文档v{read['revision']}，快照v{read['snapshot_revision']}，turn={read['turn']}）"
            report.append("- " + (f"[{label}]({read['href']})" if read["href"] else label + "；未找到对应归档链接"))
        report += ["", "记忆初末快照：`" + json.dumps(phase["snapshots"], ensure_ascii=False) + "`。", ""]
        if phase["phase"] == "return":
            report += ["返程评分直接取自独立评估器，以下不是从 MD 推测的成绩：", "",
                "```json", json.dumps(phase["return_evaluation"], ensure_ascii=False, indent=2), "```", "",
                "coverage、轨迹偏离、过门次序、绕柜侧和回到起点均按评估器记录解释；缺失字段表示未记录，不按通过处理。", ""]
        else:
            report += ["物理评分：`" + json.dumps(phase["evaluation"], ensure_ascii=False) + "`。", ""]
        if phase["tool_errors"]:
            report += ["工具错误：`" + json.dumps(phase["tool_errors"], ensure_ascii=False) + "`。", ""]
        if phase["artifact_warnings"]:
            report += ["工件限制：" + "；".join(phase["artifact_warnings"]) + "。", ""]
    report += ["## 记忆与一致性", "", "检查结果：`" + json.dumps(checks, ensure_ascii=False) + "`。", ""]
    for revision in versions:
        links = [f"[{item['path']}]({item['href']})" for item in revision["changed_documents"]]
        report.append(f"- [MD v{revision['revision']}]({revision['manifest']})：连续水位{revision['processed_event_seq']}，"
            f"稀疏ACK={revision['acknowledged_event_seqs']}；" + ("、".join(links) or "没有新增或修订正文") + "。")
    report += ["", "连续水位与稀疏ACK分别表示连续已处理前缀和越过积压后单独确认的事件；文件发布和证据引用不保证语义完全正确。", "",
        "未运行阶段：" + ("、".join(not_run) or "无") + "；返程未运行原因：`" + str(summary.get("return_not_run_reason")) + "`。", "",
        "实验错误：`" + json.dumps(analysis["error"], ensure_ascii=False) + "`；清理错误：`" +
        json.dumps(analysis["cleanup_errors"], ensure_ascii=False) + "`；计数错误：`" +
        json.dumps(analysis["accounting_errors"], ensure_ascii=False) + "`。", "",
        "本轮没有自动获得无记忆对照；即便往返成功，也只证明本次配置完成了任务，不能单凭一次成功证明记忆带来的提升。", ""]
    if warnings:
        report += ["统计警告：" + "；".join(warnings) + "。", ""]
    frames = [frame for phase in phases for frame in phase["frames"]]
    payload = json.dumps({"frames": frames}, ensure_ascii=False)
    for old, new in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"), ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        payload = payload.replace(old, new)
    media = (f'<video controls preload="metadata" src="{html.escape(clip["src"], quote=True)}"></video>' if clip
             else '<p>没有已验证连续视频；下方是离散真实观测帧。</p>')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>单班导航记忆往返</title><style>body{{max-width:1100px;margin:28px auto;padding:0 18px;font:16px/1.7 system-ui}}video,img{{width:100%;max-height:75vh;object-fit:contain;background:#111}}pre{{white-space:pre-wrap}}input{{width:100%}}</style>
<h1>去 B 后原路返回</h1><p>{html.escape(headline)}</p>{media}
<p>单连续世界；仿真时间录像不插入 RESET，推理等待从时轴移除。</p>
<details {'open' if not clip else ''}><summary>实际观测关键帧</summary><input id="frame" type="range" min="0" max="{max(0,len(frames)-1)}" value="0"><p id="label"></p><img id="image" alt="实际机器人观测"></details>
<p><a href="REPORT.zh-CN.md">实验报告</a> · <a href="analysis.json">分析数据</a> · <a href="summary.json">原始结果</a></p>
<pre>{html.escape(chr(10).join(table))}</pre><script>const data={payload};const slider=document.getElementById('frame'),image=document.getElementById('image'),label=document.getElementById('label');function show(){{const f=data.frames[Number(slider.value)];if(!f){{image.hidden=true;slider.disabled=true;label.textContent='没有已保存的观测帧';return;}}image.src=f.src;label.textContent=f.phase+' · turn '+f.turn;}}slider.oninput=show;show();</script></html>'''
    def assert_unchanged():
        if any(not (directory / relative).is_file() or sha256(directory / relative) != digest
               for relative, digest in before.items()):
            raise RuntimeError("Raw artifacts changed during report generation")
    assert_unchanged()
    for path, content in zip(outputs, ("\n".join(report), json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n", page)):
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
    assert_unchanged()
    return {"outputs": [str(path) for path in outputs], "mission_success": success,
            "api_requests": analysis["api_requests"], "model_request_attempts": attempts,
            "known_usage": known, "unknown_usage_attempts": unknown, "usage_complete": usage_complete,
            "video_verified": clip is not None, "source_artifacts_unchanged": len(before), "warnings": warnings}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--render", action="store_true", help="Render one continuous simulation-time movie before reporting")
    args = parser.parse_args(argv)
    print(json.dumps(build_report(args.directory, render=args.render), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
