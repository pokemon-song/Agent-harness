"""Offline report for MD-memory complex missions; never calls a model or edits evidence."""
from __future__ import annotations

import argparse
from datetime import datetime
import html
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_complex_memory_mission import (
    LABELS, PHASES, PRIMARY, RECEIPT_FLAGS, ROUTE_FLAGS,
    local_file, number, read_json, sha256, verdict, videos,
)

USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def count(value):
    return value if type(value) is int and value >= 0 else None


def known_usage(value):
    return {key: value[key] for key in USAGE_KEYS
            if isinstance(value, dict) and count(value.get(key)) is not None}


def event_rows(path, warnings):
    if not path.is_file():
        return []
    events = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            if line.strip():
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("An event must be an object")
                events.append(event)
        except (ValueError, TypeError):
            warnings.append(f"{path.name} 第{line_number}行无法读取；统计仅覆盖已读取部分")
            break
    return events


def role_stats(stats, events, request_event, *, enabled=True):
    stats = stats if isinstance(stats, dict) else {}
    responses = [e for e in events if e.get("event") == "model_response"]
    logged_requests = sum(e.get("event") == request_event for e in events)
    attempts = count(stats.get("api_requests"))
    if attempts is None:
        attempts = logged_requests if events or not enabled else None
    completed = count(stats.get("completed_responses"))
    if completed is None:
        completed = len(responses) if events or not enabled else None
    usage = known_usage(stats.get("usage"))
    if not usage and responses:
        for response in responses:
            for key, value in known_usage(response.get("usage")).items():
                usage[key] = usage.get(key, 0) + value
    known_attempts = sum(all(count(e.get("usage", {}).get(k)) is not None for k in USAGE_KEYS)
                         for e in responses)
    complete = (attempts is not None and attempts == completed == len(responses) == known_attempts
                and logged_requests == attempts)
    errors = []
    for event in events:
        if event.get("event") in {"model_error", "failure"}:
            source = event.get("error") if isinstance(event.get("error"), dict) else event
            errors.append({key: source.get(key) for key in ("type", "code")})
    return {"attempts": attempts, "completed_responses": completed, "logged_requests": logged_requests,
            "logged_responses": len(responses), "known_usage": usage, "usage_complete": complete,
            "unknown_usage_attempts": max(0, attempts-known_attempts) if attempts is not None else None,
            "errors": errors, "committed_batches": count(stats.get("committed_batches"))}


def snapshot_info(base, label):
    for name in (f"md-memory-{label}.json", f"memory-{label}.json"):
        path = base / name
        if path.is_file():
            value = read_json(path)
            return {"revision": value.get("revision"), "scope": value.get("scope"),
                    "files": len(value.get("files", [])), "processed_event_seq": value.get("processed_event_seq"),
                    "latest_event_seq": value.get("latest_event_seq"),
                    "acknowledged_event_seqs": value.get("acknowledged_event_seqs", []),
                    "pending_event_count": value.get("pending_event_count",
                        value.get("latest_event_seq", 0) - value.get("processed_event_seq", 0)),
                    "processed_critical_event_seq": value.get("processed_critical_event_seq"),
                    "pending_critical_events": len(value.get("pending_critical_events", [])),
                    "artifact": f"{base.name}/{name}"}
    return None


def memory_versions(directory, phase):
    """Link actual immutable Markdown files without copying their semantic contents."""
    result = []
    for path in sorted((directory / "memory" / phase / "revisions").glob("*/manifest.json")):
        manifest = read_json(local_file(directory, path.relative_to(directory)))
        revision = manifest.get("revision")
        files = []
        for item in manifest.get("files", []):
            if item.get("revision") != revision:
                continue
            document = local_file(directory, path.parent.relative_to(directory) / item["path"])
            if sha256(document) != item.get("sha256"):
                raise ValueError("Published Markdown hash changed: " + str(document))
            files.append({"path": item["path"], "href": document.relative_to(directory).as_posix(),
                          "status": item.get("status"), "sha256": item["sha256"],
                          "evidence_count": len(item.get("evidence_ids", []))})
        result.append({"revision": revision, "processed_event_seq": manifest.get("processed_event_seq"),
                       "manifest": path.relative_to(directory).as_posix(), "changed_documents": files})
    return result


def inspect_phase(directory, result):
    phase = result.get("phase")
    if phase not in PHASES:
        raise ValueError("Unknown phase in MD complex report")
    base = directory / phase
    metrics = result.get("metrics") or {}
    if (base / "metrics.json").exists() and read_json(base / "metrics.json") != metrics:
        raise ValueError("Summary disagrees with archived metrics: " + phase)
    warnings = []
    events = event_rows(base / "events.jsonl", warnings)
    memory_log = base / "memory-agent.jsonl"
    if not memory_log.exists():
        memory_log = base / "memory-events.jsonl"
    memory_events = event_rows(memory_log, warnings)
    planner_stats = {"api_requests": metrics.get("model_calls"),
                     "completed_responses": metrics.get("completed_responses"), "usage": metrics.get("usage")}
    planner = role_stats(planner_stats, events, "model_input")
    memory = role_stats(metrics.get("memory_agent"), memory_events, "model_request",
                        enabled=metrics.get("memory_mode") != "read_only")
    attempts = (planner["attempts"] + memory["attempts"]
                if planner["attempts"] is not None and memory["attempts"] is not None else None)
    if count(metrics.get("api_requests_total")) is not None and attempts != metrics["api_requests_total"]:
        warnings.append("逐角色请求数与 metrics.api_requests_total 不一致")
    path_length, path_source = metrics.get("path_length_m"), "metrics"
    trajectory = event_rows(base / "trajectory.jsonl", warnings)
    if trajectory:
        try:
            anchor = next(e["base_xy_m"] for e in events if e.get("event") == "episode_start")
            points = [anchor, *[row["base_xyz_m"][:2] for row in trajectory]]
            if not all(len(p) == 2 and all(type(v) in (float, int) and math.isfinite(v) for v in p) for p in points):
                raise ValueError("Invalid trajectory")
            path_length, path_source = sum(math.dist(a, b) for a, b in zip(points, points[1:])), "trajectory"
        except (StopIteration, ValueError, TypeError, KeyError):
            warnings.append("轨迹或初始锚点不完整，无法独立重算路径")
    errors, reads, frames = [], [], []
    for role, rows in (("planner", events), ("memory", memory_events)):
        for event in rows:
            if event.get("event") == "tool_results":
                for item in event.get("results", []):
                    entry = {"role": role, "turn": event.get("turn"), "attempt": event.get("attempt"),
                             "tool": item.get("tool"), "status": item.get("status")}
                    if item.get("status") != "succeeded":
                        errors.append({**entry, "error_code": item.get("error_code")})
                    if role == "planner" and item.get("tool") == "memory_read" and item.get("status") == "succeeded":
                        doc = item.get("document") or item.get("memory") or {}
                        reads.append({**entry, **{k: doc.get(k) for k in ("path", "revision", "snapshot_revision", "sha256", "status")}})
            relative = None
            if role == "planner" and event.get("event") == "model_input" and isinstance(event.get("image_path"), str):
                relative = Path(phase) / event["image_path"]
            image = event.get("image") or {}
            if role == "memory" and event.get("event") == "model_request" and isinstance(image.get("path"), str):
                relative = Path("memory") / phase / image["path"]
            if relative is not None:
                try:
                    local_file(directory, relative)
                    frames.append({"src": relative.as_posix(), "label": f"{LABELS[phase]} · {role}",
                                   "turn": event.get("turn", event.get("attempt", 0))})
                except ValueError:
                    warnings.append("公开图像缺失或不在实验目录中，未生成播放链接")
    evaluation = metrics.get("evaluation") or {}
    extension = evaluation.get("task_extension") or {}
    receipt = {key: extension.get(key) for key in RECEIPT_FLAGS} if extension else None
    if receipt is not None:
        receipt["route_evaluation"] = {key: (extension.get("route_evaluation") or {}).get(key) for key in ROUTE_FLAGS}
        receipt["submission"] = {key: value for key, value in (extension.get("submission") or {}).items()
                                 if key in {"dispatch_revision", "destination", "route"}}
    worker = result.get("worker") or {}
    strict = (metrics.get("success") is True and isinstance(result.get("worker"), dict)
              and not worker.get("error") and not worker.get("cleanup_errors") and not result.get("artifact_error")
              and worker.get("readonly_memory_unchanged") is not False
              and metrics.get("memory_agent_stopped") is not False and result.get("returncode", 0) == 0)
    return {"phase": phase, "label": LABELS[phase], "strict_success": strict,
            "termination": metrics.get("termination"), "error_code": (metrics.get("error") or {}).get("code"),
            "wall_elapsed_s": metrics.get("wall_elapsed_s"), "sim_elapsed_s": metrics.get("sim_elapsed_s"),
            "path_length_m": path_length, "path_length_source": path_source, "api_requests": attempts,
            "planner": planner, "memory_agent": memory, "usage_complete": planner["usage_complete"] and memory["usage_complete"],
            "snapshots": {key: snapshot_info(base, key) for key in ("initial", "final")},
            "memory_mode": metrics.get("memory_mode"), "memory_versions": memory_versions(directory, phase),
            "memory_reads": reads, "tool_errors": errors, "receipt": receipt,
            "evaluation": {key: evaluation.get(key) for key in ("collision", "fallen", "stopped", "stages", "stage_completion_rate")},
            "has_worker_error": bool(worker.get("error")), "has_cleanup_error": bool(worker.get("cleanup_errors")),
            "has_integrity_error": worker.get("readonly_memory_unchanged") is False,
            "memory_before_digest": result.get("memory_before_digest", worker.get("memory_before_digest")),
            "memory_after_digest": result.get("memory_after_digest", worker.get("memory_after_digest")),
            "artifact_warnings": warnings, "frames": frames}


def build_report(directory):
    directory = Path(directory).resolve()
    outputs = [directory / name for name in ("REPORT.zh-CN.md", "analysis.json", "watch.html")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Report outputs already exist; refusing to overwrite them")
    before = {p.relative_to(directory).as_posix(): sha256(p) for p in sorted(directory.rglob("*")) if p.is_file()}
    summary = read_json(local_file(directory, "summary.json"))
    config = read_json(local_file(directory, "experiment.json"))
    if summary.get("experiment_kind") != "md-complex-memory-v1":
        raise ValueError("This report requires md-complex-memory-v1 artifacts")
    phases = [inspect_phase(directory, item) for item in summary.get("phases", [])]
    by_phase = {p["phase"]: p for p in phases}
    if len(phases) != len(by_phase):
        raise ValueError("Duplicate phase results")
    strict_primary = all(by_phase.get(p, {}).get("strict_success") is True for p in PRIMARY)
    success = summary.get("mission_success") is True and strict_primary
    not_run = [phase for phase in PHASES if phase not in by_phase]
    warnings = []
    if (summary.get("mission_success") is True) != strict_primary:
        warnings.append("汇总成功标志与逐阶段严格结果不一致，不将其计为通过")
    known = {}
    for phase in phases:
        for role in (phase["planner"], phase["memory_agent"]):
            for key, value in role["known_usage"].items():
                known[key] = known.get(key, 0) + value
    declared_usage = known_usage(summary.get("known_usage"))
    if declared_usage and declared_usage != known:
        warnings.append("汇总 known_usage 与逐角色已记录用量不一致，以下保留两者")
    attempts = sum(p["api_requests"] for p in phases) if all(p["api_requests"] is not None for p in phases) else None
    if count(summary.get("api_requests")) is not None and attempts != summary["api_requests"]:
        warnings.append("汇总 API 次数与逐角色请求数不一致，以下保留两者")
    _, clips = videos(directory)
    failures = [LABELS[p] for p in PRIMARY if not by_phase.get(p, {}).get("strict_success")]
    headline = ("新 MD 双 Agent 架构完成三个学习班次与完整记忆终局，主任务通过。" if success
                else "本轮主任务未通过；失败或未运行项目：" + "、".join(failures or ["实验汇总或清理检查"]) + "。")
    analysis = {"generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "experiment_kind": summary["experiment_kind"], "experiment": directory.name,
        "is_llm_result": summary.get("is_llm_result") is True, "mission_success": success,
        "conclusion_zh": headline, "api_requests": attempts, "reported_api_requests": summary.get("api_requests"),
        "known_usage": known, "reported_known_usage": declared_usage,
        "usage_complete": bool(phases) and all(p["usage_complete"] for p in phases),
        "phases": phases, "not_run": not_run, "paired_audit": summary.get("paired_audit"),
        "demonstrated_memory_dependency": summary.get("demonstrated_memory_dependency") is True,
        "demonstrated_update_dependency": summary.get("demonstrated_update_dependency") is True,
        "git": config.get("git"), "model": config.get("model"), "application_version": config.get("application_version"),
        "warnings": warnings, "videos": clips, "source_artifact_sha256": before}
    table = ["| 阶段 | 严格结果 | 终止／错误 | 墙钟 | 仿真 | 路径 | 规划请求 | 记忆请求 |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for phase in phases:
        table.append(f"| {phase['label']} | {verdict(phase['strict_success'])} | "
                     f"`{phase['error_code'] or phase['termination']}` | {number(phase['wall_elapsed_s'], '秒')} | "
                     f"{number(phase['sim_elapsed_s'], '秒')} | {number(phase['path_length_m'], '米', 3)} | "
                     f"{phase['planner']['attempts']} | {phase['memory_agent']['attempts']} |")
    report = ["# MD 双 Agent：原复杂任务复测", "", headline, "",
        f"真实模型：{analysis['is_llm_result']}；模型：`{config.get('model')}`；"
        f"冻结提交：`{(config.get('git') or {}).get('commit')}`。", "",
        "原六阶段场景与物理评分规则沿用旧复杂任务。记忆角色、提示词、调用预算和历史窗口发生变化，"
        "本轮不是只改变一个因素的严格对照，不能将结果差异直接归因为记忆架构提升。", "",
        "[播放页面](watch.html) · [原始汇总](summary.json) · [冻结配置](experiment.json)", ""]
    if clips:
        report.extend(f"- [{clip['name']}]({clip['src']})：{number(clip['duration_s'], '秒')}。" for clip in clips)
    else:
        report.append("尚无经哈希与解码校验的视频；播放页仅展示已有原始关键帧，不将截图拼接称为连续录像。")
    report += ["", "视频默认按仿真时间播放。各阶段确实重新创建进程和机器人世界，跨阶段 RESET 卡表示真实重置；"
               "录像覆盖已执行部分，未运行阶段不补画面。", "", "## 实际执行", "", *table, "",
               f"逐角色请求合计：{attempts}；已知用量：`{json.dumps(known, ensure_ascii=False)}`；"
               f"用量完整：{analysis['usage_complete']}。未知请求用量不按零消耗或输出上限计费。", ""]
    for phase in phases:
        report += ["## " + phase["label"], "",
                   f"碰撞：{phase['evaluation']['collision']}；跌倒：{phase['evaluation']['fallen']}；"
                   f"停止：{phase['evaluation']['stopped']}。", ""]
        for role_name, role in (("规划 Agent", phase["planner"]), ("记忆 Agent", phase["memory_agent"])):
            report += [f"- {role_name}：请求 {role['attempts']}，完整回复 {role['completed_responses']}，"
                       f"用量未完整记录的尝试 {role['unknown_usage_attempts']}；已知用量 "
                       f"`{json.dumps(role['known_usage'], ensure_ascii=False)}`。"]
            if role["errors"]:
                report.append("  模型／worker 错误：`" + json.dumps(role["errors"], ensure_ascii=False) + "`。")
        report += ["", "MD 初末快照：`" + json.dumps(phase["snapshots"], ensure_ascii=False) + "`。",
                   f"规划器成功读取 MD 次数：{len(phase['memory_reads'])}。"
                   "这里统计新 MD 文档读取和版本，不使用旧 memory_save 工具调用数代表新架构的记忆表现。", ""]
        for revision in phase["memory_versions"]:
            links = [f"[{item['path']}]({item['href']})（{item['status']}，{item['evidence_count']}条来源）"
                     for item in revision["changed_documents"]]
            report.append(f"- [记忆 v{revision['revision']}]({revision['manifest']})，处理水位 "
                          f"{revision['processed_event_seq']}：" + ("；".join(links) or "无新增或修订文档") + "。")
        if phase["receipt"]:
            report += ["", "终局验收：`" + json.dumps(phase["receipt"], ensure_ascii=False) + "`。"]
        if phase["tool_errors"]:
            report += ["", "保留的工具错误：`" + json.dumps(phase["tool_errors"], ensure_ascii=False) + "`。"]
        if phase["artifact_warnings"]:
            report += ["", "工件限制：" + "；".join(phase["artifact_warnings"]) + "。"]
        report += [""]
    report += ["## 对照与结论边界", "",
        "full 使用第三班后实际记忆，stale 使用第二班后的实际副本，empty 使用空记忆；"
        "三个终局的记忆均只读。未运行的条件不计为通过，没有有效对照不能证明记忆依赖或修订收益。", "",
        "配对审计：`" + json.dumps(summary.get("paired_audit"), ensure_ascii=False) + "`。", "",
        f"记忆依赖判定：{analysis['demonstrated_memory_dependency']}；"
        f"修订依赖判定：{analysis['demonstrated_update_dependency']}。", "",
        "MD 来源引用、文件哈希和发布版本保证可追溯与一致性，不保证自由文本已经完整事实核验。"
        "碰撞、模型超时、缺失记忆、用量未知和未运行阶段分别保留；不能把旧记忆条件的任意失败都解释成采用过期订单。", "",
        "未运行阶段：" + ("、".join(not_run) or "无") + "。",
        "停止原因：`" + str(summary.get("stopped_reason")) + "`。", ""]
    if warnings:
        report += ["工件汇总警告：" + "；".join(warnings) + "。", ""]
    all_frames = [frame for phase in phases for frame in phase["frames"]]
    payload = json.dumps({"clips": clips, "frames": all_frames}, ensure_ascii=False)
    for original, escaped in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"), ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        payload = payload.replace(original, escaped)
    options = "".join(f'<option value="{i}">{html.escape(clip["name"])}</option>' for i, clip in enumerate(clips))
    media = (f'<select id="clip">{options}</select><video id="video" controls preload="metadata" '
             f'src="{html.escape(clips[0]["src"], quote=True)}"></video>' if clips else
             '<p>暂无已验证视频；下方为离散原始观测帧。</p>')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MD 双 Agent 复杂任务复测</title><style>body{{max-width:1100px;margin:28px auto;padding:0 18px;font:16px/1.7 system-ui}}video,img{{width:100%;max-height:75vh;object-fit:contain;background:#111}}pre{{white-space:pre-wrap}}input{{width:100%}}</style>
<h1>MD 双 Agent 复杂任务复测</h1><p>{html.escape(headline)}</p>{media}
<p>默认显示仿真时间运动版；跨阶段 RESET 表示真实进程和机器人世界重置。截图只表示离散观测。</p>
<details {'open' if not clips else ''}><summary>真实关键帧</summary><input id="frame" type="range" min="0" max="{max(0,len(all_frames)-1)}" value="0"><p id="label"></p><img id="image" alt="公开观测关键帧"></details>
<p><a href="REPORT.zh-CN.md">完整报告</a> · <a href="analysis.json">结构化分析</a> · <a href="summary.json">原始结果</a></p>
<pre>{html.escape(chr(10).join(table))}</pre>
<script>const data={payload};const select=document.getElementById('clip'),video=document.getElementById('video');if(select)select.onchange=()=>{{video.pause();video.src=data.clips[Number(select.value)].src;video.load();}};const slider=document.getElementById('frame'),image=document.getElementById('image'),label=document.getElementById('label');function show(){{const frame=data.frames[Number(slider.value)];if(!frame){{image.hidden=true;slider.disabled=true;label.textContent='没有已保存的公开图像';return;}}image.src=frame.src;label.textContent=frame.label+' · 调用 '+frame.turn;}}slider.oninput=show;show();</script></html>'''
    contents = ("\n".join(report), json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n", page)
    if any(sha256(directory / relative) != digest for relative, digest in before.items()):
        raise RuntimeError("Source artifacts changed during report preparation")
    for output, content in zip(outputs, contents):
        with output.open("x", encoding="utf-8") as stream:
            stream.write(content)
    if any(sha256(directory / relative) != digest for relative, digest in before.items()):
        raise RuntimeError("Source artifacts changed during report generation")
    return {"outputs": [str(path) for path in outputs], "source_artifacts_unchanged": len(before),
            "mission_success": success, "api_requests": attempts, "known_usage": known,
            "usage_complete": analysis["usage_complete"], "verified_videos": len(clips), "warnings": warnings}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build_report(args.directory), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
