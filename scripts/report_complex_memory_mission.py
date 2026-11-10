"""Build offline complex-mission reports; never call a model or mutate evidence.

Existing outputs are protected. A video manifest is optional: without it the
viewer labels raw front-camera observations as decision frames, not video.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import html
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo


PHASES = ("dispatch", "fragments", "revision", "full", "stale", "empty")
PRIMARY = PHASES[:3] + ("full",)
LABELS = {"dispatch": "第一班次：初始调度", "fragments": "第二班次：码片段与区域映射",
          "revision": "第三班次：调度修订", "full": "终局：完整记忆",
          "stale": "对照：修订前记忆", "empty": "对照：空记忆"}
SOURCES = {"dispatch": "初始订单、旧目标／许可通道、回执前半段",
           "fragments": "回执后半段、区域别名到标牌的映射、不同订单干扰",
           "revision": "生效修订后的目标／许可通道，以及明确标记为历史的旧公告"}
RECEIPT_FLAGS = ("success", "submitted", "order_id_match", "revision_match", "destination_match",
                 "code_match", "prefix_match", "suffix_match", "route_match", "route_success",
                 "old_revision_selected", "old_destination_selected", "old_route_selected")
ROUTE_FLAGS = ("success", "first_gate", "wrong_gate_first", "wrong_gate_crossings",
               "first_barrier_outside_crossings", "green_outside_crossings", "out_of_order_crossings",
               "green_crossed", "green_clearance_reached", "beyond_green_clearance", "invalid_path", "invalid_samples")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def local_file(directory, relative):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError("Missing or nonlocal artifact: " + str(relative))
    return path


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def number(value, suffix="", digits=2):
    return f"{value:.{digits}f}{suffix}" if type(value) in (int, float) and math.isfinite(value) else "未记录"


def verdict(value):
    return "通过" if value is True else "未通过" if value is False else "未记录"


def strict_result(result):
    worker = result.get("worker")
    return (result.get("metrics", {}).get("success") is True and isinstance(worker, dict)
            and not worker.get("error") and not worker.get("cleanup_errors") and not result.get("artifact_error"))


def source_refs(evidence, episode_phase):
    grouped = {phase: set() for phase in PHASES[:3]}
    for item in evidence:
        if not isinstance(item, dict):
            continue
        phase = episode_phase.get(item.get("public_record", {}).get("episode_id"))
        ref = item.get("evidence_ref")
        if phase in grouped and isinstance(ref, str):
            grouped[phase].add(ref)
    return grouped


def inspect_phase(directory, result, episode_phase):
    phase = result["phase"]
    if phase not in PHASES:
        raise ValueError("Unknown archived phase")
    base, metrics = directory / phase, result["metrics"]
    if (base / "metrics.json").exists() and read_json(base / "metrics.json") != metrics:
        raise ValueError("Summary and metrics disagree: " + phase)
    events, artifact_warnings = [], []
    if (base / "events.jsonl").exists():
        for line_index, line in enumerate((base / "events.jsonl").read_text(encoding="utf-8").splitlines()):
            try:
                if line.strip():
                    events.append(json.loads(line))
            except ValueError:
                artifact_warnings.append(f"events.jsonl 第{line_index + 1}行无法解析；只统计已读取部分")
                break
    path_length = None
    trajectory_file = base / "trajectory.jsonl"
    if trajectory_file.exists():
        try:
            trajectory = rows(trajectory_file)
            anchor = next(e["base_xy_m"] for e in events if e.get("event") == "episode_start")
            points = [anchor] + [row["base_xyz_m"][:2] for row in trajectory]
            if not all(len(p) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in p) for p in points):
                raise ValueError("Invalid trajectory")
            path_length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        except (ValueError, KeyError, TypeError, StopIteration):
            artifact_warnings.append("轨迹或episode_start锚点不完整，路径长度不计为零")
    responses = [e for e in events if e.get("event") == "model_response"]
    tool_results = [(e.get("turn", -1) + 1, item) for e in events if e.get("event") == "tool_results"
                    for item in e.get("results", [])]
    memory_events, errors, retrieved_refs = [], [], {p: set() for p in PHASES[:3]}
    for turn, item in tool_results:
        tool, status = item.get("tool", "unknown"), item.get("status")
        if status != "succeeded":
            errors.append({"turn": turn, "tool": tool, "error_code": item.get("error_code")})
        if tool.startswith("memory_"):
            entry = {"turn": turn, "tool": tool, "status": status}
            if tool == "memory_search" and status == "succeeded":
                entry["matches"] = len(item.get("matches", []))
            if tool == "memory_read" and status == "succeeded":
                memory = item.get("memory", {})
                entry.update(memory_id=memory.get("memory_id"), version=memory.get("version"))
                for source, refs in source_refs(item.get("evidence", []), episode_phase).items():
                    retrieved_refs[source].update(refs)
            memory_events.append(entry)
    snapshots = {}
    for label in ("initial", "final"):
        path = base / ("memory-" + label + ".json")
        if path.exists():
            snapshot = read_json(path)
            snapshots[label] = {"sha256": snapshot.get("sha256"),
                "records": len(snapshot.get("content", {}).get("records", [])),
                "evidence_sources": {p: sorted(refs) for p, refs in source_refs(
                    snapshot.get("content", {}).get("evidence", []), episode_phase).items()}}
    evaluation = metrics.get("evaluation", {})
    extension = evaluation.get("task_extension", {})
    receipt = {key: extension.get(key) for key in RECEIPT_FLAGS} if extension else None
    if receipt is not None:
        receipt["submission"] = {k: v for k, v in (extension.get("submission") or {}).items()
                                 if k in {"dispatch_revision", "destination", "route"}}
        route = extension.get("route_evaluation", {})
        receipt["route_evaluation"] = {key: route.get(key) for key in ROUTE_FLAGS}
        receipt["route_events"] = [{key: event.get(key) for key in ("gate", "direction", "sim_time_s")}
                                   for event in route.get("ordered_gate_events", [])]
    frames = []
    for event in events:
        if event.get("event") == "model_input" and isinstance(event.get("image_path"), str):
            relative = (Path(phase) / event["image_path"]).as_posix()
            try:
                local_file(directory, relative)
                frames.append({"src": relative, "turn": event.get("turn", -1) + 1,
                               "label": LABELS[phase]})
            except ValueError:
                artifact_warnings.append("一个公开观测图像缺失，播放页不生成损坏链接")
    return {"phase": phase, "label": LABELS[phase], "strict_success": strict_result(result),
        "reported_success": metrics.get("success"), "termination": metrics.get("termination"),
        "is_llm_result": metrics.get("is_llm_result") is True,
        "model_calls": metrics.get("model_calls", 0), "logged_responses": len(responses),
        "usage_complete": len(responses) == metrics.get("model_calls", 0)
            and all(type(e.get("usage", {}).get("total_tokens")) is int for e in responses),
        "usage": {k: v for k, v in metrics.get("usage", {}).items()
                  if k in {"prompt_tokens", "completion_tokens", "total_tokens"}},
        "wall_elapsed_s": metrics.get("wall_elapsed_s"), "sim_elapsed_s": metrics.get("sim_elapsed_s"),
        "model_wait_s": metrics.get("model_wait_s"), "path_length_m": path_length,
        "memory_mode": metrics.get("memory_mode"), "memory_events": memory_events,
        "retrieved_nonempty_memory": result.get("retrieved_nonempty_memory", False),
        "retrieved_source_refs": {p: sorted(refs) for p, refs in retrieved_refs.items()},
        "snapshots": snapshots, "receipt": receipt, "tool_errors": errors,
        "evaluation": {key: evaluation.get(key) for key in ("physical_success", "collision", "fallen",
            "stopped", "distance_to_target_center_m", "heading_error_rad", "stages", "stage_completion_rate")},
        "has_worker_error": bool((result.get("worker") or {}).get("error")),
        "has_cleanup_error": bool((result.get("worker") or {}).get("cleanup_errors")),
        "has_artifact_error": bool(result.get("artifact_error")), "artifact_warnings": artifact_warnings,
        "error_code": (metrics.get("error") or {}).get("code"), "frames": frames}


def stale_conclusion(summary, by_phase):
    stale = by_phase.get("stale")
    if stale is None:
        return "旧记忆对照未运行。"
    if stale["strict_success"]:
        return "旧记忆条件也成功，本次不能据此声称更新后的记忆是成功所必需。"
    if stale["evaluation"].get("collision") or stale["evaluation"].get("fallen"):
        return "旧记忆条件发生物理碰撞或跌倒；不能仅凭此失败认定模型采用了过期订单。"
    if summary.get("stale_submitted_complete_old_order") and stale["termination"] == "finish":
        return "旧记忆分支提交了完整旧订单后finish：回执码仍完整，但修订、目标和许可路线为旧值。"
    if stale["termination"] == "abort":
        return "旧记忆分支主动abort，没有完成交付；这不等同于执行了完整旧订单。"
    if summary.get("stale_submitted_complete_old_order"):
        return "旧记忆分支曾提交完整旧订单，但以其他原因终止，不能将整次失败只归因于信息过期。"
    return "旧记忆分支未成功，但没有完整旧订单提交证据；按原始结束原因报告。"


def videos(directory):
    manifest_path = directory / "video-manifest.json"
    if not manifest_path.exists():
        return None, []
    manifest, clips = read_json(manifest_path), []
    entries = [(kind, "全流程", item, Path()) for kind, item in manifest.get("combined", {}).items()]
    entries += [(kind, LABELS.get(phase["phase"], phase["phase"]), item, Path(phase["phase"]) / "third_person")
                for phase in manifest.get("phases", []) for kind, item in phase.get("videos", {}).items()]
    entries.sort(key=lambda item: item[0] != "sim")
    for kind, title, item, prefix in entries:
        relative = (prefix / item["file"]).as_posix()
        path = local_file(directory, relative)
        if not item.get("decoded_all_frames") or sha256(path) != item.get("sha256"):
            raise ValueError("Video is missing verification or its hash changed: " + relative)
        clips.append({"src": relative, "name": title + (" · 仿真时间运动版" if kind == "sim"
                         else " · 实际耗时版（运动会压缩）"), "duration_s": item.get("duration_s"), "timebase": kind})
    return manifest, clips


def build_report(directory):
    directory = Path(directory).resolve()
    outputs = [directory / name for name in ("REPORT.zh-CN.md", "analysis.json", "watch.html")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Report outputs already exist; refusing to overwrite them")
    protected = sorted(path for path in directory.rglob("*") if path.is_file())
    before = {path.relative_to(directory).as_posix(): sha256(path) for path in protected}
    summary = read_json(local_file(directory, "summary.json"))
    config = read_json(local_file(directory, "experiment.json"))
    episode_phase = {}
    for phase in PHASES:
        manifest_path = directory / phase / "manifest.json"
        if manifest_path.exists():
            episode = read_json(manifest_path).get("memory_episode_id")
            if episode:
                episode_phase[episode] = phase
    phases = [inspect_phase(directory, result, episode_phase) for result in summary.get("phases", [])]
    by_phase = {phase["phase"]: phase for phase in phases}
    if len(phases) != len(by_phase):
        raise ValueError("Duplicate phase results")
    primary_success = all(by_phase.get(p, {}).get("strict_success") is True for p in PRIMARY)
    if bool(summary.get("mission_success")) != primary_success:
        raise ValueError("Mission flag disagrees with strict primary phase results")
    actual_calls = sum(phase["model_calls"] for phase in phases)
    if actual_calls != summary.get("api_requests"):
        raise ValueError("API totals disagree")
    real_run = summary.get("is_llm_result") is True and all(p["is_llm_result"] for p in phases)
    headline = ("本轮完整任务通过：三个学习班次及完整记忆终局均通过严格验收。" if primary_success
                else "本轮完整任务未通过；失败或未运行的主任务阶段："
                + "、".join(LABELS[p] for p in PRIMARY if not by_phase.get(p, {}).get("strict_success")) + "。")
    if not real_run:
        headline = "这些工件不是完整真实模型实验，仅用于离线验证；以下是工件中记录的结果。" + headline
    stale_text = stale_conclusion(summary, by_phase)
    video_manifest, clips = videos(directory)
    audit = summary.get("paired_audit") or {}
    checks = audit.get("checks", {})
    total_tokens = sum(p["usage"].get("total_tokens", 0) for p in phases)
    total_wall = sum(p["wall_elapsed_s"] or 0 for p in phases)
    analysis = {"generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "experiment": directory.name, "is_llm_result": real_run, "mission_success": primary_success,
        "conclusion_zh": headline, "stale_conclusion_zh": stale_text,
        "demonstrated_memory_dependency": summary.get("demonstrated_memory_dependency"),
        "demonstrated_update_dependency": summary.get("demonstrated_update_dependency"),
        "controls_infrastructure_valid": summary.get("controls_infrastructure_valid"),
        "stale_submitted_complete_old_order": summary.get("stale_submitted_complete_old_order"),
        "actual_api_requests": actual_calls, "known_total_tokens": total_tokens,
        "usage_complete": all(p["usage_complete"] for p in phases), "sum_episode_wall_s": total_wall,
        "phases": phases, "paired_audit": audit,
        "not_run": [p for p in PHASES if p not in by_phase], "has_experiment_error": bool(summary.get("error")),
        "stopped_early": bool(summary.get("stopped_reason")),
        "git": config.get("git"), "application_version": config.get("application_version"),
        "model": config.get("model"), "videos": clips, "source_artifact_sha256": before}
    table = ["| 阶段 | 严格结果 | 终止原因 | 墙钟 | 仿真 | 路径 | 模型调用 |",
             "|---|---|---|---:|---:|---:|---:|"]
    for p in phases:
        table.append(f"| {p['label']} | {verdict(p['strict_success'])} | `{p['termination']}` | "
                     f"{number(p['wall_elapsed_s'], '秒')} | {number(p['sim_elapsed_s'], '秒')} | "
                     f"{number(p['path_length_m'], '米', 3)} | {p['model_calls']} |")
    report = ["# 多区交班与订单变更实验报告", "", headline, "", stale_text, "",
        f"模型：`{config.get('model')}`；版本：`{config.get('application_version')}`；"
        f"冻结提交：`{(config.get('git') or {}).get('commit')}`。", "",
        "[打开播放页面](watch.html)", ""]
    if clips:
        report.extend(f"- [{clip['name']}]({clip['src']})：{number(clip['duration_s'], '秒')}。" for clip in clips)
    else:
        report.append("尚无已验证的渲染视频。播放页仅展示现存决策时刻的原始前视图，不能当作连续录像。")
    report += ["", "仿真时间版默认播放，1秒视频对应1秒仿真；实际耗时版会把快速计算的动作压缩，"
        "不适合观察连续步态。录像从真实控制步状态离线还原，换场以RESET页标记；初始化站稳过程不录入。", "",
        "## 逐阶段结果", "", *table, "",
        f"已记录模型请求 {actual_calls} 次；已知 token {total_tokens:,}；"
        f"用量完整：{analysis['usage_complete']}。各回合墙钟合计 {total_wall:.2f} 秒，"
        "不含进程初始化、离线渲染和报告生成。路径从episode_start起点累计所有实际轨迹点；"
        "机器人换场重置，因此各阶段路径单列，不能称为一条连续跨班次路径。", "",
        "## 三班信息与来源证据", "",
        "公告在各自学习班次的任务中持续可见；保存、检索和修订均由Agent决定。"
        "任务程序不自动写答案。每班结束关闭进程，下一班重建模型、机器人、近期历史、跟踪点与临时笔记。", ""]
    full = by_phase.get("full", {})
    initial_sources = full.get("snapshots", {}).get("initial", {}).get("evidence_sources", {})
    retrieved_sources = full.get("retrieved_source_refs", {})
    for phase, description in SOURCES.items():
        report.append(f"- {LABELS[phase]}：{description}。完整记忆初始库含该班来源引用 "
            f"{len(initial_sources.get(phase, []))} 条；终局实际memory_read返回该班来源引用 "
            f"{len(retrieved_sources.get(phase, []))} 条。")
    report += ["", "来源引用覆盖不自动证明记忆文本准确；最终字段匹配及实际路线才检验是否正确应用。"
        "此报告不展示六位回执码、记忆原文、私有推理或任何密钥。", "", "## 过程与验收分项", ""]
    for p in phases:
        report += ["### " + p["label"], "",
            f"结果：{verdict(p['strict_success'])}；终止：`{p['termination']}`；"
            f"碰撞：{p['evaluation'].get('collision')}；跌倒：{p['evaluation'].get('fallen')}；"
            f"记忆模式：`{p['memory_mode']}`。", ""]
        if p["receipt"]:
            report += ["提交与最终验收：`" + json.dumps(p["receipt"], ensure_ascii=False) + "`。", ""]
        if p["memory_events"]:
            report += ["记忆工具记录：`" + json.dumps(p["memory_events"], ensure_ascii=False) + "`。", ""]
        if p["tool_errors"]:
            report += ["保留的工具失败：`" + json.dumps(p["tool_errors"], ensure_ascii=False) + "`。", ""]
        if p["artifact_warnings"]:
            report += ["工件限制：" + "；".join(p["artifact_warnings"]), ""]
        links = [("metrics", f"{p['phase']}/metrics.json"), ("events", f"{p['phase']}/events.jsonl"),
                 ("轨迹", f"{p['phase']}/trajectory.jsonl")]
        report += [" · ".join(f"[{name}]({path})" for name, path in links if (directory / path).is_file()), ""]
    report += ["## 对照与结论边界", "",
        "full读取第三班revision后的真实记忆；stale读取第二班fragments后的原样快照，"
        "没有手动删改文本或注入旧答案；empty使用空库。三个终局均为read_only，场景、起点、公开任务、"
        "模型设置、工具与预算相同。配对检查验证这些条件，缺失对照不计为通过。", "",
        f"配对检查：{sum(value is True for value in checks.values())}/{len(checks)}；总判定：{audit.get('passed')}。"
        f"记忆依赖条件：{summary.get('demonstrated_memory_dependency')}；"
        f"修订依赖条件：{summary.get('demonstrated_update_dependency')}。", "", stale_text, "",
        "空记忆结果单独列出；主动中止不等于导航碰撞，旧记忆碰撞也不能证明完整旧订单被采用。"
        "主任务严格成功要求三个学习阶段与full全部通过。基础设施错误、未完成、失败均保留，"
        "没有自动重试挑选成功样本。一次合同的一组对照不能估计总体成功率或统计显著性。", "",
        "未运行阶段：" + ("、".join(analysis["not_run"]) or "无") + "。", "",
        "[原始汇总](summary.json) · [冻结配置](experiment.json) · [结构化分析](analysis.json)", ""]
    # JSON is embedded as data only; escape HTML-significant characters and JS line separators.
    all_frames = [frame for p in phases for frame in p["frames"]]
    payload = json.dumps({"clips": clips, "frames": all_frames}, ensure_ascii=False).replace("<", "\\u003c").replace(
        ">", "\\u003e").replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    options = "".join(f'<option value="{i}">{html.escape(c["name"])}</option>' for i, c in enumerate(clips))
    video_html = (f'<select id="clip">{options}</select><video id="video" controls playsinline preload="metadata" '
                  f'src="{html.escape(clips[0]["src"], quote=True)}"></video>' if clips
                  else '<p>尚未渲染连续视频；下面是离散决策观测帧。</p>')
    download_links = " · ".join(f'<a href="{html.escape(c["src"], quote=True)}" download>{html.escape(c["name"])}</a>' for c in clips)
    rows_html = "".join("<tr>" + "".join("<td>" + html.escape(str(cell)) + "</td>" for cell in (
        p["label"], verdict(p["strict_success"]), p["termination"], number(p["wall_elapsed_s"], "秒"),
        number(p["path_length_m"], "米", 3), p["model_calls"])) + "</tr>" for p in phases)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>多区交班实验</title><style>body{{font:16px/1.6 system-ui;background:#101722;color:#e9eef5;margin:0}}main{{max-width:1100px;margin:auto;padding:24px}}video,img{{width:100%;max-height:75vh;object-fit:contain;background:#000}}a{{color:#8cc9ff}}select,input{{font:inherit;padding:6px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #465366;text-align:left}}.result{{padding:16px;background:#24334b}}.note{{color:#bdcbe0}}</style>
<main><h1>多区交班与订单变更实验</h1><p class="result">{html.escape(headline)}</p><p>{html.escape(stale_text)}</p>
{video_html}<p>{download_links}</p><p class="note">仿真时间录像默认播放；实际耗时版可能长时间静止后突跳。每个阶段使用独立进程并重置机器人；第三视角未提供给模型。</p>
<table><thead><tr><th>阶段</th><th>严格结果</th><th>终止</th><th>墙钟</th><th>路径</th><th>模型调用</th></tr></thead><tbody>{rows_html}</tbody></table>
<details {'open' if not clips else ''}><summary>原始前视决策帧：离散截图，不是连续录像</summary><input id="frame" type="range" min="0" max="{max(0,len(all_frames)-1)}" value="0"><p id="frame-label"></p><img id="frame-image" alt="原始决策观测"></details>
<p><a href="REPORT.zh-CN.md">实验报告</a> · <a href="analysis.json">结构化分析</a> · <a href="summary.json">原始汇总</a></p>
<script>const data={payload};const selector=document.getElementById('clip'),video=document.getElementById('video');if(selector)selector.addEventListener('change',()=>{{video.pause();video.src=data.clips[Number(selector.value)].src;video.load();}});const slider=document.getElementById('frame'),image=document.getElementById('frame-image'),label=document.getElementById('frame-label');function frame(){{const item=data.frames[Number(slider.value)];if(!item){{image.hidden=true;slider.disabled=true;label.textContent='没有可展示的原始图像';return;}}image.src=item.src;label.textContent=item.label+' · 第'+item.turn+'轮';}}slider.addEventListener('input',frame);frame();</script></main></html>'''
    contents = ("\n".join(report), json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n", page)
    if any(sha256(directory / relative) != digest for relative, digest in before.items()):
        raise RuntimeError("Source artifacts changed while generating report")
    for output, content in zip(outputs, contents):
        with output.open("x", encoding="utf-8") as stream:
            stream.write(content)
    if any(sha256(directory / relative) != digest for relative, digest in before.items()):
        raise RuntimeError("Source artifacts changed during output write")
    return {"outputs": [str(p) for p in outputs], "source_artifacts_unchanged": len(before),
            "mission_success": primary_success, "is_llm_result": real_run, "api_requests": actual_calls,
            "verified_videos": len(clips)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build_report(args.experiment_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
