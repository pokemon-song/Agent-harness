"""Render an offline Chinese comparison report without modifying API evidence.

Usage: python scripts/report_model_comparison.py --experiment-dir runs/model-eval/EXPERIMENT
Named local JSON inputs and this script's own source hash provide the report.
No model, simulator, networking, environment credentials, or key files are loaded.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from zoneinfo import ZoneInfo


MODEL_DIRS = ("deepseek", "qwen")
FIXED_IMAGE_TASKS = ("go_to_blue_ball", "go_avoid_texture_drawers")
DISPLAY_TIMEZONE = ZoneInfo("Asia/Shanghai")
GROUP_NAMES = {"go_to": "接近目标", "distinguish": "字母辨别",
               "go_avoid": "绕障", "go_through": "穿门"}
COMPONENTS = (("completion", "严格成功"), ("path_efficiency", "路径"),
              ("wall_time", "墙钟"), ("sim_time", "仿真"), ("model_calls", "调用"))
ERROR_DETAIL_LABELS = {"response_truncated": "响应截断", "json_decoding_rejected": "JSON解析拒绝（层级未定）",
                       "model_timeout": "模型请求超时", "unspecified": "未细分"}
RAW_METRICS = (("path_length_m", "路径/m"), ("wall_elapsed_s", "墙钟/s"),
               ("sim_elapsed_s", "仿真/s"), ("model_calls", "模型调用/次"),
               ("api_latency_s", "API累计延迟/s"), ("total_yaw_rad", "累计转向/rad"))
OFFICIAL = {
    "deepseek_pricing": "https://api-docs.deepseek.com/quick_start/pricing/",
    "deepseek_thinking": "https://api-docs.deepseek.com/guides/thinking_mode/",
    "qwen_chat": "https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions",
    "qwen_thinking": "https://help.aliyun.com/zh/model-studio/deep-thinking",
    "qwen_arrearage": "https://help.aliyun.com/zh/model-studio/error-code#overdue-payment",
}


def number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def fmt(value, digits=2):
    if not number(value):
        return "未知"
    return f"{value:,.{digits}f}" if digits else f"{value:,}"


def percent(value):
    return f"{100 * value:.2f}%" if number(value) else "未知"


def cell(value):
    if value is None:
        return "未知"
    text = str(value).replace("\r", " ").replace("\n", " ")
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")


def table(headers, rows):
    return ["| " + " | ".join(cell(v) for v in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(cell(v) for v in row) + " |" for row in rows), ""]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_object(path, issues, snapshots, *, required=True):
    if not path.exists():
        if required:
            issues.append(f"缺少输入文件：{path.name if path.parent.name not in MODEL_DIRS else path.parent.name + '/' + path.name}")
        return None
    try:
        raw = path.read_bytes()
        def reject_constant(_):
            raise ValueError("non-finite JSON number")
        value = json.loads(raw, parse_constant=reject_constant)
        if not isinstance(value, dict):
            raise ValueError("JSON root is not an object")
        snapshots.append((path, hashlib.sha256(raw).hexdigest()))
        return value
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError) as exc:
        issues.append(f"无法读取 {path.parent.name}/{path.name}：{type(exc).__name__}")
        return None


def absent_report(plan):
    """Missing reports are pending, with no invented aggregate score."""
    cases = [{**case, "status": "pending", "strict_success": False}
             for case in plan.get("cases", []) if isinstance(case, dict)]
    return {"model": plan.get("model", {}), "cases": cases,
            "expected_episodes": len(cases), "completed_episodes": 0,
            "successful_episodes": 0, "failed_episodes": 0, "invalid_episodes": 0,
            "pending_episodes": len(cases), "rankable": False}


def audit_plan_report(name, plan, report, issues):
    protocol = plan.get("protocol", {})
    if not isinstance(protocol, dict) or plan.get("protocol_sha256") != digest(protocol):
        issues.append(f"{name}：计划协议指纹不匹配")
    grid = plan.get("cases", [])
    rows = report.get("cases", [])
    if not isinstance(grid, list) or not all(isinstance(c, dict) for c in grid):
        issues.append(f"{name}：计划case列表损坏")
        return
    if not isinstance(rows, list) or not all(isinstance(c, dict) for c in rows):
        issues.append(f"{name}：报告case列表损坏")
        return
    expected = [c.get("case_id") for c in grid]
    actual = [c.get("case_id") for c in rows]
    if (any(not isinstance(k, str) for k in expected + actual)
            or len(set(expected)) != len(expected) or actual != expected):
        issues.append(f"{name}：报告case顺序、唯一性或覆盖与计划不一致")
    if report.get("model") != plan.get("model"):
        issues.append(f"{name}：报告模型标识与计划不一致")
    if (report.get("protocol_id") != protocol.get("id")
            or report.get("protocol_sha256") != plan.get("protocol_sha256")):
        issues.append(f"{name}：报告协议与计划不一致")
    for planned, row in zip(grid, rows):
        if any(row.get(k) != planned.get(k) for k in ("case_id", "task_id", "category", "repeat_id", "seed", "run_dir")):
            issues.append(f"{name}：case身份字段不一致")
            break
    statuses = Counter(r.get("status") for r in rows)
    successes = sum(r.get("status") == "completed" and r.get("strict_success") is True for r in rows)
    checks = {"expected_episodes": len(grid), "completed_episodes": statuses["completed"],
              "successful_episodes": successes, "failed_episodes": statuses["completed"] - successes,
              "invalid_episodes": statuses["invalid"], "pending_episodes": statuses["pending"]}
    if any(report.get(k) != v for k, v in checks.items()):
        issues.append(f"{name}：报告计数与case明细不一致")
    if any(r.get("status") not in {"completed", "invalid", "pending"}
           or type(r.get("strict_success")) is not bool for r in rows):
        issues.append(f"{name}：未知case状态或严格成功标志")
    task_ids = {task.get("task_id") for task in protocol.get("tasks", [])}
    if (set(report.get("tasks", {})) != task_ids
            or set(report.get("groups", {})) != set(GROUP_NAMES)):
        issues.append(f"{name}：逐任务或四类汇总不完整")
    for row in rows:
        if row.get("status") != "completed":
            continue
        components = row.get("score_components", {})
        values = [components.get(key) for key, _ in COMPONENTS]
        total = row.get("episode_score_100")
        if (not all(number(value) and value >= 0 for value in values) or not number(total)
                or not math.isclose(math.fsum(values), total, rel_tol=0, abs_tol=1e-8)
                or row.get("strict_success") is False and total != 0):
            issues.append(f"{name}：回合分项、综合分或失败零分规则不一致")
            break
    if report.get("rankable") is True and not (
            protocol.get("profile") == "standard" and len(grid) == 65
            and statuses["completed"] == 65 and not statuses["invalid"] and not statuses["pending"]
            and report.get("complete") is True and number(report.get("score_100"))):
        issues.append(f"{name}：rankable标志与完整标准计划证据不一致")


def case_map(report):
    return {r["case_id"]: r for r in report.get("cases", [])
            if isinstance(r, dict) and isinstance(r.get("case_id"), str)}


def measured(rows, key, *, total=False):
    values = [r[key] for r in rows if number(r.get(key))]
    if not values:
        return "未知（n=0）"
    value = math.fsum(values) if total else statistics.mean(values)
    return f"{fmt(value)}（n={len(values)}）"


def counts(report):
    rows = report.get("cases", [])
    return (sum(r.get("strict_success") is True and r.get("status") == "completed" for r in rows),
            sum(r.get("strict_success") is not True and r.get("status") == "completed" for r in rows),
            sum(r.get("status") == "invalid" for r in rows),
            sum(r.get("status") == "pending" for r in rows))


def component_cells(value):
    components = value.get("score_components", {})
    return [fmt(components.get(key)) for key, _ in COMPONENTS]


def format_timestamp(value):
    if not isinstance(value, str) or not value:
        return "未记录"
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if instant.tzinfo is None:
            instant = instant.replace(tzinfo=timezone.utc)
        local = instant.astimezone(DISPLAY_TIMEZONE)
    except (ValueError, OverflowError):
        return f"无法转换（原值：{value}）"
    return f"{local:%Y-%m-%d %H:%M:%S} 北京时间（UTC原值：{value}）"


def repeat_statistics(plan, report):
    archived = case_map(report)
    rounds = []
    for repeat_id in range(1, 6):
        planned = [case for case in plan.get("cases", []) if case.get("repeat_id") == repeat_id]
        paired = [(case, archived.get(case.get("case_id"), {})) for case in planned]
        completed = sum(row.get("status") == "completed" for _, row in paired)
        successes = sum(row.get("status") == "completed" and row.get("strict_success") is True for _, row in paired)
        invalid = sum(row.get("status") == "invalid" for _, row in paired)
        pending = sum(row.get("status") not in {"completed", "invalid"} for _, row in paired)
        group_scores, group_success = [], []
        for category in GROUP_NAMES:
            members = [row for case, row in paired if case.get("category") == category]
            group_scores.append(statistics.mean(
                row["episode_score_100"] if row.get("status") == "completed"
                and row.get("strict_success") is True and number(row.get("episode_score_100")) else 0
                for row in members) if members else 0)
            group_success.append(sum(row.get("status") == "completed" and row.get("strict_success") is True
                                     for row in members) / len(members) if members else 0)
        complete = (len(planned) == 13 and completed == 13 and not invalid and not pending
                    and all(number(row.get("episode_score_100")) for _, row in paired)
                    and all(any(case.get("category") == category for case in planned) for category in GROUP_NAMES))
        rounds.append({"repeat_id": repeat_id, "expected": len(planned), "completed": completed,
                       "successes": successes, "invalid": invalid, "pending": pending, "complete": complete,
                       "score_100": statistics.mean(group_scores),
                       "strict_success_score_100": 100 * statistics.mean(group_success)})
    complete_rounds = sum(row["complete"] for row in rounds)
    scores = [row["score_100"] for row in rounds]
    return {"rounds": rounds, "complete_rounds": complete_rounds, "mean_score_100": statistics.mean(scores),
            "minimum": min(scores) if complete_rounds == 5 else None,
            "maximum": max(scores) if complete_rounds == 5 else None,
            "sample_stdev": statistics.stdev(scores) if complete_rounds == 5 else None}


def repeat_stability_section(plans, reports):
    lines = ["### 3.1 逐轮重复稳定性", "",
             "一轮是同一repeat编号下的13个任务。完成数只计有效completed回合，成功数以全部13个计划槽位为分母；invalid与pending零填。只有13个有效回合齐全才称为完成轮，未齐轮的分数明确为临时值。", "",
             "每轮先在go_to、distinguish、go_avoid、go_through各组内对任务等权平均，再四类各占25%，分别计算综合分与严格成功率百分分。**成功/13是微平均计数，不能代替四类宏平均主分。**", ""]
    statistics_by_model = {name: repeat_statistics(plans[name], reports[name]) for name in MODEL_DIRS}
    rows = []
    for index in range(5):
        for name in MODEL_DIRS:
            value = statistics_by_model[name]["rounds"][index]
            rows.append((value["repeat_id"], reports[name].get("model", {}).get("label", name),
                         f"{value['completed']}/13", f"{value['successes']}/13", value["invalid"],
                         value["pending"] if value["expected"] == 13 else "计划缺项",
                         "完成轮" if value["complete"] else "临时（未齐／invalid）",
                         fmt(value["score_100"]), fmt(value["strict_success_score_100"])))
    lines += table(["repeat", "模型", "有效完成/13", "成功/13", "invalid", "pending", "轮状态",
                    "四类宏平均综合分", "四类宏平均严格SR百分分"], rows)
    summary = []
    for name in MODEL_DIRS:
        value = statistics_by_model[name]
        reference = reports[name].get("provisional_score_100")
        agreement = ("一致" if math.isclose(value["mean_score_100"], reference, rel_tol=0, abs_tol=1e-8)
                     else "不一致，需核查") if number(reference) else "不可核对"
        summary.append((reports[name].get("model", {}).get("label", name), f"{value['complete_rounds']}/5",
                        fmt(value["mean_score_100"]), agreement,
                        *(fmt(value[key]) if value[key] is not None else "不可用"
                          for key in ("minimum", "maximum", "sample_stdev"))))
    lines += table(["模型", "已完成轮/5", "五轮等权均分（含临时零填）", "与整体综合分核对",
                    "五轮最小分", "五轮最大分", "五轮样本标准差"], summary)
    lines += ["在完整平衡矩阵中，五轮分数等权平均与“每任务5次重复平均→组内任务平均→四类25%平均”的整体综合分一致；两种线性平均顺序可交换。未齐时也保留全部65个槽位作同口径零填核对，不能只平均已完成轮。", "",
              "只有某模型5轮全部完成且无invalid，才显示其5轮综合分的最小值、最大值和样本标准差（n=5、分母n−1）；未齐时均标不可用。这些是固定场景重复运行的纯描述统计，不是置信区间或显著性检验，也不能根据首轮结果宣布最终赢家。", ""]
    return lines


def fixed_image_section(directory):
    lines = ["## 11. 固定样例图像", "",
             "报告制作时固定选取第1次重复的蓝球接近与抽屉绕障任务，双模型共4个case，展示名单不随后续成绩替换。预注册的是全部130回合调度，图像展示名单未单独预注册。仅链接已存在的原始PNG，不读取或改动图像内容；初始图用普通链接，每个case最多展示一张终态原图。", ""]
    rows = []
    for task_id in FIXED_IMAGE_TASKS:
        for model in MODEL_DIRS:
            frames = directory / model / "episodes" / task_id / "repeat-01" / "frames"
            initial = sorted(path for path in frames.glob("000-*.png") if path.is_file())
            final = sorted(path for path in frames.glob("final-*.png") if path.is_file())
            def links(paths):
                return "、".join(f"[{path.name}]({path.relative_to(directory).as_posix()})" for path in paths)
            initial_link = links(initial) if initial else "初始原图未生成"
            if final:
                relative = final[0].relative_to(directory).as_posix()
                final_link = links(final) + f" ![{model} {task_id} r01终态原图]({relative})"
            else:
                final_link = "终态原图未生成"
            rows.append((f"{task_id}-r01", model, initial_link, final_link))
    lines += table(["固定case", "模型", "初始原图", "终态原图"], rows)
    lines += ["若同一前缀匹配多张PNG，全部保留普通链接，终态预览固定使用文件名排序首项。缺文件明确标为未生成；不从其他重复或任务补选图片。图像只辅助理解任务与终态，**严格评分仍依据metrics与trajectory等原始证据**，不能仅凭截图宣称成功。", ""]
    return lines


def service_block_section(diagnostic, reports, stage_requested=False):
    if not diagnostic:
        return []
    error = diagnostic.get("error", {})
    lines = ["### 1.1 服务阻塞与独立诊断", "",
             "正式Qwen回合的第1次模型请求遇到HTTP400，按冻结协议归为invalid并停止后续Qwen回合。原正式结果仅保存通用HTTP400／MODEL_HTTP_ERROR，没有归档具体服务端错误体；下面的细分错误来自随后对同一初始请求所做的**1次独立诊断**，不覆写原回合，也不替代原始评分。", ""]
    lines += table(["独立诊断记录", "值"],
                   [("关联原case", diagnostic.get("original_case")), ("请求模型", diagnostic.get("model")),
                    ("HTTP状态", diagnostic.get("http_status")), ("错误code", error.get("code")),
                    ("错误type", error.get("type")), ("服务返回message", error.get("message")),
                    ("请求历史轮数", diagnostic.get("history_turns")),
                    ("是否正式episode", diagnostic.get("is_benchmark_episode")),
                    ("原正式结果保持不变", diagnostic.get("original_result_unchanged"))])
    if error.get("code") == "Arrearage" or error.get("type") == "Arrearage":
        lines += ["诊断当时服务明确返回 `Arrearage`。阿里云官方将其解释为API Key所属账号欠费导致访问被拒绝；若账面未欠费，也需核对账号归属或排查账户异常。这支持账号计费／余额状态阻塞的归因，不支持把它算作模型任务规划失败。[官方Arrearage说明](" + OFFICIAL["qwen_arrearage"] + ")", ""]
    successes, failures, invalid, pending = counts(reports["qwen"])
    lines += table(["Qwen已记录覆盖", "成功", "有效失败", "invalid", "pending"],
                   [(f"{successes + failures + invalid}/{reports['qwen'].get('expected_episodes', '未知')}",
                     successes, failures, invalid, pending)])
    lines += ["[独立诊断原始JSON](preflight/http400-diagnostic.json)不是65回合之一，也不是任务重试或额外成功样本。该服务错误保留为invalid，**不记为模型有效失败或0分成绩**；进度表中的临时零填不改变这一归因。诊断没有可用usage时，token与费用均为未知，不能当作0。", "",
              "即使DeepSeek继续执行并最终完成65回合，只要Qwen仍存在invalid或pending，本次双模型实验就不能正式排名或宣布总体赢家。", ""]
    if stage_requested:
        lines += ["用户已选择暂不恢复Qwen账号，先完成DeepSeek与阶段报告；不再等待充值、不再发起Qwen请求。阶段范围变更来源见 [stage-request.json](stage-request.json)，原始Qwen结果与停止记录保留。", ""]
    else:
        lines += ["后续是否建立新Qwen计划或仅保留阶段报告需要另行决定；本报告只读取既有记录，不发起重复诊断或修改账户状态。", ""]
    return lines


def formal_model_ready(plan, report, audit_issues):
    successes, failures, invalid, pending = counts(report)
    protocol = plan.get("protocol", {})
    return (not audit_issues and report.get("rankable") is True and number(report.get("score_100"))
            and successes + failures == 65 and invalid == pending == 0
            and protocol.get("id") == "go2-model-eval-v2" and protocol.get("profile") == "standard"
            and protocol.get("repeat_count") == 5 and len(protocol.get("tasks", [])) == 13
            and protocol.get("scene_seed") == 0
            and protocol.get("budgets") == {"max_model_calls": 30, "max_wall_seconds": 300,
                                             "max_sim_seconds": 120, "history_turns": 6}
            and protocol.get("api_settings") == {"timeout_s": 90, "max_output_tokens": 8192,
                                                 "thinking": {"type": "enabled"}}
            and protocol.get("scoring", {}).get("weights") == {
                "completion": 60, "path_efficiency": 20, "wall_time": 10, "sim_time": 5, "model_calls": 5})


def failure_detail_section(analysis, plans, reports):
    if not analysis:
        return []
    lines = ["### 8.1 已记录失败明细", "",
             "以下读取[辅助失败分析JSON](failure-analysis.json)，完整列出其中所有有效失败和invalid，不挑选个案。辅助快照未覆盖当前新增失败时，也保留该case并标明明细待刷新。辅助分析只解释既有记录，不改变评分、错误归因或单模型正式分资格。", "",
             f"辅助快照生成时间：{format_timestamp(analysis.get('created_utc'))}。pending回合没有被当作失败读取。", ""]
    summaries, rows, evidence_issues = [], [], []
    def flag(value):
        return "是" if value is True else "否" if value is False else "未知"
    for name in MODEL_DIRS:
        model = reports[name].get("model", {}).get("label", name)
        source = analysis.get("models", {}).get(name, {})
        source_counts = source.get("counts") or {}
        successes, failures, invalid, pending = counts(reports[name])
        current = {"expected": reports[name].get("expected_episodes"), "successful": successes,
                   "valid_failed": failures, "invalid": invalid, "pending": pending,
                   "recorded": successes + failures + invalid}
        details = [row for row in source.get("failure_cases", []) if isinstance(row, dict)]
        recorded_ids = [row.get("case_id") for row in details]
        current_failures = [row for row in reports[name].get("cases", [])
                            if row.get("status") == "invalid" or
                            row.get("status") == "completed" and row.get("strict_success") is False]
        current_ids = {row.get("case_id") for row in current_failures}
        aligned = (source and all(source_counts.get(key) == value for key, value in current.items())
                   and source_counts.get("unclassified") == source_counts.get("unrepresented") == 0
                   and source.get("protocol_sha256") == reports[name].get("protocol_sha256")
                   and len(recorded_ids) == len(set(recorded_ids))
                   and set(recorded_ids) == current_ids)
        snapshot_state = ("与当前已记录快照一致" if aligned and source.get("analysis_complete") is True
                          else "待刷新／辅助证据不完整")
        summaries.append((model, source_counts.get("recorded"), current["recorded"], len(details),
                          failures + invalid, snapshot_state))
        if not aligned:
            lines += [f"{cell(model)}：辅助分析快照的计数、case覆盖或协议与当前report不一致，**待刷新**；下表不得冒充当前全部明细已完成。这不会阻挡该模型在独立正式评分中已满足条件的分数。", ""]
        for row in current_failures:
            if row.get("case_id") not in set(recorded_ids):
                details.append({**row, "auxiliary_detail_pending": True})
        tasks = {task.get("task_id"): task for task in plans[name].get("protocol", {}).get("tasks", [])}
        for row in details:
            criteria = tasks.get(row.get("task_id"), {}).get("evaluation", {})
            goal_kind = row.get("goal_kind", criteria.get("goal_kind"))
            if row.get("auxiliary_detail_pending"):
                geometry = "辅助几何明细待刷新"
            elif goal_kind == "standoff":
                geometry = (f"distance_error={fmt(row.get('distance_error_m'), 6)} m"
                            f"（阈值≤{fmt(criteria.get('distance_tolerance_m'), 3)}）；"
                            f"heading_error={fmt(row.get('heading_error_rad'), 6)} rad"
                            f"（阈值≤{fmt(criteria.get('heading_tolerance_rad'), 3)}）")
            elif goal_kind == "passage":
                geometry = f"passage_crossed={flag(row.get('passage_crossed'))}；beyond_exit={flag(row.get('beyond_exit'))}"
            else:
                geometry = "未知（未归档目标验收类型）"
            case_id = row.get("case_id", "未知")
            raw_directory = row.get("raw_directory")
            if (isinstance(raw_directory, str) and raw_directory and not Path(raw_directory).is_absolute()
                    and ".." not in Path(raw_directory).parts):
                case_id = f"[{case_id}]({raw_directory}/)"
            tool_codes = row.get("tool_error_code_counts")
            tool_summary = json.dumps(tool_codes, ensure_ascii=False, sort_keys=True) if isinstance(tool_codes, dict) else "未知"
            detail_kind = row.get("error_detail_kind")
            error_detail = (f"{detail_kind}（{ERROR_DETAIL_LABELS[detail_kind]}）"
                            if isinstance(detail_kind, str) and detail_kind in ERROR_DETAIL_LABELS
                            else "未提供（待辅助分析刷新）")
            rows.append((model, case_id, row.get("status"), row.get("termination"),
                         row.get("error_code") or "—（无episode错误码）", error_detail, fmt(row.get("model_calls"), 0),
                         fmt(row.get("wall_elapsed_s")), fmt(row.get("path_length_m"), 6), geometry,
                         flag(row.get("collision")), flag(row.get("fallen")), row.get("final_stop_status"),
                         f"声明={flag(row.get('finish_declared'))}；独立验收={flag(row.get('evaluation_success'))}",
                         tool_summary))
            evidence_issues.extend((model, row.get("case_id"), issue) for issue in row.get("issues", []))
        evidence_issues.extend((model, "分析汇总", issue) for issue in source.get("issues", []))
    lines += table(["模型", "分析快照已记录数", "当前report已记录数", "快照失败/invalid明细数", "当前失败/invalid数", "辅助快照状态"], summaries)
    lines += table(["模型", "case", "状态", "termination", "error_code", "error_detail_kind（白名单）", "calls", "wall/s", "path/m",
                    "独立几何验收数据", "碰撞", "跌倒", "最终stop", "finish与独立验收", "工具错误码计数"], rows)
    if not rows:
        lines += ["该辅助快照尚无已记录失败／invalid明细；这不表示未运行回合已经成功。", ""]
    if evidence_issues:
        lines += table(["模型", "case或汇总", "辅助证据问题"], evidence_issues)
    lines += ["`MODEL_PROTOCOL_ERROR`是适配器拒绝返回的协议总码，**不等同于JSON错误**。白名单细分中，`response_truncated`表示已记录的安全错误明确说明响应截断；`json_decoding_rejected`只说明已记录JSON解析拒绝，由于raw API响应未留存，不能确定解析涉及外层JSON还是工具参数。`model_timeout`表示已记录请求超时，`unspecified`表示没有进一步细分。不能把这些记录进一步归因为模型规划差。HTTP400／MODEL_HTTP_ERROR仍按原正式记录展示；`Arrearage`来自单独的账号错误诊断，两者的来源边界见第1.1节，不用诊断覆写原任务。", ""]
    return lines


def render(directory):
    issues, snapshots = [], []
    experiment = read_object(directory / "experiment.json", issues, snapshots) or {}
    environment = read_object(directory / "environment.json", [], snapshots, required=False) or {}
    diagnostic = read_object(directory / "preflight" / "http400-diagnostic.json", [], snapshots, required=False)
    failure_analysis = read_object(directory / "failure-analysis.json", [], snapshots, required=False)
    stage_request = read_object(directory / "stage-request.json", [], snapshots, required=False)
    stage_requested = (isinstance(stage_request, dict) and stage_request.get("kind") == "user_selected_stage_report"
                       and stage_request.get("deepseek_target_episodes") == 65
                       and stage_request.get("qwen_resume") is False)
    plans, reports, model_issues = {}, {}, {}
    for name in MODEL_DIRS:
        plans[name] = read_object(directory / name / "plan.json", issues, snapshots) or {}
        loaded = read_object(directory / name / "report.json", issues, snapshots)
        reports[name] = loaded if loaded is not None else absent_report(plans[name])
        model_issues[name] = []
        if loaded is not None:
            audit_plan_report(name, plans[name], loaded, model_issues[name])
            issues.extend(model_issues[name])
    comparison = read_object(directory / "comparison.json", issues, snapshots) or {}
    protocols = [p.get("protocol", {}) for p in plans.values()]
    fingerprints = {p.get("protocol_sha256") for p in plans.values()}
    if len(fingerprints) != 1 or None in fingerprints:
        issues.append("两模型计划的协议指纹不一致或缺失，禁止比较排名")
    if plans["deepseek"].get("cases") != plans["qwen"].get("cases"):
        issues.append("两模型未绑定同一个配对case网格")
    fingerprint = plans["deepseek"].get("protocol_sha256")
    if experiment.get("protocol_sha256") != fingerprint or comparison.get("protocol_sha256") != fingerprint:
        issues.append("experiment／comparison与计划协议指纹不一致或尚未生成")
    labels = [r.get("model", {}).get("label") for r in reports.values()]
    if any(not isinstance(label, str) for label in labels) or len(set(labels)) != len(labels):
        issues.append("模型标签缺失或重复")
    configured = experiment.get("models", {})
    for name in MODEL_DIRS:
        config = configured.get(name, {}) if isinstance(configured, dict) else {}
        model = plans[name].get("model", {})
        if any(config.get(a) != model.get(b) for a, b in
               (("label", "label"), ("model_name", "requested_model"), ("provider", "provider"))):
            issue = f"{name}：experiment模型配置与计划不一致或缺失"
            issues.append(issue)
            model_issues[name].append(issue)
    compared_models = comparison.get("models", [])
    compared = {m.get("model", {}).get("label"): m for m in compared_models if isinstance(m, dict)}
    for name, report in reports.items():
        entry = compared.get(report.get("model", {}).get("label"))
        if entry is None or any(entry.get(k) != report.get(k) for k in
                                ("model", "score_100", "provisional_score_100", "pending_episodes",
                                 "invalid_episodes", "score_components", "successful_episodes", "failed_episodes")):
            issues.append(f"{name}：comparison摘要与report不一致、过期或尚未生成")
    common_evidence = sorted(set.intersection(*[
        {key for key, row in case_map(report).items()
         if row.get("status") == "completed" and row.get("strict_success") is True}
        for report in reports.values()]))
    if comparison.get("common_success_cases") != common_evidence:
        issues.append("comparison共同成功case与report明细不一致、过期或尚未生成")
    grid = plans["deepseek"].get("cases", [])
    schedule = experiment.get("schedule", [])
    expected_schedule = [{"model": name, "case_id": case.get("case_id"), "index": index}
                         for index, case in enumerate(grid)
                         for name in (MODEL_DIRS if index % 2 == 0 else tuple(reversed(MODEL_DIRS)))]
    if not grid or schedule != expected_schedule:
        issues.append("预注册串行AB/BA调度与配对case网格不一致或缺失")
    protocol = protocols[0] if isinstance(protocols[0], dict) else {}
    if comparison.get("protocol_id") != protocol.get("id") or experiment.get("protocol_id") != protocol.get("id"):
        issues.append("experiment／comparison的协议ID不一致或缺失")
    if protocol.get("id") != "go2-model-eval-v2":
        issues.append("本五维实验报告只对v2协议提供正式排名说明")
    if protocol.get("scoring", {}).get("weights") != {
            "completion": 60, "path_efficiency": 20, "wall_time": 10, "sim_time": 5, "model_calls": 5}:
        issues.append("计划评分权重不是冻结的60／20／10／5／5")
    if protocol.get("budgets") != {"max_model_calls": 30, "max_wall_seconds": 300,
                                    "max_sim_seconds": 120, "history_turns": 6}:
        issues.append("计划预算与本文说明的标准预算不一致")
    if (protocol.get("repeat_count") != 5 or len(protocol.get("tasks", [])) != 13
            or protocol.get("scene_seed") != 0):
        issues.append("计划不是13任务、各5次、scene_seed=0的标准矩阵")
    if protocol.get("api_settings") != {"timeout_s": 90, "max_output_tokens": 8192,
                                        "thinking": {"type": "enabled"}}:
        issues.append("计划API预算或思考语义与本文标准不一致")
    rankable = (not issues and comparison.get("rankable") is True
                and all(r.get("rankable") is True for r in reports.values())
                and experiment.get("status") == "completed")
    formal_models = {name: formal_model_ready(plans[name], reports[name], model_issues[name]) for name in MODEL_DIRS}
    stopped_models = experiment.get("stopped_models", {})
    stage_complete = (stage_requested and formal_models["deepseek"]
                      and isinstance(stopped_models, dict) and bool(stopped_models.get("qwen")))
    generated = datetime.now(timezone.utc).isoformat()
    title = "# DeepSeek完整评测与Qwen已记录结果阶段报告" if stage_requested else "# 大模型 Go2 闭环实验报告"
    lines = [title, "",
             f"报告生成时间：{format_timestamp(generated)}。本文件由本地JSON离线生成，可重复刷新，不调用API、不读取密钥、不改写计划或原始日志。", ""]
    if stage_requested:
        lines += [f"当前阶段范围：{cell(stage_request.get('instruction', '先完成DeepSeek与阶段报告'))}。用户要求DeepSeek完成65个正式回合，Qwen保留已记录结果且不再恢复调用；不再等待账号充值。范围变更记录：[stage-request.json](stage-request.json)，接收时间：{format_timestamp(stage_request.get('received_utc'))}。原130回合调度与Qwen停止记录保留。", ""]
        if stage_complete:
            lines += ["**状态：阶段报告完成。** DeepSeek的65个有效回合已完整，可单独报告正式分；Qwen保留已记录阶段信息。**完整双模型对比仍未完成，不宣布总体赢家。**", ""]
        else:
            progress = "DeepSeek的65回合正式评测尚未完整通过核查" if not formal_models["deepseek"] else "DeepSeek已完成，Qwen停止记录尚未齐备"
            lines += [f"**状态：阶段进行中。** {progress}；Qwen不再追加请求。完整双模型对比仍未完成，不宣布总体赢家。", ""]
    elif rankable:
        scores = [(name, reports[name]["score_100"]) for name in MODEL_DIRS]
        best = max(v for _, v in scores)
        winners = [reports[n]["model"]["label"] for n, v in scores if v == best]
        result = ("、".join(winners) + "综合分并列。" if len(winners) > 1
                  else f"{winners[0]}在本次固定协议下综合分较高。")
        lines += [f"**状态：完整且可正式比较。** 两模型各65回合通过报告一致性检查；{cell(result)}", ""]
    else:
        lines += ["**状态：未完成或证据不满足正式比较条件，不能宣布赢家。** 各模型的正式与临时状态见第3节；临时分保留全部计划槽位，未完成或无效槽位零填，不构成正式排名。", ""]

    lines += ["## 1. 实验范围、模型与过程", "",
              "结果测量“模型＋当前服务适配器＋本Harness”的整套闭环，不能解释为裸模型通用能力、QUARD原环境成绩或真实机器人实时能力。", ""]
    process = [("原双模型批次状态（未改写）" if stage_requested else "实验状态", experiment.get("status", "未记录")),
               ("创建时间（北京时间）", format_timestamp(experiment.get("created_utc"))),
               ("开始时间（北京时间）", format_timestamp(experiment.get("started_utc"))),
               ("结束时间（北京时间）", format_timestamp(experiment.get("finished_utc"))),
               ("已尝试回合数（两模型合计）", experiment.get("attempted_episodes", "未记录")),
               ("预注册调度数", len(schedule) if isinstance(schedule, list) else "未知"),
               ("自动重试次数", experiment.get("automatic_episode_retries", "未记录")),
               ("过程说明", experiment.get("schedule_description", "未记录"))]
    lines += table(["项目", "记录"], process)
    identity = []
    for name in MODEL_DIRS:
        model = reports[name].get("model", {})
        config = configured.get(name, {}) if isinstance(configured, dict) else {}
        returned = reports[name].get("response_models", [])
        identity.append((name, model.get("label"), model.get("requested_model"), model.get("provider"),
                         "、".join(map(str, returned)) if returned else "尚无有效响应模型记录", config.get("endpoint")))
    lines += table(["目录", "模型标签", "请求模型名", "provider", "响应中的实际标识", "配置端点"], identity)
    lines += ["官方文档核对日为2026-11-03：DeepSeek的 `deepseek-flash` 别名对应文档所列的 **DeepSeek-V4.1-Flash-1017**。这是官方别名映射说明；如果本次响应仍只返回 `deepseek-flash`，日志就没有独立证实固定快照，不能把标签当作运行时版本认证。[DeepSeek模型与版本表](" + OFFICIAL["deepseek_pricing"] + ")", "",
              "Qwen请求名及实际返回标识以上表原始记录为准。`qwen3.8-max` 与 `qwen3.8-max-0902` 是文档列出的不同模型ID；不能仅凭前者推断实际运行的是后者快照。[百炼思考模型列表](" + OFFICIAL["qwen_thinking"] + ")", "",
              "预注册过程按同一case配对、逐回合串行执行，并在AB／BA之间交替模型顺序；不挑最好结果，不自动重跑失败。每回合冷启动仿真、适配器、跟踪器与近期历史，无跨回合共享记忆。预注册调度描述运行计划，不单独证明每项已经执行。", ""]
    stops = experiment.get("stopped_models", {})
    if isinstance(stops, dict) and stops:
        lines += table(["停止后续执行的模型", "case", "终止类型", "错误代码"],
                       [(name, value.get("case_id"), value.get("termination"), value.get("error_code"))
                        for name, value in stops.items() if isinstance(value, dict)])
    tasks = protocol.get("tasks", [])
    repeats = protocol.get("repeat_count")
    lines += [f"本计划包含{len(tasks)}个任务、每任务{cell(repeats)}次重复、每模型{len(grid)}个槽位。标准范围是13×5＝65回合，四类任务各占25%。场景seed固定为{cell(protocol.get('scene_seed'))}；这些场景的目标抖动为0，5次重复观察模型／服务波动，**不是5个随机新场景**。`red_marker`和两项不支持的爬行任务不进入主分；本次也没有独立姿态目标专项验收。", ""]
    lines += service_block_section(diagnostic, reports, stage_requested)

    lines += ["## 2. 冻结预算与五维评分", ""]
    budget, api = protocol.get("budgets", {}), protocol.get("api_settings", {})
    lines += table(["预算或设置", "计划值"],
                   [("模型调用上限", budget.get("max_model_calls")),
                    ("墙钟预算/s", budget.get("max_wall_seconds")),
                    ("仿真预算/s", budget.get("max_sim_seconds")),
                    ("近期完整工具轮次", budget.get("history_turns")),
                    ("输出上限（思考＋回答）/token", api.get("max_output_tokens")),
                    ("单次API超时/s", api.get("timeout_s")),
                    ("语义思考设置", json.dumps(api.get("thinking"), ensure_ascii=False)),
                    ("仿真模式", protocol.get("simulation_mode"))])
    lines += ["等待模型时仿真暂停；墙钟时间包含API延迟和本地执行等开销，仿真时间只记录物理推进。最终停止和停稳阶段也计入预算。模型服务不同默认推理档位不保证相同计算量。", "",
              "DeepSeek使用 `thinking={\"type\":\"enabled\"}`；Qwen使用 `enable_thinking=true` 和 `preserve_thinking=true`，Qwen的 `max_completion_tokens` 才同时限制思考与回答，不能用仅限制回答的 `max_tokens` 冒充相同预算。两者均回传保留轮次中完整的 `reasoning_content`；不在报告中输出私有思考文本。[DeepSeek思考协议](" + OFFICIAL["deepseek_thinking"] + ")、[百炼Chat参数](" + OFFICIAL["qwen_chat"] + ")", "",
              "严格成功 `S=1` 要求正常 `finish`、独立目标合格、最终停稳、全程无累计环境碰撞／跌倒且未超预算；有效失败 `S=0`。模型文字声明不能替代独立判定。", "",
              "```text", "P = min(1, D_lb / L)；L=D_lb=0时P=1，L=0且D_lb>0时P=0",
              "W = clip(1 - wall_seconds / 300, 0, 1)",
              "T = clip(1 - sim_seconds / 120, 0, 1)",
              "C = clip((30 - model_calls) / 29, 0, 1)",
              "回合综合分 = S × (60 + 20P + 10W + 5T + 5C)", "```", ""]
    weights = protocol.get("scoring", {}).get("weights", {})
    lines += table(["维度", "冻结满分"], [(label, weights.get(key)) for key, label in COMPONENTS])
    lines += ["每回合各分项先在同一任务的5次重复间平均，再对组内任务等权平均，最后四类任务各25%；分项相加得到总综合分。正式排名按综合分，同分并列。60分成功基础＋40分效率的取舍可能让成功率略低但效率更高的模型领先，必须同时阅读严格成功率。宏平均严格成功率百分分与65回合微平均另列，都不是综合分。", "",
              "实际路径 `L` 为约50Hz的base水平轨迹折线长度，接上 `episode_start` 的精确初始xy；排除暖机，包含所有停止漂移、finish和最终停稳，不累加动作命令。接近／辨别／绕障的固定下界 `D_lb=max(0,dist(start,target)-standoff-0.18)`，穿门为 `max(0,finish_x_min-start_x)`。该下界忽略障碍、朝向与机器人包络，**不是最短可行路径或SPL**，绕障路径满分可能不可达；参照对同任务的模型固定一致。缺失、稀疏、时间逆序、末点不一致的轨迹无效，不能获得虚假短路径奖励。", "",
              "穿门的严格验收是base参考轨迹从开口向前跨越、base到达门后至少0.8米并停稳，加上碰撞／跌倒记录；没有单独验收完整机器人包络或整机扫掠净空。", ""]

    lines += ["## 3. 总分、严格成功率与完整性", ""]
    totals = []
    for name in MODEL_DIRS:
        r = reports[name]
        successes, failures, invalid, pending = counts(r)
        formal = formal_models[name]
        state = "正式单模型结果" if formal else "阶段观察（不排名）" if stage_requested and name == "qwen" else "进行中／不可排名"
        observed = f"{successes}/{successes + failures}（{percent(successes / (successes + failures))}）" if successes + failures else "不可用（0个有效样本）"
        totals.append((r.get("model", {}).get("label", name), state,
                       fmt(r.get("score_100")) if formal else "不可用", "不适用（已正式）" if formal else fmt(r.get("provisional_score_100")),
                       fmt(r.get("strict_success_score_100")), percent(r.get("micro_success_rate")),
                       observed, successes, failures, invalid, pending))
    lines += table(["模型", "状态", "正式综合分", "临时零填综合分", "计划宏平均SR百分分（未齐零填）", "计划槽位微平均SR（未齐零填）",
                    "已完成有效样本观察SR",
                    "成功", "有效失败", "invalid", "pending"], totals)
    lines += table(["模型", *(label + "分" for _, label in COMPONENTS)],
                   [(reports[n].get("model", {}).get("label", n), *component_cells(reports[n])) for n in MODEL_DIRS])
    if not rankable:
        lines += ["每个模型只有自身65回合有效完整并通过核查后才显示正式单模型分；另一模型未齐时仍不能作总体排名。未齐模型的计划宏平均SR、计划槽位微平均SR和临时分均按完整65槽位零填，pending／invalid没有被删除；它们不是该模型的正式成功率或正式分。", ""]
    lines += ["**已完成有效样本观察成功率＝成功回合数／有效completed回合数。** invalid和pending均不进入这一观察分母；它仅描述已完成有效样本，不替代正式评分、不用于模型间排名。DeepSeek完整65回合后，该观察率与其正式整体微平均成功率一致。", ""]
    lines += repeat_stability_section(plans, reports)

    lines += ["## 4. 四类任务分组", ""]
    rows = []
    for group, group_label in GROUP_NAMES.items():
        for name in MODEL_DIRS:
            value = reports[name].get("groups", {}).get(group, {})
            rows.append((group_label + " / " + group, reports[name].get("model", {}).get("label", name),
                         f"{value.get('successes', '未知')}/{value.get('expected', '未知')}",
                         percent(value.get("success_rate")), fmt(value.get("score_100")), *component_cells(value)))
    lines += table(["组别", "模型", "成功/计划", "严格SR", "综合分", *(label for _, label in COMPONENTS)], rows)

    lines += ["## 5. 逐任务结果", "", "每项成功数的标准分母为5；共13项，每模型合计65回合。未完成时分数为临时零填口径。", ""]
    rows = []
    for task in tasks:
        for name in MODEL_DIRS:
            task_id = task.get("task_id")
            value = reports[name].get("tasks", {}).get(task_id, {})
            rows.append((task_id, reports[name].get("model", {}).get("label", name),
                         f"{value.get('successes', '未知')}/{value.get('expected', '未知')}",
                         percent(value.get("success_rate")), fmt(value.get("score_100")), *component_cells(value)))
    lines += table(["任务", "模型", "成功/计划", "严格SR", "综合分", *(label for _, label in COMPONENTS)], rows)

    lines += ["## 6. 原始路径、运行时间与调用", "",
              "下表为每回合均值，括号n是该原始量的已知样本数；API累计延迟是单回合各已返回请求延迟之和，缺失失败请求时标未知。累计转向仅旁报，不加进总分。", ""]
    maps = {name: case_map(reports[name]) for name in MODEL_DIRS}
    common = sorted(set.intersection(*[{key for key, r in rows.items()
                                      if r.get("status") == "completed" and r.get("strict_success") is True}
                                     for rows in maps.values()]))
    efficiency = []
    for name in MODEL_DIRS:
        valid = [r for r in maps[name].values() if r.get("status") == "completed"]
        subsets = (("全部有效（含失败）", valid),
                   ("该模型成功", [r for r in valid if r.get("strict_success") is True]),
                   ("两模型共同成功", [maps[name][key] for key in common]))
        for label, subset in subsets:
            efficiency.append((reports[name].get("model", {}).get("label", name), label, len(subset),
                               *(measured(subset, key) for key, _ in RAW_METRICS)))
    lines += table(["模型", "子集", "回合数", *(label for _, label in RAW_METRICS)], efficiency)
    lines += ["以下累计值只包括有效回合中已记录的原始量，不冒充基础设施无效请求的完整成本：", ""]
    lines += table(["模型", *(label + "累计" for _, label in RAW_METRICS[:5])],
                   [(reports[name].get("model", {}).get("label", name),
                     *(measured([r for r in maps[name].values() if r.get("status") == "completed"], key, total=True)
                       for key, _ in RAW_METRICS[:5])) for name in MODEL_DIRS])
    lines += ["失败早停也可能使全体样本平均时间和路径较小，不能据此宣称效率更高；跨模型效率应结合共同成功case子集，且注明其样本数。子集统计不改变综合分的完整计划分母。", ""]

    lines += ["## 7. 配对回合比较", ""]
    a, b = MODEL_DIRS
    paired = []
    outcome = Counter()
    strict_pairs = Counter()
    detail = []
    for planned in grid:
        key = planned.get("case_id")
        left, right = maps[a].get(key, {}), maps[b].get(key, {})
        x, y = left.get("episode_score_100"), right.get("episode_score_100")
        if left.get("status") == right.get("status") == "completed" and number(x) and number(y):
            paired.append(key)
            result = "DeepSeek胜" if x > y else "Qwen胜" if x < y else "平"
            outcome[result] += 1
            strict_pairs[(left.get("strict_success") is True, right.get("strict_success") is True)] += 1
        else:
            result = "尚不可配对（pending／invalid或缺分数）"
        detail.append((key, left.get("status", "缺失"), "是" if left.get("strict_success") is True else "否",
                       fmt(x), right.get("status", "缺失"), "是" if right.get("strict_success") is True else "否",
                       fmt(y), result))
    lines += table(["已完成有效配对", "DeepSeek分数较高", "同分", "Qwen分数较高", "共同成功"],
                   [(len(paired), outcome["DeepSeek胜"], outcome["平"], outcome["Qwen胜"], len(common))])
    lines += table(["两者均成功", "仅DeepSeek成功", "仅Qwen成功", "两者均失败"],
                   [(strict_pairs[(True, True)], strict_pairs[(True, False)], strict_pairs[(False, True)], strict_pairs[(False, False)])])
    lines += ["配对胜平负按保存的未显示舍入前回合分比较，仅描述已完成的有效配对；双失败通常同为0分。它不按四类权重汇总，不等同于正式总榜。未完成实验的这些数量不用于宣布最终赢家。", ""]
    lines += table(["case", "DeepSeek状态", "严格成功", "综合分", "Qwen状态", "严格成功", "综合分", "回合分比较"], detail)

    lines += ["## 8. 终止、错误与已知token用量", ""]
    for title, diagnostic_key in (("终止类型", "termination_counts"), ("错误代码", "error_counts")):
        keys = sorted({str(key) for report in reports.values()
                       for key in report.get("diagnostics", {}).get(diagnostic_key, {})})
        lines += table([title, "DeepSeek", "Qwen"],
                       [(key, *(reports[n].get("diagnostics", {}).get(diagnostic_key, {}).get(key, 0)
                                for n in MODEL_DIRS)) for key in keys] or [("尚无记录", "未知", "未知")])
    rows = []
    for name in MODEL_DIRS:
        diagnostic = reports[name].get("diagnostics", {})
        usage = diagnostic.get("known_usage", {})
        rows.append((reports[name].get("model", {}).get("label", name), diagnostic.get("collision_episodes", "未知"),
                     diagnostic.get("fallen_episodes", "未知"), diagnostic.get("tool_calls", "未知"),
                     percent(diagnostic.get("protocol_error_rate")),
                     fmt(usage.get("prompt_tokens"), 0), fmt(usage.get("completion_tokens"), 0),
                     fmt(usage.get("total_tokens"), 0), "完整" if diagnostic.get("usage_complete") is True else "未知／不完整"))
    lines += table(["模型", "碰撞回合", "跌倒回合", "工具结果数", "协议错误率", "已知输入token", "已知输出token", "已知总token", "用量完整度"], rows)
    lines += ["有效失败包括abort、超预算、独立评分未通过，以及固定API超时 `MODEL_TIMEOUT` 和响应／工具协议错误 `MODEL_PROTOCOL_ERROR`。HTTP、鉴权、限流、服务端、网络传输、配置、未知错误以及损坏证据保守归为invalid，不把接口不适配归咎为模型规划差。", "",
              "已知usage只统计已返回并被记录的计数。失败请求可能已经产生计算与费用却没有可用usage；未知或不完整不等于0 token或零费用。没有可信价格及完整计费口径，不报告美元成本；不同tokenizer的token也不代表等量计算。Token、接口错误率及累计转向量均为旁报指标，不再加进综合分。", ""]
    lines += failure_detail_section(failure_analysis, plans, reports)

    lines += ["## 9. 真实API预检与正式评测的边界", ""]
    preflight_rows, request_rows, adapter_rows = [], [], []
    for name, filename in (("deepseek", "deepseek-flash.json"), ("qwen", "qwen3.8-max.json")):
        preflight = read_object(directory / "preflight" / filename, [], snapshots, required=False)
        if preflight is None:
            preflight_rows.append((name, "未读取到预检记录", "未知", "未知", "未知", "未知", "未知"))
        else:
            calls = preflight.get("tool_calls", [])
            preflight_rows.append((name, preflight.get("kind"), preflight.get("http_status", preflight.get("status_code")),
                                   preflight.get("passed"), preflight.get("reasoning_chars"),
                                   len(calls) if isinstance(calls, list) else "未知", preflight.get("is_benchmark_episode")))
            options = preflight.get("request_options", {})
            request_rows.append((name, json.dumps({k: options[k] for k in
                                 ("thinking", "enable_thinking", "preserve_thinking", "max_tokens", "max_completion_tokens")
                                 if k in options}, ensure_ascii=False)))
        adapter_file = directory / "preflight" / filename.replace(".json", "-adapter.json")
        adapter = read_object(adapter_file, [], snapshots, required=False)
        if adapter is not None:
            adapter_rows.append((name, adapter.get("kind"), adapter.get("passed"), adapter.get("token_budget_field"),
                                 json.dumps({k: v for k, v in adapter.get("wire_parameters", {}).items()
                                             if k in {"thinking", "enable_thinking", "preserve_thinking"}}, ensure_ascii=False),
                                 adapter.get("is_benchmark_episode")))
    lines += table(["模型", "记录类型", "HTTP状态", "通过", "思考字符数", "工具请求数", "是否正式episode"], preflight_rows)
    if request_rows:
        lines += table(["模型", "预检记录中的思考与预算请求字段"], request_rows)
    if adapter_rows:
        lines += table(["模型", "适配器多轮预检", "通过", "输出预算字段", "思考协议字段", "是否正式episode"], adapter_rows)
    preflight_summary = read_object(directory / "preflight" / "summary.json", [], snapshots, required=False)
    if preflight_summary is not None:
        summary_models = preflight_summary.get("models", {})
        fields = ("requests", "prompt_tokens", "completion_tokens", "total_tokens")
        totals = {key: sum(value[key] for value in summary_models.values())
                  if summary_models and all(isinstance(value, dict) and number(value.get(key))
                                            for value in summary_models.values()) else None for key in fields}
        lines += table(["预检请求模型", "真实API请求数", "已记录输入token", "已记录输出token", "已记录总token"],
                       [(name, *(fmt(value.get(key), 0) for key in fields))
                        for name, value in summary_models.items() if isinstance(value, dict)]
                       + [("预检合计（独立于正式评测）", *(fmt(totals[key], 0) for key in fields))])
        lines += [f"[预检汇总原始记录](preflight/summary.json)记录共{fmt(totals['requests'], 0)}次真实API请求，全部通过标志为{cell(preflight_summary.get('all_passed'))}。这些预检请求及其token独立列示，**不计入每模型65个正式回合、主分或第8节正式评测token汇总**。", ""]
        if diagnostic is not None:
            outside_requests = totals["requests"] + 1 if number(totals["requests"]) else None
            lines += table(["正式评测之外的用途", "请求数", "已记录总token", "用量说明"],
                           [("能力预检（原汇总）", fmt(totals["requests"], 0), fmt(totals["total_tokens"], 0), "保留原预检统计"),
                            ("独立HTTP400诊断", 1, "未知", "未返回可用usage，不按0处理"),
                            ("非正式请求合计", fmt(outside_requests, 0), "不完整", "仅能力预检的已知token可核对，诊断token未知")])
            lines += ["额外的[1次服务错误诊断](preflight/http400-diagnostic.json)不并入原能力预检的请求／token汇总；两类请求都不属于正式65回合。非正式请求总token不完整，不能把能力预检的已知总量冒充全部请求成本。", ""]
    elif diagnostic is not None:
        lines += ["另有[1次独立HTTP400诊断](preflight/http400-diagnostic.json)，不属于正式65回合；其token未知，不按0处理。未读取到能力预检汇总时，不推算全部非正式请求数或成本。", ""]
    lines += ["图片＋思考＋工具预检只证明该请求组合可以响应，不验证机器人任务成功，也不计入65回合、主分或正式token汇总。预检通过不保证后续服务持续可用。Qwen商业思考模型支持非流式返回；适配参数以官方文档和原始请求元信息为准。[百炼非流式思考说明](" + OFFICIAL["qwen_thinking"] + ")", ""]

    lines += ["## 10. 环境、协议指纹与输入完整性", ""]
    test_result = environment.get("unit_tests_before_run", {})
    test_count = test_result.get("passed") if isinstance(test_result, dict) else None
    lines += table(["环境记录项目", "值"],
                   [("平台", environment.get("platform")), ("Python", environment.get("python")),
                    ("CPU逻辑核数", environment.get("cpu_count")), ("渲染方式", environment.get("rendering")),
                    ("同时运行的正式回合数", environment.get("concurrent_benchmark_episodes")),
                    ("正式运行前单元测试通过数", fmt(test_count, 0)),
                    ("单元测试是否真实调用模型", test_result.get("real_model_calls") if isinstance(test_result, dict) else None)])
    if (directory / "frozen-source").is_dir():
        lines += ["冻结源码归档：[frozen-source/](frozen-source/)。环境记录：[environment.json](environment.json)。归档与运行时源码hash用于核对实验使用的版本。", ""]
    else:
        lines += ["未发现 `frozen-source/` 源码归档目录；环境记录如存在则见 [environment.json](environment.json)。", ""]
    lines += table(["项目", "DeepSeek", "Qwen"],
                   [("评分协议", *(plans[n].get("protocol", {}).get("id") for n in MODEL_DIRS)),
                    ("协议SHA256", *(plans[n].get("protocol_sha256") for n in MODEL_DIRS)),
                    ("Python", *(plans[n].get("protocol", {}).get("python_version") for n in MODEL_DIRS)),
                    ("资产lock SHA256", *(plans[n].get("protocol", {}).get("asset_lock_sha256") for n in MODEL_DIRS)),
                    ("系统提示SHA256", *(plans[n].get("protocol", {}).get("system_prompt_sha256") for n in MODEL_DIRS)),
                    ("工具schema SHA256", *(plans[n].get("protocol", {}).get("tools_sha256") for n in MODEL_DIRS)),
                    ("源码hash映射的SHA256", *(digest(plans[n].get("protocol", {}).get("code_sha256", {})) for n in MODEL_DIRS))])
    lines += [f"实验运行脚本冻结SHA256：{cell(experiment.get('runner_sha256'))}。本次报告生成器SHA256：{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}。资产lock指纹记录了资产版本绑定；不能仅凭这一字段声称重新校验了每个资产文件。", ""]
    packages = sorted({key for p in protocols if isinstance(p, dict) for key in p.get("package_versions", {})})
    lines += table(["依赖", "DeepSeek计划", "Qwen计划"],
                   [(package, *(plans[n].get("protocol", {}).get("package_versions", {}).get(package) for n in MODEL_DIRS)) for package in packages])
    code_files = sorted({key for p in protocols if isinstance(p, dict) for key in p.get("code_sha256", {})})
    lines += ["<details>", "<summary>冻结源码SHA256</summary>", ""]
    lines += table(["源码", "DeepSeek计划", "Qwen计划"],
                   [(file, *(plans[n].get("protocol", {}).get("code_sha256", {}).get(file) for n in MODEL_DIRS)) for file in code_files])
    lines += ["</details>", "", "本生成器检查计划／报告／比较摘要的模型、协议、case身份与计数一致性，不重新运行物理仿真、不重打独立目标分，也不构成防篡改认证。运行时独立评测器与离线benchmark评分器的证据检查仍是正式评分依据。", ""]
    if issues:
        lines += ["### 当前完整性问题", "", *("- " + cell(issue) for issue in issues), ""]
    else:
        lines += ["计划、报告与比较摘要在本次读取快照中一致。", ""]
    score_issues = [(name, issue) for name, report in reports.items() for issue in report.get("issues", [])]
    if score_issues:
        lines += table(["模型", "离线评分器问题记录"], score_issues)
    lines += ["<details>", "<summary>本次报告读取的JSON快照SHA256与文件链接</summary>", ""]
    lines += table(["输入文件", "SHA256"],
                   [(f"[{path.relative_to(directory).as_posix()}]({path.relative_to(directory).as_posix()})", checksum)
                    for path, checksum in snapshots])
    lines += ["</details>", ""]
    lines += fixed_image_section(directory)
    lines += ["## 12. 结论适用范围", "",
              "5次重复是小样本，固定的13个场景也不足以证明跨场景泛化。报告不编造置信区间、统计显著性或美元成本，不把小分差称作稳定领先。墙钟效率还受到模型服务、网络、本机执行和负载影响；串行交错顺序减小时段偏差，不能消除全部混杂因素。", "",
              "当前系统没有全局自动绕障规划、独立Task Agent或跨回合长期记忆；模型通过同一套工具闭环决策。感知使用仿真标定与理想化本体信息，未证明真实传感器噪声、实机迁移或实时控制效果。本轮导航评分不能说明静止姿态专项能力、保持低姿态移动或爬行能力。", "",
              "只有两模型完整、无invalid、同协议且原始证据通过检查，才可讨论本协议下的最终比较；权重、提示、工具、预算或控制器变化需要新计划，不能混用已有成绩。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    directory = args.experiment_dir.expanduser().resolve()
    if not directory.is_dir():
        parser.error("--experiment-dir must be an existing experiment directory")
    try:
        contents = render(directory)
        destination = directory / "REPORT.zh-CN.md"
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(contents, encoding="utf-8")
        temporary.replace(destination)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        print(f"Report generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Report: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
