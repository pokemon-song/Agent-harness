"""Offline queue/context regression replay on a copy; no model API or simulation."""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from go2_demo.md_memory.context import document_context, encoded_bytes, project_event
from go2_demo.md_memory.store import MDMemoryStore

DEFAULT_SOURCE = ROOT / "runs/20261110-202117-md-complex-memory-27b735"


def tree_hashes(root):
    """Only inspect regular files; a replay must never follow evidence symlinks."""
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise ValueError("Replay source cannot contain symlinks")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def size(value):
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return {"characters": len(serialized), "utf8_bytes": encoded_bytes(value)}


def size_comparison(before, after):
    old, new = size(before), size(after)
    return {"before": old, "after": new,
            "byte_reduction_fraction": 1 - new["utf8_bytes"] / old["utf8_bytes"]}


def brief(snapshot):
    return {key: snapshot[key] for key in (
        "revision", "processed_event_seq", "acknowledged_event_seqs", "pending_event_count")}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def build_replay(source_run, output_directory, *, phase="fragments", target_seq=35, batch_limit=8):
    """Copy a completed run's memory, then fixture-acknowledge its pending queue.

    Empty changes deliberately test queue acknowledgement only. They do not
    assert that an LLM understood, summarized, or correctly remembered an event.
    The historical source is never opened by MDMemoryStore or by SQLite here.
    """
    if not isinstance(phase, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", phase):
        raise ValueError("phase must be a simple directory name")
    if type(target_seq) is not int or target_seq < 1:
        raise ValueError("target_seq must be a positive integer")
    if type(batch_limit) is not int or not 1 <= batch_limit <= 100:
        raise ValueError("batch_limit must be in 1..100")
    source_run = Path(source_run).resolve(strict=True)
    source_memory = source_run / "memory" / phase
    requested_output = Path(output_directory).absolute()
    output = requested_output.resolve()
    if output.is_relative_to(source_run):
        raise ValueError("Replay output must be outside the historical source run")
    if output.exists() or requested_output.is_symlink():
        raise FileExistsError("Replay output must be a new directory")
    if not source_memory.is_dir() or source_memory.resolve() != source_memory:
        raise ValueError("Source phase has no regular memory directory")
    # Copying a live WAL database is not a reproducible historical replay.
    if any((source_memory / name).exists() for name in (
            "index.sqlite3-wal", "index.sqlite3-shm", "index.sqlite3-journal")):
        raise ValueError("Source must be a closed historical memory database")
    before_hashes = tree_hashes(source_memory)
    if "index.sqlite3" not in before_hashes:
        raise ValueError("Source memory has no event index")
    output.mkdir(parents=True, exist_ok=False)
    copied_memory = output / "memory-copy"
    shutil.copytree(source_memory, copied_memory)
    if tree_hashes(copied_memory) != before_hashes:
        raise RuntimeError("Memory copy did not preserve every source artifact")
    # Even scope discovery reads the copy, never the historical SQLite file.
    with closing(sqlite3.connect((copied_memory / "index.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
        scope = db.execute("SELECT scope FROM settings WHERE id=1").fetchone()[0]
        event_count = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    store = MDMemoryStore(copied_memory, scope=scope)
    initial = store.snapshot()
    if initial["pending_event_count"] > 100:
        raise ValueError("This bounded replay supports at most 100 pending events")
    pending = store.pending_events(limit=100)
    target = next((event for event in pending if event["seq"] == target_seq), None)
    if target is None or not target["critical"]:
        raise ValueError("The target must be a pending critical event in the source snapshot")
    fifo = store.pending_events(limit=batch_limit, prioritize_critical=False)
    priority = store.pending_events(limit=batch_limit, prioritize_critical=True)
    projected = [project_event(event) for event in pending]
    documents = [store.read(entry["path"]) for entry in initial["files"]]
    initial_document_hashes = {entry["path"]: entry["sha256"] for entry in initial["files"]}
    selected = [event["seq"] for event in priority]
    store.commit(initial["revision"], max(selected), [], event_seqs=selected,
                 call_id="offline-fixture-priority-ack")
    after_priority = store.snapshot()
    remaining = store.pending_events(limit=100)
    pending_sequences = [event["seq"] for event in pending]
    remaining_sequences = [event["seq"] for event in remaining]
    acknowledgements = [selected]
    while store.snapshot()["pending_event_count"]:
        batch = store.pending_events(limit=batch_limit, prioritize_critical=True)
        sequences = [event["seq"] for event in batch]
        store.commit(store.snapshot()["revision"], max(sequences), [], event_seqs=sequences,
                     call_id=f"offline-fixture-drain-{len(acknowledgements)}")
        acknowledgements.append(sequences)
    final = store.snapshot()
    acknowledged = [seq for batch in acknowledgements for seq in batch]
    checks = {
        "critical_target_in_first_priority_batch": target_seq in selected,
        "fifo_first_batch_misses_target": target_seq not in [event["seq"] for event in fifo],
        "unselected_events_remain_pending": remaining_sequences == [seq for seq in pending_sequences if seq not in selected],
        "priority_did_not_jump_contiguous_prefix": after_priority["processed_event_seq"] == initial["processed_event_seq"],
        "critical_projection_verbatim": all(project_event(event) == event for event in pending if event["critical"]),
        "all_pending_acknowledged_exactly_once": sorted(acknowledged) == pending_sequences,
        "queue_drained": final["pending_event_count"] == 0,
        "published_semantic_documents_unchanged": initial_document_hashes == {
            entry["path"]: entry["sha256"] for entry in final["files"]},
        "source_memory_unchanged": tree_hashes(source_memory) == before_hashes,
    }
    report = {
        "experiment_kind": "md-memory-offline-regression-v1",
        "is_llm_result": False, "model_api_requests": 0, "simulator_used": False,
        "fixture_acknowledgements_only": True,
        "source_run": str(source_run), "source_phase": phase,
        "source_event_count": event_count, "source_memory_hashes": before_hashes,
        "copied_memory": str(copied_memory), "target_seq": target_seq,
        "initial": brief(initial), "after_priority": brief(after_priority), "final": brief(final),
        "old_fifo_first_batch": [event["seq"] for event in fifo],
        "new_priority_first_batch": selected,
        "fifo_batch_number_for_target": pending_sequences.index(target_seq) // batch_limit + 1,
        "older_pending_feedback_count": sum(event["seq"] < target_seq and event["kind"] == "robot_feedback" for event in pending),
        "pending_after_priority": remaining_sequences,
        "fixture_acknowledgement_batches": acknowledgements,
        "input_sizes": {
            "first_batch_selection_and_projection": size_comparison(fifo, [project_event(event) for event in priority]),
            "same_fifo_batch_projection_only": size_comparison(fifo, [project_event(event) for event in fifo]),
            "all_pending_projection_only": size_comparison(pending, projected),
            "document_read_context": size_comparison(documents, [document_context(document) for document in documents]),
        },
        "checks": checks, "passed": all(checks.values()),
        "limitations": [
            "Deterministic fixture queue acknowledgements are not model decisions or semantic memory extraction.",
            "Counts measure compact JSON text only; they exclude prompts, tool schemas, history, image encodings and provider tokenization.",
            "Changing event selection and projecting the same events are separate comparisons; neither establishes task success, latency, token cost or LLM quality.",
            "Historical Markdown is preserved without asserting that its semantic contents are correct.",
        ],
    }
    write_json(output / "replay.json", report)
    text = ["# Markdown 记忆离线回归回放", "",
            "这是确定性 fixture 回放，模型 API 调用为 0，未运行仿真。只修改新目录里的记忆副本。", "",
            f"来源：`{source_run.name}/{phase}`；目标公告 seq={target_seq}，前方积压普通反馈 {report['older_pending_feedback_count']} 条。",
            f"旧 FIFO 首批：`{report['old_fifo_first_batch']}`；优先队列首批：`{selected}`。",
            f"确认首批后，连续水位仍为 {after_priority['processed_event_seq']}；未选中的 {len(remaining)} 条事件仍待处理。", "",
            "| 对比 | 原始 JSON 字节 | 新 JSON 字节 | 缩减 |", "|---|---:|---:|---:|"]
    labels = {"first_batch_selection_and_projection": "首批选择及投影（事件集合不同）",
              "same_fifo_batch_projection_only": "同一旧首批，仅字段投影",
              "all_pending_projection_only": "全部待处理事件，仅字段投影",
              "document_read_context": "已发布文档读取，移除重复 content"}
    for key, item in report["input_sizes"].items():
        text.append(f"| {labels[key]} | {item['before']['utf8_bytes']} | {item['after']['utf8_bytes']} | {item['byte_reduction_fraction']:.1%} |")
    text += ["", "这些是 JSON 文本大小，不能当作真实 token、延迟、成本或任务成绩。系统提示、工具 schema、历史和图像编码不在统计内。", "",
             "fixture 使用 `changes=[]` 仅验证稀疏事件确认及排空队列；不声称公告已被模型理解或写入语义记忆。副本中的原有 MD 正文及原始历史文件均保留。", "",
             f"全部检查：{'通过' if report['passed'] else '未通过'}；源记忆 {len(before_hashes)} 个文件 SHA256 保持不变。完整统计见 [replay.json](replay.json)。", ""]
    with (output / "REPORT.zh-CN.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(text))
    if tree_hashes(source_memory) != before_hashes:
        raise RuntimeError("Historical source changed during replay")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", nargs="?", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path,
                        default=Path(tempfile.gettempdir()) / ("md-memory-replay-" + uuid.uuid4().hex[:12]))
    parser.add_argument("--phase", default="fragments")
    parser.add_argument("--target-seq", type=int, default=35)
    parser.add_argument("--batch-limit", type=int, default=8)
    args = parser.parse_args(argv)
    report = build_replay(args.source_run, args.output, phase=args.phase,
                          target_seq=args.target_seq, batch_limit=args.batch_limit)
    print(json.dumps({"output": str(args.output), "passed": report["passed"],
                      "is_llm_result": False, "model_api_requests": 0,
                      "old_fifo_first_batch": report["old_fifo_first_batch"],
                      "new_priority_first_batch": report["new_priority_first_batch"],
                      "input_sizes": report["input_sizes"]}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
