"""Deterministic public event memory checks; no model, API or simulator."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from go2_demo.general_memory import EpisodeMemory

FIELDS = {"observation": {"local_odometry", "grounded_targets", "robot_state", "task_feedback", "target_ids"},
          "action": {"requested_tool", "requested_arguments", "feedback", "before_pose", "after_pose", "target_id", "target_ids"}}


def select_all(question, candidates):
    return {"event_ids": [item["event_id"] for item in candidates]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="New JSON report path; never overwrites")
    args = parser.parse_args()
    data = json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))
    rows = []
    for case in data["cases"]:
        store = EpisodeMemory(public_fields=FIELDS)
        for event in data["events"]:
            store.append(**event)
        def selector(question, candidates):
            return select_all(question, candidates[-1:] if case["selection"] == "latest" else candidates)
        scope = {k: case[k] for k in ("event_types", "target_ids") if k in case}
        result = store.query(case["question"], selector, **scope)
        repeated = store.query(case["question"], None, **scope)
        ids = result["card"]["event_ids"] if result["card"] else []
        rows.append({"id": case["id"], "description": case["description"], "expected": case["expected"],
                     "actual": ids, "candidate_count": result["candidate_count"], "reader_called": result["reader_called"],
                     "repeat_cache_hit": repeated["cache_hit"],
                     "passed": ids == case["expected"] and repeated["cache_hit"] and not repeated["reader_called"]})
    store = EpisodeMemory(public_fields=FIELDS)
    store.append(**data["events"][0])
    first = store.query("B?", select_all, target_ids=["B"])
    store.append(**data["events"][1])
    unrelated = store.query("B?", None, target_ids=["B"])
    store.append(**data["events"][3])
    changed = store.query("B?", select_all, target_ids=["B"])
    rows.append({"id": "L07", "description": "无关事件保留缓存，相关事件使缓存失效",
                 "passed": unrelated["cache_hit"] and not changed["cache_hit"] and changed["reader_called"]
                 and first["evidence_version"] != changed["evidence_version"]})
    report = {"mode": "synthetic_scripted_selector", "passed": sum(r["passed"] for r in rows),
              "total": len(rows), "results": rows}
    for row in rows:
        print(f"{row['id']} {'PASS' if row['passed'] else 'FAIL'} {row['description']}")
    print(f"Passed {report['passed']}/{report['total']}; no model/API used.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return int(report["passed"] != report["total"])


if __name__ == "__main__":
    raise SystemExit(main())
