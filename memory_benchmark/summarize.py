"""Extract per-run metrics; does not grade semantic relevance or causality."""
import argparse
import csv
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for run in args.runs:
        metrics, protocol = read(run / "metrics.json"), read(run / "protocol.json")
        feedback, tokens = [], {"planner": 0, "reader": 0}
        with (run / "events.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                if event.get("event") == "feedback" and event.get("tool") == "memory_query":
                    feedback.append(event.get("feedback", {}))
                if event.get("event") == "request_ended" and event.get("role") in tokens:
                    tokens[event["role"]] += (event.get("usage") or {}).get("total_tokens", 0) or 0
        req = metrics.get("requests", {})
        rows.append({"run": str(run), "task": protocol.get("task", {}).get("id"), "mode": protocol.get("mode"),
                     "seed": protocol.get("seed"), "success": metrics.get("success"), "termination": metrics.get("termination"),
                     "query_choices": metrics.get("tools_used", {}).get("memory_query", 0), "query_feedbacks": len(feedback),
                     "cache_observations": sum("cache_hit" in f for f in feedback),
                     "cache_hits": sum(f.get("cache_hit") is True for f in feedback),
                     "reader_called_feedbacks": sum(f.get("reader_called") is True for f in feedback),
                     "planner_requests": req.get("planner", 0), "reader_requests": req.get("reader", 0),
                     "planner_known_tokens": tokens["planner"], "reader_known_tokens": tokens["reader"],
                     "unknown_usage_requests": metrics.get("unknown_usage_requests"), "score": metrics.get("score"),
                     "collision_count": metrics.get("collision_count"), "elapsed_s": metrics.get("elapsed_s")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
