"""Public-event integrity, scoped incremental retrieval and callback isolation."""
from copy import deepcopy
import json

import pytest

from go2_demo.general_memory import EpisodeMemory, InvalidMemorySelection, MemoryLimitError


FIELDS = {"observation": {"target_id", "target_ids", "pose", "targets", "note", "measurement"},
          "action_result": {"tool", "ok", "target_id"}, "instruction": {"note"}}


def memory(**limits):
    return EpisodeMemory(public_fields=FIELDS, **limits)


def observe(store, step, target="valve/东-7", **data):
    return store.append("observation", step, f"camera-{step}", {"target_id": target, **data})


class Select:
    def __init__(self, ids=None, answer=None):
        self.ids, self.answer, self.calls = ids, answer, []

    def __call__(self, question, candidates):
        self.calls.append((question, deepcopy(candidates)))
        reply = {"event_ids": self.ids if self.ids is not None else [c["event_id"] for c in candidates]}
        if self.answer is not None:
            reply["answer"] = self.answer
        return reply


def test_public_whitelist_rejects_private_inputs_without_dropping_fields_or_consuming_ids():
    policy = {"observation": {"target_id", "pose"}}
    store = EpisodeMemory(public_fields=policy)
    policy["observation"].add("evaluator_answer")
    with pytest.raises(ValueError, match="whitelist"):
        store.append("observation", 0, "rgb-0", {"target_id": "cabinet-19", "evaluator_answer": True})
    with pytest.raises(ValueError, match="whitelist"):
        store.append("private_evaluation", 0, None, {})
    event = store.append("observation", 0, "rgb-0", {"target_id": "cabinet-19", "pose": [1., 2.]})
    assert event["event_id"] == "E000001"
    assert event["payload"] == {"kind": "observation", "step": 0, "observation_id": "rgb-0",
                                 "data": {"target_id": "cabinet-19", "pose": [1., 2.]}}


def test_duplicate_event_identity_is_idempotent_but_conflicting_facts_are_rejected():
    store = memory()
    first = observe(store, 4, measurement=12.)
    version = store.snapshot()["evidence_version"]
    assert observe(store, 4, measurement=12.) == first
    assert store.snapshot()["event_count"] == 1 and store.snapshot()["evidence_version"] == version
    with pytest.raises(ValueError, match="Conflicting"):
        observe(store, 4, measurement=13.)
    second = store.append("action_result", 4, "camera-4", {"tool": "stop", "ok": True})
    third = store.append("observation", 5, "camera-4", {"target_id": "socket#Ω", "measurement": 13.})
    assert [e["event_id"] for e in store.snapshot()["events"]] == ["E000001", "E000002", "E000003"]
    assert second["payload"]["observation_id"] == third["payload"]["observation_id"] == "camera-4"


def test_arbitrary_targets_use_exact_explicit_metadata_including_nested_and_numeric_ids():
    store = memory()
    a = observe(store, 0, "rack/Ω-17")
    b = store.append("observation", 1, "rgb-multi", {"targets": [{"target_id": "tap:北/2"}, {"target_id": 73}]})
    c = store.append("observation", 2, "rgb-many", {"target_ids": ["item-δ", "73"]})
    observe(store, 3, "other", note="The prose mentions rack/Ω-17 and tap:北/2, but is not target metadata.")
    assert store.query("where?", Select(), target_ids=["rack/Ω-17"])["card"]["event_ids"] == [a["event_id"]]
    assert store.query("where?", Select(), target_ids=[73])["card"]["event_ids"] == [b["event_id"]]
    assert store.query("where?", Select(), target_ids=["73"])["card"]["event_ids"] == [c["event_id"]]
    assert store.query("where?", Select(), target_ids=["TAP:北/2"])["card"] is None


def test_scope_is_intersection_and_reader_never_receives_unmatched_public_events():
    store = memory()
    selected = observe(store, 0, "device-27")
    observe(store, 1, "device-28")
    store.append("action_result", 2, None, {"tool": "stop", "target_id": "device-27", "ok": True})
    reader = Select()
    result = store.query("Find evidence about device-27", reader, event_types=["observation"], target_ids=["device-27"])
    assert reader.calls == [("Find evidence about device-27", [selected])]
    assert result["candidate_count"] == 1 and result["reader_called"] and not result["cache_hit"]
    assert result["card"]["evidence"] == [selected]


def test_empty_scope_and_absent_evidence_do_not_invoke_a_reader():
    store = memory()
    reader = Select()
    first = store.query("any observation?", reader)
    assert first["card"] is None and not first["reader_called"] and not first["cache_hit"]
    assert store.query("any observation?", None)["cache_hit"]
    observe(store, 0)
    for kwargs in ({"event_types": []}, {"target_ids": []}, {"event_types": ["not-recorded"]}):
        result = store.query("nothing in this scope", reader, **kwargs)
        assert result["card"] is None and not result["reader_called"]
    assert not reader.calls


def test_incremental_matching_evidence_invalidates_cache_without_freezing_at_a_stage_boundary():
    store, reader = memory(), Select()
    observe(store, 0, "checkpoint-4", measurement=0.)
    first = store.query("latest checkpoint measurement?", reader, target_ids=["checkpoint-4"])
    assert store.query("latest checkpoint measurement?", None, target_ids=["checkpoint-4"])["cache_hit"]
    observe(store, 1, "checkpoint-5", measurement=9.)
    unrelated = store.query("latest checkpoint measurement?", None, target_ids=["checkpoint-4"])
    assert unrelated["cache_hit"] and unrelated["evidence_version"] == first["evidence_version"]
    # A new instruction/task stage does not freeze the existing event store.
    store.append("instruction", 2, None, {"note": "Continue the second stage"})
    latest = observe(store, 3, "checkpoint-4", measurement=7.)
    second = store.query("latest checkpoint measurement?", reader, target_ids=["checkpoint-4"])
    assert second["reader_called"] and not second["cache_hit"] and len(reader.calls) == 2
    assert second["evidence_version"] != first["evidence_version"]
    assert second["card"]["evidence"][-1] == latest and second["candidate_count"] == 2


def test_empty_result_cache_is_invalidated_when_the_first_matching_event_arrives():
    store, reader = memory(), Select()
    first = store.query("position?", reader, target_ids=["camera:unknown"])
    observe(store, 0, "elsewhere")
    assert store.query("position?", None, target_ids=["camera:unknown"])["cache_hit"]
    event = observe(store, 1, "camera:unknown")
    result = store.query("position?", reader, target_ids=["camera:unknown"])
    assert result["reader_called"] and not result["cache_hit"] and result["card"]["evidence"] == [event]
    assert first["evidence_version"] != result["evidence_version"]


def test_scope_order_and_duplicates_are_canonical_but_question_text_is_not_semantically_rewritten():
    store, reader = memory(), Select()
    observe(store, 0, "sensor-1")
    store.append("action_result", 1, None, {"target_id": "sensor-2", "ok": True})
    store.query("original question", reader, event_types=["action_result", "observation"], target_ids=["sensor-1", "sensor-2"])
    cached = store.query("original question", None, event_types=["observation", "action_result", "observation"],
                         target_ids=["sensor-2", "sensor-1"])
    assert cached["cache_hit"] and len(reader.calls) == 1
    store.query("original question ", reader, event_types=["observation", "action_result"], target_ids=["sensor-1", "sensor-2"])
    assert len(reader.calls) == 2 and reader.calls[-1][0] == "original question "


@pytest.mark.parametrize("selection", ["invented", "duplicate", "outside_scope", "extra_facts"])
def test_reader_cannot_forge_duplicate_or_select_out_of_scope_evidence_and_failure_is_not_cached(selection):
    store = memory()
    chosen, other = observe(store, 0, "door-1"), observe(store, 1, "door-2")
    def bad(question, candidates):
        if selection == "invented":
            return {"event_ids": ["E999999"]}
        if selection == "duplicate":
            return {"event_ids": [chosen["event_id"], chosen["event_id"]]}
        if selection == "outside_scope":
            return {"event_ids": [other["event_id"]]}
        return {"event_ids": [chosen["event_id"]], "evidence": [{"invented": True}]}
    with pytest.raises(InvalidMemorySelection) as error:
        store.query("door evidence?", bad, target_ids=["door-1"])
    assert error.value.reader_called
    retry = store.query("door evidence?", Select(), target_ids=["door-1"])
    assert retry["reader_called"] and not retry["cache_hit"] and retry["card"]["evidence"] == [chosen]


def test_maximum_24_selected_events_is_enforced_without_pruning_the_source_store():
    store = memory()
    events = [observe(store, step) for step in range(25)]
    with pytest.raises(InvalidMemorySelection):
        store.query("all?", Select())
    result = store.query("all?", Select([e["event_id"] for e in events[:24]]))
    assert len(result["card"]["evidence"]) == 24 and store.snapshot()["event_count"] == 25


def test_returned_events_candidates_cards_and_snapshots_cannot_mutate_source_facts():
    store = memory()
    original = {"target_id": "marker-5", "pose": [1., 2.]}
    event = store.append("observation", 0, "rgb0", original)
    original["pose"][0] = 100.
    event["payload"]["data"]["pose"][0] = 200.
    def reader(question, candidates):
        eid = candidates[0]["event_id"]
        candidates[0]["event_id"] = "E999999"
        candidates[0]["payload"]["data"]["pose"][0] = 300.
        return {"event_ids": [eid]}
    result = store.query("pose?", reader)
    assert result["card"]["evidence"][0]["payload"]["data"]["pose"] == [1., 2.]
    result["card"]["evidence"][0]["payload"]["data"]["pose"][0] = 400.
    snapshot = store.snapshot()
    snapshot["events"][0]["payload"]["data"]["pose"][0] = 500.
    assert store.query("pose?", None)["card"]["evidence"][0]["payload"]["data"]["pose"] == [1., 2.]


def test_callback_cannot_cite_an_event_appended_after_its_candidate_snapshot():
    store = memory()
    observe(store, 0)
    def reader(question, candidates):
        newly_added = observe(store, 1)
        return {"event_ids": [newly_added["event_id"]]}
    with pytest.raises(InvalidMemorySelection):
        store.query("what is known?", reader)
    next_result = store.query("what is known?", Select())
    assert next_result["reader_called"] and next_result["candidate_count"] == 2


def test_callback_budget_or_network_errors_propagate_and_do_not_cache_a_false_answer():
    store = memory()
    observe(store, 0)
    failure = RuntimeError("caller budget exhausted")
    def unavailable(question, candidates):
        raise failure
    with pytest.raises(RuntimeError) as error:
        store.query("where?", unavailable)
    assert error.value is failure
    assert store.query("where?", Select())["reader_called"]


def test_reader_interpretation_is_not_semantically_judged_or_promoted_into_facts():
    store = memory()
    event = observe(store, 0, measurement=3.)
    interpretation = "No previous reading exists; the value must be 99."
    result = store.query("reading?", Select(answer=interpretation))
    assert result["card"]["answer"] == interpretation and result["card"]["answer_is_model_generated"]
    assert result["card"]["evidence"] == [event]
    assert store.snapshot()["events"] == [event]
    empty = store.query("a different question", Select([], answer="I cannot cite a record."))
    assert empty["reader_called"] and empty["card"]["evidence"] == []


def test_event_and_store_limits_reject_whole_inputs_without_truncating_existing_facts():
    store = memory(max_event_bytes=256, max_store_bytes=300)
    first = store.append("observation", 0, "rgb0", {"note": "x" * 100})
    with pytest.raises(MemoryLimitError, match="max_event_bytes"):
        store.append("observation", 1, "rgb1", {"note": "x" * 300})
    with pytest.raises(MemoryLimitError, match="capacity"):
        store.append("observation", 1, "rgb1", {"note": "y" * 100})
    assert store.snapshot()["events"] == [first]
    assert store.append("observation", 0, "rgb0", {"note": "x" * 100}) == first


def test_large_candidate_set_requires_explicit_narrowing_instead_of_silent_truncation():
    store, reader = memory(max_candidate_bytes=250), Select()
    a, b = observe(store, 0, "sensor-1", note="a reading"), observe(store, 1, "sensor-2", note="a reading")
    with pytest.raises(MemoryLimitError, match="narrow"):
        store.query("readings?", reader)
    assert not reader.calls and store.snapshot()["events"] == [a, b]
    assert store.query("readings?", reader, target_ids=["sensor-2"])["card"]["evidence"] == [b]


def test_question_answer_and_event_count_limits_are_enforced_before_cache_commit():
    store, reader = memory(max_question_chars=10, max_answer_chars=10, max_events=1), Select(answer="x" * 11)
    observe(store, 0)
    with pytest.raises(MemoryLimitError):
        observe(store, 1)
    with pytest.raises(MemoryLimitError):
        store.query("q" * 11, reader)
    assert not reader.calls
    with pytest.raises(InvalidMemorySelection):
        store.query("short?", reader)
    assert store.query("short?", Select())["reader_called"]


def test_bounded_query_cache_evicts_answers_but_never_source_events():
    store, reader = memory(max_cache_entries=1), Select()
    event = observe(store, 0)
    for question in ("one?", "two?", "one?"):
        assert store.query(question, reader)["reader_called"]
    assert len(reader.calls) == 3 and store.snapshot()["events"] == [event]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), (1, 2), {1: "non-string-key"}, object()])
def test_non_json_values_are_rejected_without_normalization(value):
    store = memory()
    with pytest.raises(ValueError):
        observe(store, 0, measurement=value)
    assert store.snapshot()["event_count"] == 0


@pytest.mark.parametrize("kwargs", [{"target_ids": "sensor-1"}, {"event_types": "observation"},
                                   {"target_ids": [True]}, {"event_types": [1]}])
def test_invalid_scope_shapes_are_not_silently_broadened(kwargs):
    with pytest.raises(ValueError):
        memory().query("where?", Select(), **kwargs)


def test_json_markdown_exports_preserve_original_payloads_and_omit_reader_answers(tmp_path):
    store = memory()
    event = observe(store, 0, note='原始内容 ``` "quoted"\nsecond line', pose=[.25, -.5])
    store.query("what?", Select(answer="READER_INTERPRETATION_ONLY"))
    paths = store.write(tmp_path)
    saved = json.loads(paths["json"].read_text())
    text = paths["markdown"].read_text()
    assert saved == store.snapshot() and paths["json"].name == "memory.json" and paths["markdown"].name == "memory.md"
    body = text.split("````json\n", 1)[1].split("\n````", 1)[0]
    assert json.loads(body) == event["payload"]
    assert "READER_INTERPRETATION_ONLY" not in text and "READER_INTERPRETATION_ONLY" not in paths["json"].read_text()
    observe(store, 1)
    store.write(tmp_path)
    assert json.loads(paths["json"].read_text())["event_count"] == 2
    assert not list(tmp_path.glob(".memory-*.tmp"))
