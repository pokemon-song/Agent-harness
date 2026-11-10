"""A current contract must be reconstructed across three independent shifts."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from go2_demo import complex_memory_mission as mission
from go2_demo.types import ToolCall


@pytest.fixture
def contract():
    return {
        "order_id": "order-fixture-main", "decoy_order_id": "order-fixture-other",
        "alias_to_destination": {"冷藏区": "C", "质检区": "A", "回收区": "B"},
        "old_alias": "冷藏区", "new_alias": "质检区",
        "old_route": "blue", "new_route": "yellow",
        "code_prefix": "007", "code_suffix": "042", "receipt_code": "007042",
        "old_revision": 1, "new_revision": 2,
        "decoy_alias": "回收区", "decoy_route": "blue", "decoy_receipt_code": "159263",
    }


@pytest.fixture
def route_spy(monkeypatch):
    calls = []

    def evaluate(trajectory, route, start_xy=(0, 0)):
        calls.append({"trajectory": deepcopy(trajectory), "route": route, "start_xy": start_xy})
        return {"success": trajectory == [{"passed_route": route}], "observed": deepcopy(trajectory)}

    monkeypatch.setattr(mission, "evaluate_route", evaluate)
    return calls


def arguments(contract, *, old=False):
    which = "old" if old else "new"
    return {"order_id": contract["order_id"], "dispatch_revision": contract[which + "_revision"],
            "destination": contract["alias_to_destination"][contract[which + "_alias"]],
            "receipt_code": contract["receipt_code"], "route": contract[which + "_route"]}


def make_receipt(contract, route=None):
    state = SimpleNamespace(trajectory=[{"passed_route": route or contract["new_route"]}])
    return mission.ComplexMissionReceipt(contract, lambda: state)


def call(values, call_id="submit"):
    return ToolCall(call_id, "submit_mission", values)


def test_generated_contract_has_bijection_and_actual_destination_and_route_changes():
    for _ in range(40):
        c = mission.make_contract()
        assert c["order_id"] != c["decoy_order_id"]
        assert set(c["alias_to_destination"]) == set(mission.ALIASES)
        assert set(c["alias_to_destination"].values()) == {"A", "B", "C"}
        assert c["old_alias"] != c["new_alias"]
        assert c["alias_to_destination"][c["old_alias"]] != c["alias_to_destination"][c["new_alias"]]
        assert {c["old_route"], c["new_route"]} == {"blue", "yellow"}
        assert c["receipt_code"] == c["code_prefix"] + c["code_suffix"]
        assert len(c["receipt_code"]) == 6 and c["receipt_code"].isascii() and c["receipt_code"].isdigit()
        assert c["old_revision"] == 1 and c["new_revision"] == 2


def test_generator_preserves_zero_fragments_and_does_not_encode_them_in_identifiers(monkeypatch):
    identities = iter(("aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"))
    monkeypatch.setattr(mission.secrets, "token_hex", lambda _: next(identities))
    monkeypatch.setattr(mission.secrets, "choice", lambda population: population[0])
    monkeypatch.setattr(mission.secrets, "randbelow", lambda _: 0)
    c = mission.make_contract()
    assert c["code_prefix"] == c["code_suffix"] == "000"
    assert c["receipt_code"] == "000000"
    assert c["order_id"] == "order-aaaaaaaaaaaaaaaa"
    assert c["decoy_order_id"] == "order-bbbbbbbbbbbbbbbb"


@pytest.mark.parametrize("field,value", [
    ("decoy_order_id", "order-fixture-main"),
    ("new_alias", "冷藏区"),
    ("new_route", "blue"),
    ("new_revision", True),
    ("new_revision", 1),
    ("new_revision", 101),
    ("receipt_code", "700042"),
    ("code_prefix", "００７"),
    ("code_suffix", 42),
    ("alias_to_destination", {"冷藏区": "A", "质检区": "A", "回收区": "B"}),
    ("alias_to_destination", {"冷藏区": [], "质检区": "A", "回收区": "B"}),
])
def test_bad_private_contract_rejected_before_any_task_can_be_exposed(contract, field, value):
    contract[field] = value
    with pytest.raises(ValueError):
        mission.ComplexMissionReceipt(contract, lambda: None)
    with pytest.raises(ValueError):
        mission.learning_instruction("dispatch", contract)


@pytest.mark.parametrize("changed_field,new_value,only_stage", [
    ("code_prefix", "918", "dispatch"),
    ("code_suffix", "365", "fragments"),
    ("new_alias", "回收区", "revision"),
])
def test_each_learning_shift_is_required_by_an_independent_counterfactual(
        contract, changed_field, new_value, only_stage):
    changed = deepcopy(contract)
    changed[changed_field] = new_value
    changed["receipt_code"] = changed["code_prefix"] + changed["code_suffix"]
    for stage in mission.LEARNING_STAGES:
        original_text = mission.learning_instruction(stage, contract)
        changed_text = mission.learning_instruction(stage, changed)
        assert (original_text != changed_text) is (stage == only_stage)
    # The final prompt/schema cannot resolve the missing fact.
    assert mission.recall_instruction(contract["order_id"]) == mission.recall_instruction(changed["order_id"])
    assert make_receipt(contract).tools == make_receipt(changed).tools


def test_area_mapping_and_distractor_order_are_exposed_only_in_fragment_shift(contract):
    changed = deepcopy(contract)
    changed["alias_to_destination"] = {"冷藏区": "B", "质检区": "C", "回收区": "A"}
    changed["decoy_receipt_code"] = "836251"
    for stage in mission.LEARNING_STAGES:
        differs = mission.learning_instruction(stage, contract) != mission.learning_instruction(stage, changed)
        assert differs is (stage == "fragments")


def test_revision_update_does_not_repeat_fragments_or_mapping_and_archive_is_marked(contract):
    text = mission.learning_instruction("revision", contract)
    assert contract["new_alias"] in text and contract["old_alias"] in text
    assert "ACTIVE DISPATCH 修订号=2" in text
    assert "ARCHIVED DISPATCH 修订号=1" in text
    assert text.index("ACTIVE DISPATCH") < text.index("ARCHIVED DISPATCH")
    for secret in (contract["code_prefix"], contract["code_suffix"], contract["receipt_code"], contract["decoy_receipt_code"]):
        assert secret not in text
    assert "冷藏区→C" not in text and "质检区→A" not in text and "回收区→B" not in text
    assert "出现得更晚不等于版本更高" in text
    assert "本班任务文字中持续可见" in text


def test_initial_order_cannot_be_mistaken_for_current_learning_navigation_goal(contract):
    for stage in mission.LEARNING_STAGES:
        text = mission.learning_instruction(stage, contract)
        assert text.startswith(mission.learning_scenario(stage).instruction)
        assert "本班巡检仍按上面的本班目标完成" in text
        assert "只有你自行保存的长期记忆能够交班" in text
    with pytest.raises(ValueError, match="stage"):
        mission.learning_instruction("missing", contract)


def test_final_prompt_does_not_name_hidden_revision_or_answer(contract):
    text = mission.recall_instruction(contract["order_id"])
    assert contract["order_id"] in text
    assert contract["decoy_order_id"] not in text
    assert "修订号=2" not in text and "R2" not in text
    for secret in (contract["code_prefix"], contract["code_suffix"], contract["receipt_code"], *mission.ALIASES):
        assert secret not in text
    assert "当前有效" in text and "ARCHIVED" in text
    for invalid in ("", "with spaces", "../secret", None):
        with pytest.raises(ValueError):
            mission.recall_instruction(invalid)


def test_matching_submission_succeeds_only_when_physical_route_matches(contract, route_spy):
    extension = make_receipt(contract)
    result = extension.execute(call(arguments(contract)))
    assert result["status"] == "succeeded" and not route_spy
    verdict = extension.evaluate()
    assert verdict["success"] is True and verdict["route_success"] is True
    assert all(verdict[key] for key in ("order_id_match", "revision_match", "destination_match", "code_match",
                                       "prefix_match", "suffix_match", "route_match"))
    assert not any(verdict[key] for key in ("old_revision_selected", "old_destination_selected", "old_route_selected"))
    assert route_spy == [{"trajectory": [{"passed_route": "yellow"}], "route": "yellow", "start_xy": (0, 0)}]


def test_exact_old_contract_has_complete_correct_code_but_fails_revision_target_route(contract, route_spy):
    extension = make_receipt(contract, route=contract["old_route"])
    extension.execute(call(arguments(contract, old=True)))
    result = extension.evaluate()
    assert result["code_match"] and result["prefix_match"] and result["suffix_match"]
    assert result["old_revision_selected"] and result["old_destination_selected"] and result["old_route_selected"]
    assert not result["revision_match"] and not result["destination_match"] and not result["route_match"]
    assert not result["route_success"] and not result["success"]
    assert route_spy[0]["route"] == contract["new_route"], "Score the effective route, never the claimed old route"


def test_correct_text_cannot_replace_actual_route_execution(contract, route_spy):
    extension = make_receipt(contract, route=contract["old_route"])
    extension.execute(call(arguments(contract)))
    result = extension.evaluate()
    assert result["route_match"] and result["code_match"]
    assert not result["route_success"] and not result["success"]


@pytest.mark.parametrize("field,value,flag", [
    ("order_id", "order-fixture-other", "order_id_match"),
    ("dispatch_revision", 1, "revision_match"),
    ("destination", "C", "destination_match"),
    ("route", "blue", "route_match"),
    ("receipt_code", "107042", "prefix_match"),
    ("receipt_code", "007043", "suffix_match"),
])
def test_format_valid_wrong_values_have_identical_online_reply_but_fail_final_scoring(
        contract, route_spy, field, value, flag):
    correct, wrong = make_receipt(contract), make_receipt(contract)
    values = arguments(contract)
    values[field] = value
    assert correct.execute(call(arguments(contract))) == wrong.execute(call(values))
    assert not route_spy, "No private route evaluation may run during online submission"
    result = wrong.evaluate()
    assert result[flag] is False and result["success"] is False
    assert result["route_success"] is True


@pytest.mark.parametrize("field,value", [
    ("dispatch_revision", True), ("dispatch_revision", 0), ("dispatch_revision", 101),
    ("dispatch_revision", 2.0), ("dispatch_revision", "2"),
    ("receipt_code", 7042), ("receipt_code", "７０００４２"), ("receipt_code", "00704"),
    ("receipt_code", "007042\n"), ("order_id", "../secret"),
    ("destination", "a"), ("destination", ["A"]), ("route", "green"), ("route", ["blue"]),
])
def test_invalid_submission_does_not_lock_the_receipt(contract, field, value):
    extension = make_receipt(contract)
    values = arguments(contract)
    values[field] = value
    bad = extension.execute(call(values))
    assert bad["error_code"] == "INVALID_ARGUMENTS"
    assert extension.execute(call(arguments(contract), "correct"))["status"] == "succeeded"


def test_missing_and_extra_fields_do_not_lock_receipt(contract):
    extension = make_receipt(contract)
    for values in (None, {}, {**arguments(contract), "explanation": "extra"}):
        assert extension.execute(call(values))["error_code"] == "INVALID_ARGUMENTS"
    assert extension.execute(call(arguments(contract)))["status"] == "succeeded"


def test_first_format_valid_wrong_answer_is_locked_and_exact_replay_is_idempotent(contract, route_spy):
    extension = make_receipt(contract)
    wrong = arguments(contract, old=True)
    first = extension.execute(call(wrong))
    assert extension.execute(call(wrong, "duplicate")) == {**first, "call_id": "duplicate", "deduplicated": True}
    assert extension.execute(call(arguments(contract), "changed"))["error_code"] == "RECEIPT_ALREADY_SUBMITTED"
    assert extension.evaluate()["submission"] == wrong


def test_contract_submission_and_returned_report_cannot_be_mutated_from_outside(contract, route_spy):
    extension = make_receipt(contract)
    values = arguments(contract)
    extension.execute(call(values))
    contract["new_alias"] = contract["old_alias"]
    values["receipt_code"] = "999999"
    result = extension.evaluate()
    assert result["success"] is True
    result["submission"]["destination"] = "B"
    result["route_evaluation"]["success"] = False
    assert extension.evaluate()["success"] is True


def test_private_snapshot_is_only_read_by_final_evaluation(contract, route_spy):
    calls = []

    def snapshot():
        calls.append("private")
        return SimpleNamespace(trajectory=[{"passed_route": "yellow"}])

    extension = mission.ComplexMissionReceipt(contract, snapshot)
    assert extension.tools[0]["function"]["parameters"]["properties"]["dispatch_revision"] == {
        "type": "integer", "minimum": 1, "maximum": 100}
    extension.execute(call(arguments(contract)))
    assert calls == [] and route_spy == []
    extension.evaluate()
    assert calls == ["private"] and len(route_spy) == 1


def test_tools_and_final_prompt_are_identical_for_hidden_contract_variants(contract):
    changed = deepcopy(contract)
    changed.update(new_alias="回收区", code_prefix="918", code_suffix="365", receipt_code="918365",
                   old_revision=7, new_revision=12)
    expected_tools = make_receipt(contract).tools
    assert expected_tools == make_receipt(changed).tools
    encoded = json.dumps(expected_tools, ensure_ascii=False)
    assert contract["order_id"] not in encoded and contract["receipt_code"] not in encoded
    for alias in mission.ALIASES:
        assert alias not in encoded
    assert mission.recall_instruction(contract["order_id"]) == mission.recall_instruction(changed["order_id"])


def test_no_submission_is_not_success_even_with_correct_route(contract, route_spy):
    result = make_receipt(contract).evaluate()
    assert result["route_success"] is True
    assert result["submitted"] is False and result["success"] is False
    assert result["submission"] is None
    assert not result["old_revision_selected"] and not result["old_destination_selected"]


def test_invalid_private_route_result_fails_closed(contract, monkeypatch):
    monkeypatch.setattr(mission, "evaluate_route", lambda *args, **kwargs: {"success": 1})
    extension = make_receipt(contract)
    extension.execute(call(arguments(contract)))
    with pytest.raises(TypeError, match="native bool"):
        extension.evaluate()


def test_unknown_tool_and_bad_constructor_never_read_private_state(contract):
    def forbidden_snapshot():
        pytest.fail("Private snapshot was read outside final evaluation")

    extension = mission.ComplexMissionReceipt(contract, forbidden_snapshot)
    assert extension.execute(ToolCall("bad", "unknown", {}))["error_code"] == "UNKNOWN_TOOL"
    with pytest.raises(TypeError):
        extension.execute({"name": "submit_mission"})
    with pytest.raises(TypeError):
        mission.ComplexMissionReceipt(contract, None)
