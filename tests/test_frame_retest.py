from copy import deepcopy
from types import SimpleNamespace

import pytest

from go2_demo.types import ToolCall
from scripts.retest_occluded_frame import make_trial_observation, score_action


def frame_inputs():
    old = {"question": "Where was B?", "support": "partial", "evidence": [{"id": "E006"}]}
    template = {
        "image_id": "frame-00066",
        "sim_time_s": 133.1999999999079,
        "target_candidates": [],
        "tracked_points": [{"point_id": "point-003", "valid": True}],
        "return_memory": {"revision": "locked", "mode": "query", "current": old},
    }
    history = [{
        "observation": {"image_id": "frame-00065"},
        "calls": [{"name": "memory_query", "arguments": {"question": "Where was B?"}}],
        "results": [{"tool": "memory_query", "status": "succeeded"}],
    }]
    new = {"question": "Which recent positions were measured?", "support": "partial",
           "evidence": [{"id": "E022"}, {"id": "E021"}]}
    return template, history, new


def test_arms_differ_only_in_current_answer_with_identical_public_history():
    template, history, new = frame_inputs()
    previous = deepcopy(template["return_memory"]["current"])
    control = make_trial_observation(template, history, previous)
    treatment = make_trial_observation(template, history, new)

    assert control["return_memory"]["current"] == previous
    assert treatment["return_memory"]["current"] == new
    assert all(control[key] == template[key] for key in template if key != "return_memory")
    assert control["return_memory"] == template["return_memory"]
    assert set(control) == set(treatment)
    history_key, = set(control) - set(template)
    assert control[history_key] == history
    assert set(control[history_key][0]) == {"observation", "calls", "results"}
    treatment["return_memory"]["current"] = previous
    assert treatment == control


def test_arm_cannot_mutate_common_observation_history_or_other_answer():
    template, history, new = frame_inputs()
    original_template, original_history, original_answer = deepcopy((template, history, new))
    trial = make_trial_observation(template, history, new)
    history_key, = set(trial) - set(template)
    trial["tracked_points"][0]["valid"] = False
    trial["return_memory"]["current"]["evidence"].clear()
    trial[history_key][0]["results"].clear()
    assert (template, history, new) == (original_template, original_history, original_answer)


def test_multiple_robot_calls_fail_before_any_action_executes():
    calls = [ToolCall(str(i), "turn", {"yaw_rad": .3}) for i in range(2)]
    assert score_action(None, calls) == {
        "kind": "protocol", "status": "failed", "error_code": "ONE_TOOL_PER_TURN"}


@pytest.mark.parametrize(("args", "status"), [
    ({"question": "Where was the last measured position?"}, "requested"),
    ({"question": " "}, "failed"), ({"question": "x" * 401}, "failed"),
])
def test_new_question_is_classified_separately_from_robot_execution(args, status):
    result = score_action(None, [ToolCall("id", "memory_query", args)])
    assert (result["kind"], result["status"]) == ("memory", status)


def test_scores_actual_dispatched_result_and_measured_odometry():
    observations = iter([
        {"sim_time_s": 10., "local_odometry": {"translation_m": [1., 2.], "yaw_rad": 3.13}},
        {"sim_time_s": 10.6, "local_odometry": {"translation_m": [1.03, 2.04], "yaw_rad": -3.13}},
    ])
    executed = []

    def dispatch(call):
        executed.append(call)
        return {"status": "failed", "error_code": "COLLISION"}

    harness = SimpleNamespace(observation=observations.__next__, dispatch=dispatch,
                              backend=SimpleNamespace(collision_terminated=True, _fallen_ever=False))
    call = ToolCall("id", "turn", {"yaw_rad": .1})
    result = score_action(harness, [call])
    assert executed == [call]
    assert (result["kind"], result["status"], result["error_code"]) == ("motion", "failed", "COLLISION")
    assert result["translation_m"] == pytest.approx(.05)
    assert result["yaw_change_rad"] == pytest.approx(.0231853)
    assert result["elapsed_sim_s"] == pytest.approx(.6)
    assert result["collision"] and not result["fallen"]


def test_successful_perception_is_not_counted_as_motion():
    observation = {"sim_time_s": 10., "local_odometry": {"translation_m": [0., 0.], "yaw_rad": .2}}
    harness = SimpleNamespace(observation=lambda: observation, dispatch=lambda call: {"status": "succeeded"},
                              backend=SimpleNamespace(collision_terminated=False, _fallen_ever=False))
    result = score_action(harness, [ToolCall("id", "get_geometry", {})])
    assert result["kind"] == "perception" and result["status"] == "succeeded"
    assert result["translation_m"] == result["yaw_change_rad"] == 0.
