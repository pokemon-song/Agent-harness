"""Public scene catalog capabilities are independent of the API provider."""

from go2_demo.scenarios import get_scenario, list_scenarios


def test_public_task_catalog_has_distinct_goals_and_honest_unsupported_cases():
    scenarios = list_scenarios()
    assert len({item.task_id for item in scenarios}) == 17
    assert sum(item.supported for item in scenarios if item.category in {"go_to", "distinguish", "go_avoid", "go_through"}) == 13
    assert {item.category for item in scenarios} == {"baseline", "go_to", "distinguish", "go_avoid", "go_through", "crawl", "composite"}
    assert all(not item.supported and item.unsupported_reason for item in scenarios if item.category == "crawl")
    assert get_scenario("go_to_blue_ball").target.kind == "sphere"
    assert get_scenario("go_to_green_cube").target.color == "green"


def test_composite_combines_fixed_drawers_gate_and_letter_b_without_changing_navigation_catalog():
    scenario = get_scenario("composite_drawers_gate_b")
    assert scenario.category == scenario.goal_kind == "composite"
    assert scenario.source == "local_composite_mujoco"
    assert scenario.supported and scenario.target_jitter_m == 0
    assert scenario.target.letter == "B" and scenario.target.xy == (5.6, 0.0)
    assert scenario.cabinet_exit_x == 2.3
    assert scenario.passage_x == 3.4 and scenario.passage_half_width == .60
    assert scenario.finish_x_min == 4.2 and scenario.passage_clearance_m == .8
    objects = {item.name: item for item in scenario.objects}
    assert objects["drawers_body"].position[0] == 1.65
    assert objects["sign_A"].position[:2] == (5.6, -1.0)
    assert objects["sign_C"].position[:2] == (5.6, 1.0)
    assert {objects[name].position[0] for name in ("gate_post_0", "gate_post_1", "gate_top")} == {3.4}
    assert {abs(objects[name].position[1]) - objects[name].size[1]
            for name in ("gate_post_0", "gate_post_1")} == {.60}
