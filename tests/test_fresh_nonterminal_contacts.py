"""Real physics for score-only contacts in the fresh A/B task; no model API."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import math

import mujoco
import pytest

from go2_demo.backend import MujocoGo2Backend
from go2_demo.fresh_episode_run import _collision_score, episode_scene
from test_nonterminal_b_contact import place_base


@pytest.fixture
def score_only_backend(monkeypatch):
    assets = Path(__file__).resolve().parents[1] / "assets"
    if not (assets / "go2_rl_lab/resources/robots/go2/go2.xml").is_file():
        pytest.skip("Go2 assets unavailable")
    # These checks run the real policy, contacts and independent evaluator but
    # do not need camera rendering or a graphics context.
    monkeypatch.setattr(mujoco, "Renderer", lambda *a, **kw: SimpleNamespace(close=lambda: None))
    backend = MujocoGo2Backend(assets, scenario=episode_scene((24, 18)), stop_on_first_collision=False)
    try:
        yield backend
    finally:
        backend.close()


@pytest.mark.parametrize("obstacle,start,advance,timeout", [
    ("cabinet", (3.23, 0.), .2, 1.),
    ("B", (1.17, .6), .37, 2.),
])
def test_real_contact_allows_retreat_and_successful_ordered_stops(
        score_only_backend, obstacle, start, advance, timeout):
    backend = score_only_backend
    place_base(backend, *start)
    impact = backend.move_relative(advance, 0., timeout_s=timeout)
    assert impact["collision"] and impact["error_code"] != "COLLISION"
    assert backend.collision_events and {event["object"] for event in backend.collision_events} == {obstacle}
    assert backend.data.time > backend.collision_events[0]["sim_time_s"]
    assert not backend.collision_terminated
    assert backend.evaluate()["collision"] and not backend.evaluate()["success"]
    if obstacle == "cabinet":
        assert impact["cabinet_contact"] and backend.cabinet_contact_events

    before = backend.data.time
    retreat = backend.move_relative(-.3, 0.)
    assert retreat["status"] == "succeeded"
    assert backend.data.time > before
    assert retreat["measured"]["translation_start_frame_m"][0] < -.1

    # Isolate scoring from navigation by placing the robot at each checkpoint;
    # stop/settling and the evaluator still run on measured MuJoCo state.
    place_base(backend, 5.4, 0.)
    assert backend.stop()["status"] == "succeeded"
    assert backend.evaluate()["checkpoints"]["A"]["passed"]
    place_base(backend, .8, .6)
    assert backend.stop()["status"] == "succeeded"
    result = backend.evaluate()
    assert result["success"] and result["collision"]
    assert result["checkpoints"]["A"]["passed"] and result["checkpoints"]["B"]["passed"]
    assert result["collision_count"] == len(backend.collision_events) > 0
    assert result["collision_events"] == backend.collision_events
    assert backend.public_task_spec["collision_free_required"] is False
    assert backend.public_task_spec["collision_penalty_points"] == 5
    assert backend.public_task_spec["b_contact_ends_episode"] is False
    assert backend.public_task_spec["other_scene_contacts_end_episode"] is False
    score = _collision_score(result, result["success"], backend.scenario.collision_penalty_points)
    assert score["goal_points"] == 100
    assert score["collision_penalty"] == 5 * result["collision_count"]
    assert score["score"] == max(0, 100 - score["collision_penalty"]) < 100

    # Exercise the zero-score boundary without introducing twenty costly impacts.
    many_contacts = {**deepcopy(result), "collision_count": 20}
    unchanged = deepcopy(many_contacts)
    zero = _collision_score(many_contacts, many_contacts["success"], 5)
    assert zero["score"] == 0 and zero["collision_penalty"] == 100
    assert many_contacts == unchanged and many_contacts["success"] is True


def test_score_only_contacts_do_not_allow_a_fall_to_pass(score_only_backend):
    backend = score_only_backend
    place_base(backend, 5.4, 0.)
    assert backend.stop()["status"] == "succeeded"
    place_base(backend, .8, .6)
    assert backend.stop()["status"] == "succeeded"
    assert backend.evaluate()["success"]

    # A real sideways pose is a fall; start clear of the ground to avoid deeply
    # penetrating geometry. No collision or fall flag is fabricated.
    backend.data.qpos[2] = .8
    backend.data.qpos[3:7] = (math.sqrt(.5), math.sqrt(.5), 0., 0.)
    backend.data.qvel[:] = 0.
    mujoco.mj_forward(backend.model, backend.data)
    assert backend._fallen()
    action = backend.turn(.1)
    assert action["status"] == "failed" and action["error_code"] == "ROBOT_FELL"
    result = backend.evaluate()
    assert result["fallen"] and not result["success"]
    assert not result["checkpoints"]["A"]["passed"] and not result["checkpoints"]["B"]["passed"]
