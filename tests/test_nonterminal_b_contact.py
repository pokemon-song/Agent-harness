from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import pytest

from go2_demo.backend import MujocoGo2Backend
from go2_demo.fresh_episode_run import episode_scene


@pytest.fixture
def make_backend(monkeypatch):
    assets = Path(__file__).resolve().parents[1] / "assets"
    if not (assets / "go2_rl_lab/resources/robots/go2/go2.xml").is_file():
        pytest.skip("Go2 assets unavailable")
    monkeypatch.setattr(mujoco, "Renderer", lambda *a, **kw: SimpleNamespace(close=lambda: None))
    backends = []

    def make(scene):
        backend = MujocoGo2Backend(assets, scenario=scene, stop_on_first_collision=True,
                                   continue_on_b_contact=True)
        backends.append(backend)
        return backend

    yield make
    for backend in backends:
        backend.close()


def place_base(backend, x, y):
    # Place the robot for individual contact and scoring cases while preserving episode history.
    backend.data.qpos[:2] = (x, y)
    mujoco.mj_forward(backend.model, backend.data)


def test_b_contact_allows_retreat_and_later_ordered_stops(make_backend):
    backend = make_backend(episode_scene((24, 18)))
    place_base(backend, 1.17, .6)
    approach = backend.move_relative(.37, 0., timeout_s=2.)
    assert approach["collision"] and approach["error_code"] != "COLLISION"
    assert [event["object"] for event in backend.collision_events] == ["B"]
    impact_time = backend.collision_events[0]["sim_time_s"]
    assert backend.data.time > impact_time and not backend.collision_terminated
    initial = backend.evaluate()
    assert initial["collision"] and not initial["success"]
    assert not initial["checkpoints"]["B"]["eligible"]

    before_retreat = backend.data.time
    retreat = backend.move_relative(-.3, 0.)
    assert retreat["status"] == "succeeded" and backend.data.time > before_retreat
    assert retreat["measured"]["translation_start_frame_m"][0] < -.1

    place_base(backend, 5.4, 0.)
    assert backend.stop()["status"] == "succeeded"
    assert backend.evaluate()["checkpoints"]["A"]["passed"]
    # A B touch on the return leg still must satisfy B's distance and stop rules.
    place_base(backend, 1.4, .6)
    backend.stop()
    at_b = backend.evaluate()
    assert at_b["checkpoints"]["B"]["eligible"] and not at_b["success"]
    assert "DISTANCE_OUT_OF_RANGE" in at_b["checkpoints"]["B"]["failure_reasons"]
    assert not backend.collision_terminated

    place_base(backend, .8, .6)
    assert backend.stop()["status"] == "succeeded"
    result = backend.evaluate()
    assert result["collision"] and result["success"]
    assert result["checkpoints"]["B"]["passed"]

    # A later cabinet impact must still freeze the experiment, even after accepted B contacts.
    place_base(backend, 3.23, 0.)
    assert backend.move_relative(.2, 0., timeout_s=1.)["error_code"] == "COLLISION"
    assert backend.collision_events[-1]["object"] == "cabinet"
    assert backend.collision_terminated and not backend.evaluate()["success"]
    final_time = backend.data.time
    assert backend.stop()["error_code"] == "COLLISION" and backend.data.time == final_time


def test_simultaneous_b_and_cabinet_contacts_still_end_the_episode(make_backend):
    scene = episode_scene((24, 18))
    cube = next(obj for obj in scene.objects if obj.name == "memory_goal_b")
    # Align the front faces in this fixture to obtain two contacts on the same physics step.
    x, y, _ = cube.position
    objects = tuple(
        replace(obj, position=(x - cube.size[0] + obj.size[0], y, obj.position[2]))
        if obj.name == "memory_drawers_body" else obj for obj in scene.objects
    )
    backend = make_backend(replace(scene, objects=objects))
    place_base(backend, 1.42, .6)
    assert backend.turn(.2)["error_code"] == "COLLISION"
    assert backend.collision_terminated
    assert {event["object"] for event in backend.collision_events} == {"B", "cabinet"}
    assert all(event["sim_time_s"] == backend.data.time for event in backend.collision_events)
