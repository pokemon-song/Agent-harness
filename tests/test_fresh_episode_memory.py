from types import SimpleNamespace

from go2_demo.fresh_episode_memory import FreshEpisodeMemory


def ask(slot, budget=1):
    return slot.handle("本轮走过哪些机身路点、见到哪些目标？", timeout_s=5., remaining_requests=budget)


def harness(x, frame):
    obs = {"image_id": frame, "sim_time_s": float(x + 1), "image_size": {"width": 1, "height": 1},
           "local_odometry": {"translation_m": [x, 0.], "yaw_rad": 0.}, "tracked_points": []}
    return SimpleNamespace(observation=lambda: obs,
                           frame=SimpleNamespace(rgb=None, depth_m=None, intrinsics=None))


def test_ledger_grows_only_from_own_observations_then_freezes_and_reverses():
    ledger = FreshEpisodeMemory("ledger", "memory_ab_occluded_v1")
    other = FreshEpisodeMemory("ledger", "memory_ab_occluded_v1")
    assert ledger.documents() == other.documents() == {} and ledger.card is None
    ledger.observe(harness(0., "frame-00001"), 0)
    first = ask(ledger, 0)
    assert first["requests"] == 0 and first["card"]["target_evidence"] is None
    ledger.observe(harness(.9, "frame-00002"), 1)
    second = ask(ledger, 0)
    assert first["card"]["waypoints"][0]["id"] == second["card"]["waypoints"][0]["id"]
    assert [p["x"] for p in second["card"]["waypoints"]] == [0., .9]
    ledger.freeze()
    assert ledger.card is None
    ledger.observe(harness(3., "frame-00003"), 2)
    assert [p["x"] for p in ask(ledger, 0)["card"]["waypoints"]] == [.9, 0.]
    assert other.documents() == {}
    empty = FreshEpisodeMemory("empty", "memory_ab_occluded_v1")
    empty.observe(None, 0)
    assert ask(empty, 0)["status"] == "unavailable" and empty.documents() == {}


def test_reader_accepts_partial_evidence_and_charges_failed_validation():
    class Reader:
        def decide(self, payload, image, tools, history, *, timeout_s):
            assert payload["phase"] == "find_A" and not history and image is None
            doc = payload["documents"]["route/leg-01.md"]
            eid = doc.split("[")[1].split("]")[0]
            self.raw_usage = {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
            choice = SimpleNamespace(name="select_episode_evidence",
                                     arguments={"evidence_ids": ["unseen"] if self.unknown else [eid]})
            return SimpleNamespace(calls=[choice])
    reader = Reader()
    reader.unknown = False
    slot = FreshEpisodeMemory("query", "memory_ab_occluded_v1", reader=reader)
    slot.observe(harness(1., "frame-00001"), 0)
    assert ask(slot, 0)["requests"] == 0
    valid = ask(slot)
    assert valid["card"]["waypoints"][0]["x"] == 1. and valid["card"]["target_evidence"] is None
    reader.unknown = True
    rejected = ask(slot)
    assert rejected["status"] == "error" and rejected["requests"] == 1
    assert rejected["known_usage"]["total_tokens"] == 6 and rejected["card"] == valid["card"]


def test_candidate_alone_is_marked_as_candidate_and_live_ids_remain_stable():
    slot = FreshEpisodeMemory("ledger", "memory_ab_occluded_v1")
    slot.observe(harness(0., "frame-00001"), 0)
    old_point = ask(slot)["card"]["waypoints"][0]
    candidate = {"task_label": "B", "description": "绿方块", "pixel_uv": [0, 0], "camera_xyz_m": [0., 0., 1.]}
    slot._notes.sensor = SimpleNamespace(inspect=lambda *args: ({"target_candidates": [candidate]}, {}))
    slot.observe(harness(.9, "frame-00002"), 1)
    card = ask(slot)["card"]
    assert card["waypoints"][0] == old_point and "颜色匹配候选" in card["sightings"][0]["excerpt"]
    slot.freeze()
    result = ask(slot)["card"]
    assert result["target_evidence"]["path"] == "targets/target-B.md"
