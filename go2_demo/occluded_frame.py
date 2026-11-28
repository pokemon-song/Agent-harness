"""Reproduce the public state of the interrupted sixth planning decision."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .memory_perception import TaskLabelRGBD, task_targets
from .occluded_handoff import _digest, restore_at_a
from .types import ToolCall


FIXTURE = Path(__file__).resolve().parents[1] / "datasets" / "occluded-return-frame6"


def replay_return_prefix(harness, documents, fixture=None):
    """Replay the four physical tools and one read receipt, with no model calls."""
    fixture = Path(fixture) if fixture is not None else FIXTURE
    data = json.loads((fixture / "frame6.json").read_text(encoding="utf-8"))
    memory = deepcopy(data["memory"])
    if memory["revision"] != _digest(documents):
        raise ValueError("Outbound documents differ from the original memory revision")
    original_card = memory["current"]
    memory["current"] = None
    harness.task = data["task"]
    detector = TaskLabelRGBD(task_targets(harness.backend.scenario.task_id))
    history = []
    turns = data["turns"]
    if len(turns) != 6:
        raise ValueError("Expected five tool decisions followed by the sixth observation")
    for i, turn in enumerate(turns):
        frame = harness.frame
        if frame.frame_id != turn["frame"]:
            raise ValueError(f"Frame identity differs at decision {i + 1}")
        with Image.open(fixture / (frame.frame_id + ".png")) as image:
            if not np.array_equal(frame.rgb, np.asarray(image.convert("RGB"))):
                raise ValueError(f"RGB differs at decision {i + 1}")
        obs = harness.observation()
        view, _ = detector.inspect(obs, frame.rgb, frame.depth_m, frame.intrinsics)
        obs["target_candidates"] = view["target_candidates"]
        obs["return_memory"] = deepcopy(memory)
        if _digest(obs) != turn["observation_sha256"]:
            raise ValueError(f"Public observation differs at decision {i + 1}")
        if i == len(turns) - 1:
            if "call" in turn:
                raise ValueError("Sixth decision must be unexecuted")
            break
        call = ToolCall(**turn["call"])
        if i == 0:
            if call.name != "memory_query" or call.arguments != {"question": original_card["question"]}:
                raise ValueError("Unexpected historical memory request")
            memory["current"] = original_card
            result = {"call_id": call.call_id, "tool": call.name, "status": "succeeded", "cache_hit": False}
        else:
            result = harness.dispatch(call)
        if _digest([result]) != turn["results_sha256"]:
            raise ValueError(f"Tool result differs at decision {i + 1}")
        history.append({"observation": {k: v for k, v in obs.items() if k != "return_memory"},
                        "calls": [turn["call"]], "results": [result]})
    history_hash = _digest(history)
    if history_hash != data["history_sha256"]:
        raise ValueError("Public planning history differs from the archived prefix")
    audit = {"source_run": data["source_run"], "historical_planner_decision": 6,
             "checked_public_observations": len(turns), "checked_prior_tool_results": len(history),
             "frame_id": frame.frame_id, "sim_time_s": frame.sim_time_s,
             "rgb_sha256": hashlib.sha256(frame.rgb.tobytes()).hexdigest(),
             "public_history_sha256": history_hash, "public_observation_sha256": _digest(obs),
             "sensor_observation_sha256": _digest({k: v for k, v in obs.items() if k != "return_memory"}),
             "qpos_sha256": hashlib.sha256(harness.backend.data.qpos.tobytes()).hexdigest(),
             "qvel_sha256": hashlib.sha256(harness.backend.data.qvel.tobytes()).hexdigest(),
             "documents_sha256": memory["revision"],
             "history_format": "Five public observations, calls and tool receipts. Provider reasoning and assistant text were not archived."}
    return obs, history, audit


def restore_frame_six(backend, archive, fixture=None):
    """Start with a fresh backend; return the live harness at turn 6, ready for either arm."""
    harness, documents, handoff = restore_at_a(backend, archive)
    observation, history, audit = replay_return_prefix(harness, documents, fixture)
    audit["outbound_trajectory_sha256"] = handoff["prefix_trajectory_sha256"]
    return harness, documents, observation, history, audit
