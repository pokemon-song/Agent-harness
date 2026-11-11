"""Lossless ledger, smaller model context: field projection, never semantic prose."""

from copy import deepcopy
import json


PROJECTION_VERSION = "memory-feedback-fields-v1"


def encoded_bytes(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def project_event(event):
    """Keep announcements verbatim; repeated robot task/config stays in the ledger.

    Results, requested actions and measured post-action state are preserved. The
    model can explicitly inspect the original event, including before/after
    observations. This projection makes no claims about task success or meaning.
    """
    value = deepcopy(event)
    payload = value.get("payload", {})
    if event.get("kind") != "robot_feedback" or event.get("critical"):
        return value
    # Only project the known navigation feedback envelope. Unknown formats retain
    # all their fields so an external producer's evidence is not silently lost.
    expected = {"episode_id", "scene_id", "coordinate_reference", "call", "result",
                "observation", "next_observation"}
    if (not isinstance(payload, dict) or set(payload) - expected
            or not isinstance(payload.get("next_observation"), dict)
            or not isinstance(payload.get("observation"), dict)):
        return value
    after = payload["next_observation"]
    value["payload"] = {key: payload[key] for key in expected - {"observation", "next_observation"}
                        if key in payload}
    repeated_config = {"task", "completion_requirements", "capabilities", "coordinate_convention",
                       "posture_limits", "sensor_calibration", "image_size", "simulation_mode"}
    omitted = {key for key in repeated_config if key in after and payload["observation"].get(key) == after[key]}
    # Keep tracked geometry and new observation fields; only known repeated
    # configuration is omitted. A changed task/calibration is preserved too.
    value["payload"]["next_observation"] = {key: item for key, item in after.items() if key not in omitted}
    value["context_projection"] = {"version": PROJECTION_VERSION,
        "omitted_fields": ["payload.observation", *[
            "payload.next_observation." + key for key in after
            if key not in value["payload"]["next_observation"]]],
        "full_event_tool": "inspect_event", "event_id": event["id"]}
    return value


def document_index(files):
    return [{key: item[key] for key in ("path", "revision", "sha256", "status",
             "observed_at", "evidence_max_seq") if key in item} for item in files]


def document_context(document):
    """The body plus structured metadata suffices; content repeats both verbatim."""
    return deepcopy({key: value for key, value in document.items() if key != "content"})
