"""Shared, serializable runtime identity for episodes and frozen experiments."""

import hashlib
import json
from pathlib import Path
import subprocess


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    # Robot plugins and environment modules are nested packages; glob("*.py")
    # would silently leave their implementation outside the frozen experiment.
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / "go2_demo").rglob("*.py"))}


def git_identity(root: Path) -> dict:
    """Archive local revision identity, without exposing remotes or credentials."""
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                  capture_output=True, text=True, check=True, timeout=5).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                capture_output=True, text=True, check=True, timeout=5).stdout
        return {"commit": revision, "dirty": bool(status)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def runtime_identity(profile) -> dict:
    snapshot = profile.to_dict()
    return {
        "robot_id": profile.robot_id,
        "robot_profile": snapshot,
        "robot_profile_sha256": digest(snapshot),
        "execution_contract": "synchronous_robot_backend_v1",
        "simulation_mode": profile.simulation_mode,
        "observation_protocol": "capability_sensor_observation_v3",
        "action_protocol": "capability_driven_robot_tools_v3",
        "completion_contract": "public_body_reference_and_clearance_v2",
    }
