"""Private simulation state archive for offline third-person video.

Sampling only copies state; rendering and encoding happen after the experiment.
Nothing in this archive is a robot observation or a model tool result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np


class SimulationRecording:
    def __init__(self, model, data, directory, *, control_hz=50):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self.state_size = mujoco.mj_stateSize(model, self.state_spec)
        self.control_hz = control_hz
        self.started = time.monotonic()
        self.states, self.sim_times, self.wall_times = [], [], []
        self.closed = False
        # A binary MuJoCo model includes the exact scene, meshes and materials.
        mujoco.mj_saveModel(model, str(self.directory / "model.mjb"), None)
        self.sample(model, data)

    def sample(self, model, data):
        if self.closed:
            return
        state = np.empty(self.state_size, dtype=np.float64)
        mujoco.mj_getState(model, data, state, self.state_spec)
        self.states.append(state)
        self.sim_times.append(float(data.time))
        self.wall_times.append(time.monotonic() - self.started)

    def finish(self):
        """Idempotent; retain samples if writing fails so cleanup can retry."""
        if self.closed:
            return
        archive = self.directory / "states.npz"
        temporary = self.directory / "states.tmp.npz"
        np.savez_compressed(temporary, state=np.asarray(self.states),
                            sim_time_s=np.asarray(self.sim_times),
                            wall_time_s=np.asarray(self.wall_times))
        temporary.replace(archive)
        model_path = self.directory / "model.mjb"
        metadata = {
            "schema_version": 1, "mujoco_version": mujoco.__version__,
            "state_spec": self.state_spec, "state_size": self.state_size,
            "model_file": model_path.name, "states_file": archive.name,
            "sample_count": len(self.states), "control_hz": self.control_hz,
            "sim_start_s": self.sim_times[0], "sim_end_s": self.sim_times[-1],
            "wall_start_s": self.wall_times[0], "wall_duration_s": self.wall_times[-1],
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "states_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "capture": "initial, before/after each control step, sensor capture, final close",
            "includes_initial_settling": False,
            "final_stop_policy": "sample all executed stop steps; verify coverage against episode metrics",
            "privacy": "private offline archive; never supplied to the model",
        }
        temp_metadata = self.directory / "metadata.json.tmp"
        temp_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        temp_metadata.replace(self.directory / "metadata.json")
        self.closed = True

