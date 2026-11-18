"""Record true robot poses and render them offline on the original static A/B scene."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class OccludedRecording:
    """Sample once per executed control interval, including a shortened collision interval."""

    def __init__(self, backend, directory):
        if backend._recording is not None:
            raise RuntimeError("The backend already has a recording")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.backend = backend
        self.times = []
        self.poses = []
        self.handoff_s = None
        self.closed = False
        mujoco.mj_saveModel(backend.model, str(self.directory / "model.mjb"), None)
        self.sample()
        backend._recording = self

    def sample(self):
        now = float(self.backend.data.time)
        if not self.times or now != self.times[-1]:
            self.times.append(now)
            self.poses.append(self.backend.data.qpos.copy())

    def mark_handoff(self):
        self.sample()
        self.handoff_s = self.times[-1]

    def finish(self):
        if self.closed:
            return
        self.sample()
        path = self.directory / "poses.npz"
        temp = self.directory / "poses.tmp.npz"
        np.savez_compressed(temp, time=self.times, qpos=self.poses)
        temp.replace(path)
        metadata = {
            "sample_count": len(self.times), "start_s": self.times[0], "end_s": self.times[-1],
            "handoff_s": self.handoff_s, "model_sha256": _sha(self.directory / "model.mjb"),
            "poses_sha256": _sha(path), "mujoco_version": mujoco.__version__,
            "timebase": "simulation_seconds", "physics_sampling": "initial and every executed control step",
        }
        tmp = self.directory / "capture.tmp.json"
        tmp.write_text(json.dumps(metadata, indent=2) + "\n")
        tmp.replace(self.directory / "capture.json")
        self.backend._recording = None
        self.closed = True


def _ffmpeg():
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def render(directory, output=None):
    """Offline simulation-time MP4 from recorded qpos; no physics integration or API calls."""
    directory = Path(directory)
    output = Path(output) if output is not None else directory / "full.mp4"
    if output.exists():
        raise FileExistsError(output)
    meta = json.loads((directory / "capture.json").read_text())
    if meta["mujoco_version"] != mujoco.__version__:
        raise ValueError("MuJoCo version differs from the recording")
    model_file, poses_file = directory / "model.mjb", directory / "poses.npz"
    if _sha(model_file) != meta["model_sha256"] or _sha(poses_file) != meta["poses_sha256"]:
        raise ValueError("Recording digest mismatch")
    model = mujoco.MjModel.from_binary_path(str(model_file))
    with np.load(poses_file, allow_pickle=False) as archive:
        times, poses = archive["time"], archive["qpos"]
    if (times.shape != (meta["sample_count"],) or poses.shape != (len(times), model.nq)
            or len(times) < 1 or not np.isfinite(times).all() or not np.isfinite(poses).all()
            or np.any(np.diff(times) <= 0) or times[0] != meta["start_s"] or times[-1] != meta["end_s"]):
        raise ValueError("Invalid recording timeline")
    fps, width, height = 12.5, 640, 360
    count = max(2, math.ceil((times[-1] - times[0]) * fps - 1e-8) + 1)
    ticks = times[0] + np.arange(count) / fps
    sources = np.searchsorted(times, ticks + 1e-9, side="right") - 1
    sources[0], sources[-1] = 0, len(times) - 1
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Start, terminal, and scene props all stay in this continuous fixed-camera view.
    box = np.vstack([poses[:, :2], [0., 0.], model.geom("scenario_target").pos[:2],
                     model.geom("memory_goal_b").pos[:2]])
    camera.lookat[:2] = (box.min(axis=0) + box.max(axis=0)) / 2
    camera.lookat[2] = .18
    camera.azimuth, camera.elevation = 90, -59
    camera.distance = max(8., 1.2 * np.ptp(box[:, 0]) + 1.8, 2.6 * np.ptp(box[:, 1]) + 1.8)
    data = mujoco.MjData(model)
    binary = _ffmpeg()
    part = output.with_suffix(".partial.mp4")
    cmd = [binary, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
           "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v",
           "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(part)]
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for idx in sources:
                data.time = times[idx]
                data.qpos[:] = poses[idx]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                img = Image.fromarray(renderer.render())
                drawing = ImageDraw.Draw(img)
                returning = meta["handoff_s"] is not None and times[idx] >= meta["handoff_s"]
                phase = "RETURN TO B  |  LIVE AGENT" if returning else "OUTBOUND TO A  |  SHARED RECORDED ACTIONS"
                drawing.rectangle((0, 0, width, 21), fill=(24, 30, 36))
                drawing.text((9, 3), f"{phase}   |   simulation {times[idx]:.2f} s", fill="white")
                process.stdin.write(img.tobytes())
        finally:
            try:
                process.stdin.close()
            finally:
                status = process.wait()
    if status != 0:
        raise RuntimeError(f"ffmpeg encoder failed with exit {status}")
    verification = subprocess.run([binary, "-hide_banner", "-loglevel", "error", "-xerror", "-i", str(part),
                                   "-map", "0:v:0", "-f", "null", "-", "-progress", "pipe:1", "-nostats"],
                                  check=True, capture_output=True, text=True)
    frames = re.findall(r"^frame=(\d+)$", verification.stdout, re.MULTILINE)
    if not frames or int(frames[-1]) != count:
        raise ValueError("Decoded video frame count differs from source timeline")
    part.replace(output)
    result = {"file": output.name, "sha256": _sha(output), "frame_count": count,
              "fps": fps, "duration_s": count / fps, "decoded_all_frames": True,
              "initial_and_final_states_included": True, "source": meta}
    (directory / "video.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(render(parser.parse_args().directory), indent=2))
