"""Offline episode replay without named targets, task stages or scene-specific cameras."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .occluded_video import _ffmpeg


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(directory, output=None):
    directory = Path(directory)
    output = Path(output) if output else directory / "full.mp4"
    if output.exists():
        raise FileExistsError(output)
    meta = json.loads((directory / "capture.json").read_text())
    if meta["mujoco_version"] != mujoco.__version__:
        raise ValueError("Replay requires the recorded MuJoCo version")
    model_path, pose_path = directory / "model.mjb", directory / "poses.npz"
    if digest(model_path) != meta["model_sha256"] or digest(pose_path) != meta["poses_sha256"]:
        raise ValueError("Recorded model or poses changed")
    model = mujoco.MjModel.from_binary_path(str(model_path))
    with np.load(pose_path, allow_pickle=False) as data:
        times, poses = data["time"], data["qpos"]
    if (times.shape != (meta["sample_count"],) or poses.shape != (len(times), model.nq) or not len(times)
            or not np.isfinite(times).all() or not np.isfinite(poses).all() or np.any(np.diff(times) <= 0)
            or times[0] != meta["start_s"] or times[-1] != meta["end_s"]):
        raise ValueError("Invalid recording timeline")
    points = [*poses[:, :2]]
    for index in range(model.ngeom):
        if model.geom_bodyid[index] == 0 and model.geom_type[index] != mujoco.mjtGeom.mjGEOM_PLANE and model.geom(index).name != "floor":
            radius = float(model.geom_rbound[index])
            points.extend([model.geom_pos[index, :2] - radius, model.geom_pos[index, :2] + radius])
    box = np.asarray(points)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:2] = (box.min(axis=0) + box.max(axis=0)) / 2
    camera.lookat[2] = .2
    camera.azimuth, camera.elevation = 90, -59
    camera.distance = max(7., 1.2 * np.ptp(box[:, 0]) + 2., 2.6 * np.ptp(box[:, 1]) + 2.)
    model.vis.map.zfar = max(model.vis.map.zfar, (camera.distance + np.linalg.norm(np.ptp(box, axis=0)) + 2.) / model.stat.extent)
    model.vis.quality.offsamples = 0
    fps, width, height = 12.5, 640, 360
    count = max(2, math.ceil((times[-1] - times[0]) * fps - 1e-8) + 1)
    sources = np.searchsorted(times, times[0] + np.arange(count) / fps + 1e-9, side="right") - 1
    sources[0], sources[-1] = 0, len(times) - 1
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_suffix(".partial.mp4")
    if part.exists():
        raise FileExistsError(part)
    binary = _ffmpeg()
    cmd = [binary, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo", "-pixel_format", "rgb24",
           "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(part)]
    data = mujoco.MjData(model)
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for index in sources:
                data.time, data.qpos[:] = times[index], poses[index]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                frame = Image.fromarray(renderer.render())
                drawing = ImageDraw.Draw(frame)
                drawing.rectangle((0, 0, width, 21), fill=(24, 30, 36))
                drawing.text((8, 3), f"GENERIC AGENT | simulation {times[index]:.2f} s", fill="white")
                process.stdin.write(frame.tobytes())
        finally:
            try:
                process.stdin.close()
            finally:
                code = process.wait()
    if code:
        raise RuntimeError(f"Video encoder failed: {code}")
    verification = subprocess.run([binary, "-hide_banner", "-loglevel", "error", "-xerror", "-i", str(part),
        "-map", "0:v:0", "-f", "null", "-", "-progress", "pipe:1", "-nostats"], check=True, capture_output=True, text=True)
    frames = re.findall(r"^frame=(\d+)$", verification.stdout, re.MULTILINE)
    if not frames or int(frames[-1]) != count:
        raise ValueError("Decoded video differs from the recorded timeline")
    part.replace(output)
    result = {"file": output.name, "sha256": digest(output), "frame_count": count, "fps": fps,
              "duration_s": count / fps, "decoded_all_frames": True, "initial_and_final_states_included": True}
    (directory / "general-video.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    print(json.dumps(render(args.directory, args.output), indent=2))
