"""Render archived MuJoCo states, without stepping physics or reading task answers.

Each phase has a wall-clock movie (including inference waits) and a simulation
movie (25 fps). The combined movies explicitly mark process/scene resets. This
offline program never creates a robot controller or calls a model API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PHASES = ("briefing", "with_memory", "empty_memory")
LABELS = {"briefing": "SHIFT 1: BRIEFING + PATROL",
          "with_memory": "SHIFT 2: WITH SAVED MEMORY",
          "empty_memory": "CONTROL: EMPTY MEMORY"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def frame_schedule(times, fps=25):
    """Zero-order hold at CFR; keep both endpoints, including duplicate times.

    The first frame is the initial sample. At every later output time, use the
    last source sample no later than that time. The final frame is always the
    final source sample; the encoded interval can exceed the original interval
    by at most two frame periods (one rounded sample and its display period).
    """
    times = np.asarray(times, dtype=np.float64)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or np.any(np.diff(times) < 0) or not isinstance(fps, int) or fps <= 0):
        raise ValueError("Timestamps must be finite, nonempty and monotonic; fps must be positive")
    elapsed = times - times[0]
    count = max(2, math.ceil(float(elapsed[-1]) * fps - 1e-9) + 1)
    output_times = np.minimum(np.arange(count, dtype=np.float64) / fps, elapsed[-1])
    indices = np.searchsorted(elapsed, output_times + 1e-10, side="right") - 1
    indices[0], indices[-1] = 0, len(times) - 1
    return indices, output_times


def load_recording(directory):
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    # These fixed basenames also prevent an archive from redirecting the reader.
    for key, filename in (("model_file", "model.mjb"), ("states_file", "states.npz")):
        if metadata.get(key, filename) != filename:
            raise ValueError("Unexpected recording filename")
    hashes = {name: sha256(directory / name) for name in ("model.mjb", "states.npz", "metadata.json")}
    for key, name in (("model_sha256", "model.mjb"), ("states_sha256", "states.npz")):
        if metadata.get(key) != hashes[name]:
            raise ValueError(f"Recording hash mismatch: {name}")
    with np.load(directory / "states.npz", allow_pickle=False) as archive:
        state = np.asarray(archive["state"], dtype=np.float64)
        sim = np.asarray(archive["sim_time_s"], dtype=np.float64)
        wall = np.asarray(archive["wall_time_s"], dtype=np.float64)
    if (state.ndim != 2 or not len(state) or not np.isfinite(state).all()
            or sim.shape != (len(state),) or wall.shape != (len(state),)
            or metadata["sample_count"] != len(state)):
        raise ValueError("Invalid recording dimensions or sample count")
    frame_schedule(sim)
    frame_schedule(wall)
    if wall[0] < 0 or sim[0] < 0:
        raise ValueError("Recording timestamps cannot be negative")
    return metadata, state, sim, wall, hashes


def ffmpeg_path():
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("Install video dependencies: pip install '.[video]'") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


class Encoder:
    def __init__(self, executable, path, width, height, fps):
        self.path = Path(path)
        self.temporary = self.path.with_name(self.path.stem + ".partial.mp4")
        self.errors = tempfile.TemporaryFile()
        self.process = subprocess.Popen([
            executable, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", f"{width}x{height}", "-framerate", str(fps),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.temporary)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.errors)
        self.frames = 0

    def write(self, rgb):
        self.process.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self, *, failed=False):
        try:
            self.process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        code = self.process.wait()
        self.errors.seek(0)
        error = self.errors.read().decode("utf-8", errors="replace")
        self.errors.close()
        if code or failed:
            self.temporary.unlink(missing_ok=True)
            if code and not failed:
                raise RuntimeError(f"Video encoder failed: {error[-1500:]}")
        else:
            self.temporary.replace(self.path)


def verify_video(executable, path, expected_frames, fps):
    """Decode every frame; fail on corrupt, truncated or incomplete output."""
    result = subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-xerror",
                             "-i", str(path), "-map", "0:v:0", "-an", "-f", "null", "-",
                             "-progress", "pipe:1", "-nostats"],
                            capture_output=True, text=True, check=False)
    frames = re.findall(r"^frame=(\d+)$", result.stdout, flags=re.MULTILINE)
    times = re.findall(r"^out_time_us=(\d+)$", result.stdout, flags=re.MULTILINE)
    decoded = int(frames[-1]) if frames else 0
    if result.returncode or decoded != expected_frames:
        raise RuntimeError(f"Video validation failed: decoded {decoded}/{expected_frames}; {result.stderr[-1000:]}")
    return {"decoded_all_frames": True, "frame_count": decoded, "fps": fps,
            "duration_s": decoded / fps,
            "decoded_duration_s": int(times[-1]) / 1e6 if times else None,
            "file": Path(path).name, "sha256": sha256(path), "bytes": Path(path).stat().st_size}


def font(size):
    for name in ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def decorate(rgb, phase, timebase, sim_time, width, height):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    size = max(12, int(width / 58))
    face = font(size)
    draw.rectangle((0, 0, width, size * 3.4), fill=(16, 23, 32))
    draw.text((14, 8), LABELS.get(phase, phase), font=face, fill=(255, 255, 255))
    label = "WALL TIME - includes all inference waits" if timebase == "wall" else "SIMULATION TIME - inference pauses removed"
    draw.text((14, 12 + size), f"{label} | simulation t = {sim_time:.2f} s", font=face, fill=(184, 210, 230))
    draw.text((14, height - size - 12), "Third-person reconstruction from recorded states; no physics rerun",
              font=face, fill=(245, 245, 245), stroke_width=1, stroke_fill=(10, 10, 10))
    return np.asarray(image)


def fixed_camera(mujoco, phase):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [2.65, 0., .35] if phase == "briefing" else [1.5, 0., .30]
    camera.distance = 8.5 if phase == "briefing" else 6.0
    camera.azimuth = 45
    camera.elevation = -38
    return camera


def render_phase(directory, phase, executable, *, width=1280, height=720, fps=25, overwrite=False,
                 timebases=("wall", "sim")):
    import mujoco
    directory = Path(directory)
    metadata, state, sim, wall, hashes = load_recording(directory)
    if metadata["mujoco_version"] != mujoco.__version__:
        raise ValueError("Archived MJB must be rendered with its original MuJoCo version")
    outputs = {kind: directory / f"third-person-{kind}.mp4" for kind in timebases}
    if not overwrite and any(path.exists() for path in (*outputs.values(), directory / "video-manifest.json")):
        raise FileExistsError("Video outputs already exist; use --overwrite explicitly")
    model = mujoco.MjModel.from_binary_path(str(directory / "model.mjb"))
    state_spec = int(metadata["state_spec"])
    if state.shape[1] != mujoco.mj_stateSize(model, state_spec):
        raise ValueError("Recorded state width disagrees with model/state specification")
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    data = mujoco.MjData(model)
    camera = fixed_camera(mujoco, phase)
    renderer = mujoco.Renderer(model, height=height, width=width)
    report = {"phase": phase, "input_sha256": hashes, "sample_count": len(state),
              "resolution": [width, height], "codec": "h264", "pixel_format": "yuv420p",
              "state_reconstruction": "mj_setState + mj_forward; never mj_step",
              "camera": {"lookat": camera.lookat.tolist(), "distance": camera.distance,
                         "azimuth": camera.azimuth, "elevation": camera.elevation},
              "videos": {}, "note": "CFR endpoint holds add at most two frames; model-inference waits remain in wall movie."}
    try:
        for timebase, timeline in (("wall", wall), ("sim", sim)):
            if timebase not in timebases:
                continue
            indices, output_times = frame_schedule(timeline, fps)
            encoder = Encoder(executable, outputs[timebase], width, height, fps)
            failed = True
            try:
                last_index, rgb = None, None
                for index in indices:
                    if index != last_index:
                        mujoco.mj_setState(model, data, state[index], state_spec)
                        mujoco.mj_forward(model, data)
                        renderer.update_scene(data, camera=camera)
                        rgb = decorate(renderer.render(), phase, timebase, float(sim[index] - sim[0]), width, height)
                        last_index = index
                    encoder.write(rgb)
                failed = False
            finally:
                encoder.close(failed=failed)
            verified = verify_video(executable, outputs[timebase], len(indices), fps)
            verified.update(source_duration_s=float(timeline[-1] - timeline[0]),
                            first_source_index=int(indices[0]), last_source_index=int(indices[-1]),
                            first_source_sim_time_s=float(sim[indices[0]]),
                            last_source_sim_time_s=float(sim[indices[-1]]),
                            first_source_wall_time_s=float(wall[indices[0]]),
                            last_source_wall_time_s=float(wall[indices[-1]]),
                            first_and_final_states_included=bool(indices[0] == 0 and indices[-1] == len(state) - 1),
                            unique_rendered_samples=len(np.unique(indices)))
            report["videos"][timebase] = verified
            print(f"Rendered {phase}/{outputs[timebase].name}: {len(indices)} frames, {len(indices) / fps:.2f}s", flush=True)
    finally:
        renderer.close()
    write_json(directory / "video-manifest.json", report)
    return report


def reset_card(executable, path, phase, width, height, fps):
    image = Image.new("RGB", (width, height), (16, 23, 32))
    draw = ImageDraw.Draw(image)
    for y, text, size in ((height * .30, "RESET", width // 24),
                          (height * .46, "Fresh process, robot scene, and short-term history", width // 48),
                          (height * .58, LABELS[phase], width // 40)):
        draw.text((width / 2, y), text, anchor="mm", font=font(size), fill="white")
    encoder = Encoder(executable, path, width, height, fps)
    failed = True
    try:
        for _ in range(2 * fps):
            encoder.write(np.asarray(image))
        failed = False
    finally:
        encoder.close(failed=failed)


def concat_movies(executable, paths, output, expected_frames, fps):
    output = Path(output)
    temporary = output.with_name(output.stem + ".partial.mp4")
    with tempfile.TemporaryDirectory(prefix="go2-video-concat-") as work:
        listing = Path(work) / "clips.txt"
        listing.write_text("".join("file '" + str(Path(path).resolve()).replace("'", "'\\''") + "'\n" for path in paths), encoding="utf-8")
        result = subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
                                 "-safe", "0", "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(temporary)],
                                capture_output=True, text=True, check=False)
        if result.returncode:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Video concatenation failed: {result.stderr[-1500:]}")
        verified = verify_video(executable, temporary, expected_frames, fps)
        temporary.replace(output)
        verified["file"] = output.name
        return verified


def render_mission(directory, *, width=1280, height=720, fps=25, overwrite=False,
                   timebases=("wall", "sim")):
    if width < 160 or height < 120 or width % 2 or height % 2 or fps <= 0:
        raise ValueError("Use even video dimensions >=160x120 and positive fps")
    if not timebases or len(set(timebases)) != len(timebases) or set(timebases) - {"wall", "sim"}:
        raise ValueError("Select unique timebases from wall and sim")
    directory = Path(directory).resolve()
    executable = ffmpeg_path()
    phases = [phase for phase in PHASES if (directory / phase / "third_person/metadata.json").exists()]
    if not phases:
        raise FileNotFoundError("No phase recordings found")
    outputs = {"wall": directory / "mission-third-person-full.mp4",
               "sim": directory / "mission-third-person-motion.mp4"}
    outputs = {kind: path for kind, path in outputs.items() if kind in timebases}
    protected = [*outputs.values(), directory / "video-manifest.json"]
    protected.extend(directory / phase / "third_person" / filename for phase in phases
                     for filename in (*(f"third-person-{kind}.mp4" for kind in timebases), "video-manifest.json"))
    if not overwrite and any(path.exists() for path in protected):
        raise FileExistsError("Video outputs already exist; use --overwrite explicitly")
    reports = [render_phase(directory / phase / "third_person", phase, executable,
                            width=width, height=height, fps=fps, overwrite=overwrite,
                            timebases=timebases) for phase in phases]
    combined = {}
    with tempfile.TemporaryDirectory(prefix="go2-video-cards-") as work:
        cards = {}
        for phase in phases[1:]:
            cards[phase] = Path(work) / (phase + ".mp4")
            reset_card(executable, cards[phase], phase, width, height, fps)
        for timebase in timebases:
            paths, count = [], 0
            for index, (phase, report) in enumerate(zip(phases, reports)):
                if index:
                    paths.append(cards[phase])
                    count += 2 * fps
                paths.append(directory / phase / "third_person" / f"third-person-{timebase}.mp4")
                count += report["videos"][timebase]["frame_count"]
            combined[timebase] = concat_movies(executable, paths, outputs[timebase], count, fps)
    report = {"schema_version": 1, "phases": reports, "combined": combined,
              "all_phases_recorded": phases == list(PHASES), "missing_phases": [p for p in PHASES if p not in phases],
              "fps": fps, "resolution": [width, height], "reset_card_seconds": 2,
              "timebases": list(timebases),
              "wall_semantics": "Each recorded phase at wall-clock speed, including paused inference. Process setup gaps are replaced by 2-second RESET cards.",
              "sim_semantics": "Each recorded phase at simulation-clock speed. Repeated timestamps (inference waits) collapse. No motion tool or final stop is skipped.",
              "no_task_answers_read": True, "no_model_api": True, "no_physics_rerun": True}
    write_json(directory / "video-manifest.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    report = render_mission(args.experiment_dir, width=args.width, height=args.height,
                            fps=args.fps, overwrite=args.overwrite)
    print(json.dumps({"videos": report["combined"], "missing_phases": report["missing_phases"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
