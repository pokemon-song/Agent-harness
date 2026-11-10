"""Render every recorded complex-mission phase at simulation speed by default."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import render_memory_mission_video as video

PHASES = ("dispatch", "fragments", "revision", "full", "stale", "empty")
LABELS = {"dispatch": "SHIFT 1: INITIAL DISPATCH",
          "fragments": "SHIFT 2: CODE FRAGMENT + LOCATION MAP",
          "revision": "SHIFT 3: REVISED DISPATCH + ARCHIVED NOTICE",
          "full": "DELIVERY: FULL MEMORY", "stale": "CONTROL: PRE-REVISION MEMORY",
          "empty": "CONTROL: EMPTY MEMORY"}


def camera(mujoco, phase):
    result = mujoco.MjvCamera()
    result.type = mujoco.mjtCamera.mjCAMERA_FREE
    x, distance = {"dispatch": (2.65, 9.0), "fragments": (2.0, 7.2),
                   "revision": (1.6, 6.4)}.get(phase, (3.5, 11.8))
    result.lookat[:] = [x, 0., .25]
    result.distance, result.azimuth, result.elevation = distance, 45, -65
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--include-wall-time", action="store_true",
                        help="Also encode actual compute time; fast simulation produces visible jumps")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    video.PHASES, video.LABELS, video.fixed_camera = PHASES, LABELS, camera
    report = video.render_mission(args.experiment_dir, width=args.width, height=args.height, fps=25,
        overwrite=args.overwrite, timebases=("wall", "sim") if args.include_wall_time else ("sim",))
    print(report["combined"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
