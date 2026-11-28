"""Recreate one measured outbound prefix up to the first valid A stop.

Only the transition gate uses the private evaluator. The shared Markdown handed
to both arms is built from public, chronological RGB-D and odometry samples.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .harness import Harness
from .memory_perception import TaskLabelRGBD, task_targets
from .perception import PointTracker
from .types import ToolCall


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class _Pose:
    step: int
    frame: str
    time: float
    x: float
    y: float
    yaw: float

    @classmethod
    def from_observation(cls, observation, step):
        odometry = observation["local_odometry"]
        return cls(step, observation["image_id"], observation["sim_time_s"],
                   *odometry["translation_m"], odometry["yaw_rad"])

    def stamp(self):
        return f"去程步 {self.step} / {self.frame} / t={self.time:.2f}s"

    def describe(self):
        return (f"{self.stamp()}：实测机身里程计 x={self.x:.3f}, y={self.y:.3f} m，"
                f"yaw={self.yaw:.3f} rad；同一去程起点坐标系，x 向前、y 向左。")


class _OutboundNotes:
    def __init__(self, task_id, *, sample_distinct_views=False):
        self.sensor = TaskLabelRGBD(task_targets(task_id))
        self.route = []
        self.by_target = {}
        self.last_pose = None
        self.sample_distinct_views = sample_distinct_views
        self._last_sighting_view = {}

    def observe(self, harness, step):
        observation = harness.observation()
        frame = harness.frame
        pose = _Pose.from_observation(observation, step)
        previous = self.route[-1] if self.route else None
        self.last_pose = pose
        if previous is None:
            self.route.append(pose)
        else:
            distance = math.hypot(pose.x - previous.x, pose.y - previous.y)
            angle = abs(math.remainder(pose.yaw - previous.yaw, 2 * math.pi))
            if distance >= .75 or (distance >= .20 and angle >= .60):
                self.route.append(pose)

        view, masks = self.sensor.inspect(observation, frame.rgb, frame.depth_m, frame.intrinsics)
        for candidate in view["target_candidates"]:
            label = candidate["task_label"]
            track = None
            for point in observation["tracked_points"]:
                matched = self.sensor.associate(point, view, masks, frame.depth_m, frame.intrinsics)
                if matched and matched["task_label"] == label:
                    track = point
                    break
            # Preserve an observation at each end rather than repeatedly publishing static frames.
            key = label, bool(track)
            if self.sample_distinct_views:
                xyz = tuple(candidate["camera_xyz_m"])
                previous_view = self._last_sighting_view.get(key)
                if previous_view is not None:
                    station, old_xyz = previous_view
                    if (math.hypot(pose.x - station.x, pose.y - station.y) < .20
                            and abs(math.remainder(pose.yaw - station.yaw, 2 * math.pi)) < .30
                            and math.dist(xyz, old_xyz) < .25):
                        continue
                self._last_sighting_view[key] = (pose, xyz)
            records = self.by_target.setdefault(key, [])
            pixel = candidate["pixel_uv"]
            if track:
                forward, left = track["relative_xy_m"]
                c, s = math.cos(pose.yaw), math.sin(pose.yaw)
                x, y = pose.x + c * forward - s * left, pose.y + s * forward + c * left
                text = (f"{pose.stamp()}：目标 {label}（{candidate['description']}）同色区域上的"
                        f"有效 RGB-D 跟踪点 {track['point_id']}，机身前/左=({forward:.3f},{left:.3f}) m；"
                        f"结合实测机身里程计，目标可见表面在去程起点坐标系约 ({x:.3f},{y:.3f}) m。"
                        "表面点与物体中心可能有偏差。")
                if self.sample_distinct_views:
                    text += f"当时机身去程起点坐标 ({pose.x:.3f},{pose.y:.3f}) m，yaw={pose.yaw:.3f} rad。"
            else:
                xyz = ", ".join(f"{v:.3f}" for v in candidate["camera_xyz_m"])
                text = (f"{pose.stamp()}：目标 {label}（{candidate['description']}）颜色匹配候选，"
                        f"像素 ({pixel[0]},{pixel[1]})，实测相机光学坐标（右、下、前）=({xyz}) m；"
                        f"当时机身去程起点坐标 ({pose.x:.3f},{pose.y:.3f}) m，yaw={pose.yaw:.3f} rad。")
            if len(records) < 2:
                records.append(text)
            else:
                records[-1] = text

    def documents(self):
        sections = {}
        for name, paired in sorted(self.by_target):
            filename = f"targets/target-{name}.md"
            lines = sections.setdefault(filename, [])
            lines.extend(self.by_target[(name, paired)])
        route = self.route.copy()
        if route[-1] != self.last_pose:
            route.append(self.last_pose)
        for start in range(0, len(route), 6):
            sections[f"route/leg-{start // 6 + 1:02d}.md"] = [p.describe() for p in route[start:start + 6]]
        sections["handoff/last-observation.md"] = [self.last_pose.describe()]
        documents = {}
        index = 0
        for filename, rows in sections.items():
            body = []
            for row in dict.fromkeys(rows):
                index += 1
                body.append(f"[E{index:03d}] {row}")
            documents[filename] = "# " + filename + "\n\n" + "\n".join(body) + "\n"
        return documents


def _compare_observations(current, archived, changes):
    # The legacy archive declared the pre-penalty collision policy. The fork may
    # expose additional policy fields. No measured sensor or pose field is ignored.
    def comparable(obs):
        obs = dict(obs)
        rules = dict(obs["completion_requirements"])
        for key in ("collision_free_required", "cabinet_contact_ends_episode", "collision_penalty_points"):
            rules.pop(key, None)
        obs["completion_requirements"] = rules
        return obs

    if current["completion_requirements"] != archived["completion_requirements"]:
        changes.add("completion_requirements.collision_policy")
    if comparable(current) != comparable(archived):
        raise ValueError("Live and archived public observations disagree")


def _verify_trajectory(rows, archive):
    digest = hashlib.sha256()
    with (archive / "trajectory.jsonl").open() as stream:
        for number, measured in enumerate(rows, 1):
            line = stream.readline()
            if not line:
                raise ValueError("Archived trajectory ends before A")
            archived = json.loads(line)
            for key in ("base_xyz_m", "yaw_rad", "sim_time_s", "velocity_world_mps", "command"):
                if not np.allclose(measured[key], archived[key], atol=1e-9, rtol=0):
                    raise ValueError(f"Archived trajectory disagrees at tick {number}, field {key}")
            if archived["fallen"] or archived["collision_steps"] or measured["fallen"] or measured["collision_steps"]:
                raise ValueError("Outbound trajectory contains an unsafe step")
            digest.update(json.dumps([measured[k] for k in ("sim_time_s", "base_xyz_m", "yaw_rad", "command")],
                                     separators=(",", ":"), allow_nan=False).encode() + b"\n")
    return digest.hexdigest()


def restore_at_a(backend, archive: Path):
    """Return (same-world harness, shared immutable-by-convention Markdown, private audit)."""
    archive = Path(archive)
    manifest = json.loads((archive / "manifest.json").read_text())
    task_id = backend.scenario.task_id
    if task_id != "memory_ab_occluded_v1" or manifest["task_id"] != task_id or not manifest["external_memory"]:
        raise ValueError("Expected a matching occluded A/B outbound archive")
    if backend.trajectory or backend._frame_counter:
        raise ValueError("Start replay with a fresh backend")
    harness = Harness(backend, manifest["task"])
    notes = _OutboundNotes(task_id)
    notes.observe(harness, 0)
    completed = inputs = actions = 0
    calls = None
    changes = set()
    evidence = hashlib.sha256()
    with (archive / "events.jsonl").open() as stream:
        for line in stream:
            entry = json.loads(line)
            event = entry["event"]
            if event == "model_input":
                if entry["turn"] != completed or inputs != completed:
                    raise ValueError("Non-contiguous archived input stream")
                relative = Path(entry["image_path"])
                if relative.parent != Path("frames") or relative.suffix != ".png":
                    raise ValueError("Expected an archived frame")
                with Image.open(archive / relative) as im:
                    if not np.array_equal(np.asarray(im.convert("RGB")), harness.frame.rgb):
                        raise ValueError(f"RGB mismatch at decision {completed + 1}")
                evidence.update(harness.frame.rgb.tobytes())
                inputs += 1
            elif event == "model_response":
                if entry["turn"] != completed or inputs != completed + 1 or calls is not None:
                    raise ValueError("Unmatched archived decision")
                calls = [ToolCall(**c) for c in entry["calls"]]
            elif event == "tool_results":
                if entry["turn"] != completed or calls is None:
                    raise ValueError("Unmatched archived action")
                expected = entry["results"]
                gated = any(r.get("error_code") in {"MEMORY_CONSTRAINT", "MEMORY_NO_PROGRESS"} for r in expected)
                if not gated:
                    actual = harness.dispatch_many(calls)
                    expected = [{k: v for k, v in r.items() if k not in {"plan_id", "subgoal_id"}} for r in expected]
                    if actual != expected:
                        raise ValueError(f"Native tool results differ at decision {completed + 1}")
                    actions += 1
                _compare_observations(harness.observation(), entry["next_observation"], changes)
                completed += 1
                notes.observe(harness, completed)
                calls = None
                if backend._ordered_evaluator.a_witness is not None:
                    break
            elif event in {"final_stop", "episode_end", "discarded_response"}:
                raise ValueError("Episode ended before the measured A stop")
        else:
            raise ValueError("No qualifying A stop found in the archive")

    path_digest = _verify_trajectory(backend.trajectory, archive)
    frame = harness.frame
    if backend._collision_steps or backend._fallen_ever:
        raise ValueError("Unsafe A handoff")
    docs = notes.documents()
    pose = notes.last_pose
    audit = {"source_decisions": completed, "input_frames_checked": inputs, "native_actions": actions,
             "controls_checked": len(backend.trajectory), "seed": manifest["seed"],
             "sim_time_s": pose.time, "image_id": frame.frame_id,
             "public_pose_m_yaw": [pose.x, pose.y, pose.yaw], "prefix_trajectory_sha256": path_digest,
             "image_sha256": hashlib.sha256(frame.rgb.tobytes()).hexdigest(),
             "verified_input_images_sha256": evidence.hexdigest(), "documents_sha256": _digest(docs),
             "observation_schema_differences": sorted(changes),
             "ground_truth_used_for_transition_only": True}
    harness.tracker = PointTracker()
    harness.tracker.update(frame)
    harness.cache.clear()
    return harness, docs, audit
