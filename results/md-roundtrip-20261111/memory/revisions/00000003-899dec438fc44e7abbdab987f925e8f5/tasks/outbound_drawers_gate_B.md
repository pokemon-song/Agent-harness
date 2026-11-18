---
path: "tasks/outbound_drawers_gate_B.md"
revision: 3
scope: "roundtrip-82c2daea3ebf493a825a40d1d3d78685"
status: "confirmed"
evidence_ids: ["event-fc908ef9a21540a6a2f4ca18f8fea319","event-cda90a3aaa174c25be8a2177252b25c5","event-413a987d870546328879bcbee395cee1","event-109e14fabe2b4673b702ef5b12696f87","event-d455aa59fbf04c6e99badb3ddf8978ff","event-27c4b11529c04365834daa2770dcc4d1"]
evidence_max_seq: 15
observed_at: "2026-11-11T12:11:58.801286+00:00"
evidence: [{"id":"event-fc908ef9a21540a6a2f4ca18f8fea319","observed_at":"2026-11-11T12:10:44.361056+00:00","seq":10},{"id":"event-cda90a3aaa174c25be8a2177252b25c5","observed_at":"2026-11-11T12:11:00.219045+00:00","seq":11},{"id":"event-413a987d870546328879bcbee395cee1","observed_at":"2026-11-11T12:11:21.536622+00:00","seq":12},{"id":"event-109e14fabe2b4673b702ef5b12696f87","observed_at":"2026-11-11T12:11:33.255241+00:00","seq":13},{"id":"event-d455aa59fbf04c6e99badb3ddf8978ff","observed_at":"2026-11-11T12:11:46.100573+00:00","seq":14},{"id":"event-27c4b11529c04365834daa2770dcc4d1","observed_at":"2026-11-11T12:11:58.801286+00:00","seq":15}]
---
# Outbound leg: drawers → green gate → letter B

Scope `roundtrip-82c2daea3ebf493a825a40d1d3d78685`; scene `composite_drawers_gate_b`; episode `65e33cbb0a944297a54e00cd0ef23872`.

## Registered instruction (public announcement, seq 1, 2026-11-11T12:07:44Z)
Ordered stages (stages_ordered = true):

1. `avoid_drawers` — go around the front drawer cabinet so the **base origin** passes behind the cabinet body and leaves body clearance.
2. `pass_gate` — then pass the base origin through the opening of the **green** rectangular door frame; while crossing, the base origin must be inside the opening; continue at least **0.8 m** past the frame plane in the crossing direction.
3. `approach_b` — among the white letter plaques A, B, C behind the gate, select **B**, move to about **1.0 m** from B's plaque center, face B, stop, then call `finish`.

Limits: no contact with cabinet, frame, plaques or any scene object (collision-free required). Stages must be done in order; it is forbidden to skip the gate and head straight for B. Tolerances at finish: distance 0.18 m, heading 0.25 rad, stopped speed ≤ 0.12 m/s, stopped yaw rate ≤ 0.25 rad/s, reference body = base.

Note in the instruction: the front camera sits 0.32 m ahead of the base origin, so the gate leaving the image does not prove the base has passed it. Track reliable surface points on the frame while it is visible and confirm crossing plus the 0.8 m extra from local_odometry and actual motion feedback — do not substitute requested motion for measured displacement.

After `finish` only the outbound leg ends: the planner's conversation, temporary notes and tracked points are cleared, but the robot, scene, simulation time, pose and local_odometry do **not** reset. The return leg must be planned from this memory plus fresh perception, retracing the outbound route.

## Coordinates and frames
- coordinate_frame_id `roundtrip-82c2daea3ebf493a825a40d1d3d78685`; coordinate_reference "continuous_mission_start; physical world and odometry origin are preserved".
- `local_odometry`: origin and horizontal axes fixed at the mission's initial base pose, preserved across planner-context resets (shared by outbound and return legs).
- `tracked_points.relative_xy_m`: relative to the current base body reference, x forward, y left, metres; yaw positive left.
- Body axes x forward, y left, z up; camera frame front_rgbd_optical at +0.32 m forward, +0.08 m up in body, optical x right / y down / z forward.

## Scene geometry in the mission frame (x forward, y left)
- **Cabinet:** front plane x ≈ 1.43 m (point-001 at y ≈ −0.04, point-007 at y ≈ +0.12). Depth and full width still unmeasured; the corner track was lost on frame-00005 and the cabinet is out of the camera view in frame-00006.
- **Green gate:** frame plane x ≈ **3.30 m** (point-008 re-observed on two poses lands on x = 3.313 both times; point-009 at 3.289). Tracked post points: (3.313, +0.59) and (3.289, −0.715) → opening lies between roughly y = +0.59 and y = −0.72, centre ≈ **y = −0.06**, ~1.3 m between the sampled points (clear inner width a little less).
- **Plaque B:** centre at mission ≈ **(5.571, −0.072)** (point-010, range 4.57 m), i.e. directly behind the opening centre; C and A are legible in frame-00006 (C leftmost, A right of B) but were not tracked, so their coordinates are unknown.

## Route status (through frame-00006, sim_time 13.92 s)
- **Pose:** `local_odometry` (1.0866, 0.8225) m, yaw 0.0171 rad; idle, upright (height 0.358 m, pitch −0.0051, roll −0.0169). Net so far ≈1.09 m forward, ≈0.82 m left of the start line.
- **Executed:** `move_relative(lateral +0.6)`, `(lateral +0.3)`, `(forward 0.6)`, `(forward 0.6)` — all succeeded. Measured increments 0.537, 0.229, 0.5311, 0.5304 m: every forward request delivers ≈0.53 m per 0.6 m asked (~11–12 % short); lateral shims also under-delivered. Always confirm with odometry.
- **Stage `avoid_drawers`: still incomplete.** Base x = 1.09 m has not yet passed the cabinet front plane (x ≈ 1.43 m) and the cabinet rear is unknown. Laterally the robot is clear and to the cabinet's **left** (cabinet front-face points sit at y ≈ +0.12 and below, base at y = 0.82 → ~0.6 m clearance).
- **Stage `pass_gate`: not started.** No crossing and no 0.8 m overshoot. Gate plane is 2.21 m ahead of the base.
- **Stage `approach_b`: not started.** B is 4.47 m ahead in x; a 1.0 m standoff therefore sits near mission x ≈ 4.57, y ≈ −0.07, facing +x (yaw ≈ 0) — which also satisfies gate plane + 0.8 m (x ≥ 4.10).
- **Key correction (supersedes the earlier rough estimate of an opening centre near y ≈ +0.2):** the robot's current y = +0.82 m is about **0.87 m left** of the gate/plaque-B line at y ≈ −0.06. The path must therefore move **right (−y)**, not further left.
- **Ordering constraint:** the rightward shift must not begin before the base has cleared the cabinet body (cabinet occupies y ≲ +0.2 in front of the base, x from 1.43 m to an unmeasured rear), otherwise it drives back into it. Clear the cabinet in +x first, then shift toward y ≈ −0.06 and cross at x ≈ 3.30.
- **Live tracked points:** only point-008 (left post, mission (3.313, ≈0.59)), point-009 (right post) and point-010 (B) are valid as of frame-00006; points 001–006 were invalidated (`inconsistent_flow`) and 007 was lost. Keep 008/009 alive to confirm the crossing by odometry.

**Return-leg note:** the outbound leg will finish near (4.57, −0.07) facing +x. The return must first go back through the same opening heading −x (cross at y ≈ −0.06), then mirror the outbound drift by moving left (+y) to ≈ +0.8 m to clear the cabinet on its left side, and finish near (0, 0) with yaw ≈ 0. Cabinets' rear extent is still unknown — re-measure on the way back.

Detailed point table and per-move feedback: `episodes/outbound_01_measurements.md`.
