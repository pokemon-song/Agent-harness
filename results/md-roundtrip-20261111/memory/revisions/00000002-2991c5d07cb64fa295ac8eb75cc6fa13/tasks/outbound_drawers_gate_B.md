---
path: "tasks/outbound_drawers_gate_B.md"
revision: 2
scope: "roundtrip-82c2daea3ebf493a825a40d1d3d78685"
status: "confirmed"
evidence_ids: ["event-441db0767338435185050b2808c585e8","event-6523fceee54e40bc84c7238ef6ad86f6","event-27ab25176ee04953807c9cbb06018152","event-6ff4106e5bb9443ca682d4aea22987a6","event-efb2fa5b7c884c52951ea3ceabf7a267","event-aeeb08013da345dc81549ac62751618a","event-08eaa5e19a62437ab1f37cc461f294b9","event-00e171239188472496e0dbdf2a90849a"]
evidence_max_seq: 9
observed_at: "2026-11-11T12:10:37.489902+00:00"
evidence: [{"id":"event-441db0767338435185050b2808c585e8","observed_at":"2026-11-11T12:08:02.542183+00:00","seq":2},{"id":"event-6523fceee54e40bc84c7238ef6ad86f6","observed_at":"2026-11-11T12:08:05.425996+00:00","seq":3},{"id":"event-27ab25176ee04953807c9cbb06018152","observed_at":"2026-11-11T12:08:26.133289+00:00","seq":4},{"id":"event-6ff4106e5bb9443ca682d4aea22987a6","observed_at":"2026-11-11T12:08:47.735518+00:00","seq":5},{"id":"event-efb2fa5b7c884c52951ea3ceabf7a267","observed_at":"2026-11-11T12:08:56.372775+00:00","seq":6},{"id":"event-aeeb08013da345dc81549ac62751618a","observed_at":"2026-11-11T12:09:16.338821+00:00","seq":7},{"id":"event-08eaa5e19a62437ab1f37cc461f294b9","observed_at":"2026-11-11T12:10:03.735958+00:00","seq":8},{"id":"event-00e171239188472496e0dbdf2a90849a","observed_at":"2026-11-11T12:10:37.489902+00:00","seq":9}]
---
# Outbound leg: drawers → green gate → letter B

Scope `roundtrip-82c2daea3ebf493a825a40d1d3d78685`; scene `composite_drawers_gate_b`; episode `65e33cbb0a944297a54e00cd0ef23872`.

## Registered instruction (public announcement, seq 1, 2026-11-11T12:07:44Z)
Ordered stages (stages_ordered = true):

1. `avoid_drawers` — go around the front drawer cabinet so the **base origin** passes behind the cabinet body and leaves body clearance.
2. `pass_gate` — then pass the base origin through the opening of the **green** rectangular door frame; while crossing, the base origin must be inside the opening; continue at least **0.8 m** past the frame plane in the crossing direction.
3. `approach_b` — among the white letter plaques A, B, C behind the gate, select **B**, move to about **1.0 m** from B's plaque center, face B, stop, then call `finish`.

Limits: no contact with cabinet, frame, plaques or any scene object (collision-free required). Stages must be done in order; it is forbidden to skip the gate and head straight for B. Tolerances at finish: distance 0.18 m, heading 0.25 rad, stopped speed ≤ 0.12 m/s, stopped yaw rate ≤ 0.25 rad/s, reference body = base.

Note in the instruction: the front camera sits 0.32 m ahead of the base origin, so the gate leaving the image does not prove the base has passed it. Select reliable surface points on the frame while it is visible and confirm crossing plus the 0.8 m extra from local_odometry and actual motion feedback — do not substitute requested motion for measured displacement.

After `finish` only the outbound leg ends: the planner's conversation, temporary notes and tracked points are cleared, but the robot, scene, simulation time, pose and local_odometry do **not** reset. The return leg must be planned from this memory plus fresh perception, retracing the outbound route.

## Coordinates and frames
- coordinate_frame_id `roundtrip-82c2daea3ebf493a825a40d1d3d78685`; coordinate_reference "continuous_mission_start; physical world and odometry origin are preserved".
- `local_odometry`: origin and horizontal axes fixed at the mission's initial base pose, preserved across planner-context resets (shared by outbound and return legs).
- `tracked_points.relative_xy_m`: relative to the current base body reference, x forward, y left, metres; yaw positive left.
- Body axes x forward, y left, z up; camera frame front_rgbd_optical at +0.32 m forward, +0.08 m up in body, optical x right / y down / z forward.

## Initial observation (frame-00002, sim_time 1.0 s)
- local_odometry translation [0.0, 0.0] m, yaw 0.0 rad — this is the shared origin for the whole round trip.
- Robot idle, speed ≈0.017 m/s, yaw rate ≈0.014 rad/s, not fallen. Body height 0.3587 m, pitch −0.0299 rad, roll −0.0185 rad (height limits 0.24–0.36 m, pitch ±0.2 rad, height + 0.1934·|pitch| ≤ 0.38 m — currently satisfied).
- Capabilities: rgb, depth, odometry, metric geometry, planar motion, lateral motion, turn in place, height/pitch control; moving height/pitch control unavailable.
- 640×360 RGB: a brown drawer cabinet stands directly ahead, centered in view; the green rectangular door frame is visible above and behind it. Three surface points were then tracked from this frame (table in `episodes/outbound_01_measurements.md`).

## Route status (through frame-00004, sim_time 7.42 s)
- **Cabinet:** point-001 sits on its slotted front face at 1.4316 m ahead of the shared origin (y ≈ −0.04) ⇒ cabinet front plane ≈ **x = 1.43 m** in the mission frame; its horizontal extent is still unmeasured.
- **Executed:** `move_relative(lateral_m=+0.6)` then `move_relative(lateral_m=+0.3)`, both succeeded; measured increments 0.537 m and 0.229 m (both ~10–25 % short of the request). Current pose (0.0260, 0.7663) m, yaw 0.0195 rad, idle, upright.
- **Net displacement so far:** ≈0.77 m **left** (+y) of the start line, forward only ≈0.03 m.
- **Stage truth:** the base is still ~1.4 m before the cabinet front plane, so `avoid_drawers` is **not** complete. No gate crossing, no 0.8 m overshoot, no approach to B has happened.
- **Green gate:** stands ahead at roughly 3.3 m with its opening slightly right of the camera axis; posts reach the ground and the top bar is above cabinet height (points 002/003, high far surfaces).
- **Plaques behind the gate (frame-00004):** white letters **C** (leftmost) and **B** (just right of C) were legible; a third plaque nearer the gate centreline had an unreadable letter. **A is not positively identified** — it may be that third plaque or hidden behind the cabinet.
- **All six tracked points were invalidated** (`inconsistent_flow`) by frame-00003/frame-00004. Re-acquire fresh gate and ground points before using any crossing check; the old coordinates are stale.
- **Rough, unconfirmed estimate:** gate opening centre near mission y ≈ +0.2 m, so the current y ≈ +0.77 m may be ~0.5 m left of the opening centre. Verify with fresh depth samples; do not act on this number alone.

**Return-leg note:** the outbound leg has only shifted left (+y); the return must come back through the same gate opening, then mirror that with rightward (−y) shims while staying clear of the cabinet on its left (+y) side, finishing near (0, 0) with yaw ≈ 0.

Detailed point table and per-move feedback: `episodes/outbound_01_measurements.md`.
