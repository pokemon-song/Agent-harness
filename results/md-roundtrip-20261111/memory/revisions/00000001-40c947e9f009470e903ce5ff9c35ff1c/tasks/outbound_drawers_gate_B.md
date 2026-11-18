---
path: "tasks/outbound_drawers_gate_B.md"
revision: 1
scope: "roundtrip-82c2daea3ebf493a825a40d1d3d78685"
status: "confirmed"
evidence_ids: ["event-6cf0048c2b8346fab57fc80bfedc1e7c"]
evidence_max_seq: 1
observed_at: "2026-11-11T12:07:44.826444+00:00"
evidence: [{"id":"event-6cf0048c2b8346fab57fc80bfedc1e7c","observed_at":"2026-11-11T12:07:44.826444+00:00","seq":1}]
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
- 640×360 RGB: a brown drawer cabinet stands directly ahead, centered in view; the green rectangular door frame is visible above and behind it. No metric ranges were recorded and `tracked_points` was empty.

## Route knowledge so far
- No motion has been executed in this batch; there are no measured waypoints yet. Which side of the drawer cabinet will be used, the gate opening extent/center, the achieved 0.8 m overshoot, and B's pose relative to the gate are all **unknown** until observed and must not be invented.
