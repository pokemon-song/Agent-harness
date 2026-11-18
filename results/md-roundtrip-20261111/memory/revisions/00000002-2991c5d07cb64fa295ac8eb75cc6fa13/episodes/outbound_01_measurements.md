---
path: "episodes/outbound_01_measurements.md"
revision: 2
scope: "roundtrip-82c2daea3ebf493a825a40d1d3d78685"
status: "confirmed"
evidence_ids: ["event-441db0767338435185050b2808c585e8","event-6523fceee54e40bc84c7238ef6ad86f6","event-27ab25176ee04953807c9cbb06018152","event-6ff4106e5bb9443ca682d4aea22987a6","event-efb2fa5b7c884c52951ea3ceabf7a267","event-aeeb08013da345dc81549ac62751618a","event-08eaa5e19a62437ab1f37cc461f294b9","event-00e171239188472496e0dbdf2a90849a"]
evidence_max_seq: 9
observed_at: "2026-11-11T12:10:37.489902+00:00"
evidence: [{"id":"event-441db0767338435185050b2808c585e8","observed_at":"2026-11-11T12:08:02.542183+00:00","seq":2},{"id":"event-6523fceee54e40bc84c7238ef6ad86f6","observed_at":"2026-11-11T12:08:05.425996+00:00","seq":3},{"id":"event-27ab25176ee04953807c9cbb06018152","observed_at":"2026-11-11T12:08:26.133289+00:00","seq":4},{"id":"event-6ff4106e5bb9443ca682d4aea22987a6","observed_at":"2026-11-11T12:08:47.735518+00:00","seq":5},{"id":"event-efb2fa5b7c884c52951ea3ceabf7a267","observed_at":"2026-11-11T12:08:56.372775+00:00","seq":6},{"id":"event-aeeb08013da345dc81549ac62751618a","observed_at":"2026-11-11T12:09:16.338821+00:00","seq":7},{"id":"event-08eaa5e19a62437ab1f37cc461f294b9","observed_at":"2026-11-11T12:10:03.735958+00:00","seq":8},{"id":"event-00e171239188472496e0dbdf2a90849a","observed_at":"2026-11-11T12:10:37.489902+00:00","seq":9}]
---
# Outbound leg — tracked points and motion feedback (frames 00002–00004)

Episode `65e33cbb0a944297a54e00cd0ef23872`; scope `roundtrip-82c2daea3ebf493a825a40d1d3d78685`; scene `composite_drawers_gate_b`. `relative_xy_m` is body-frame (x forward, y left) for the base pose listed; mission-frame pose comes from `local_odometry` (origin = mission start, shared by both legs).

## Surface points

| point | image | base pose (x, y, yaw) | uv | range (m) | relative xy (m) |
|---|---|---|---|---|---|
| point-001 | frame-00002 | (0, 0, 0) | (330, 240) | 1.4316 | (1.431, −0.040) |
| point-002 | frame-00002 | (0, 0, 0) | (310, 66) | 3.3060 | (3.304, +0.111) |
| point-003 | frame-00002 | (0, 0, 0) | (400, 100) | 3.3717 | (3.288, −0.745) |
| point-004 | frame-00003 | (0.0152, 0.5371, 0.0154) | (382, 220) | 1.6703 | (1.648, −0.269) |
| point-005 | frame-00003 | (0.0152, 0.5371, 0.0154) | (362, 163) | 5.5898 | (5.546, −0.701) |
| point-006 | frame-00003 | (0.0152, 0.5371, 0.0154) | (500, 285) | 1.5094 | (1.376, −0.620) |

Reading: point-001 is on the drawer cabinet's slotted front face, the closest solid surface (1.43 m ahead of the origin). point-002/003 are high, far surfaces (~3.3 m), plausibly on or near the green gate plane since it is the only green structure. point-004/006 are low, close, forward-right surfaces, most consistent with the cabinet's near-left corner region; point-005 is far background at 5.6 m. **All six were already invalid on frame-00003/frame-00004 (`inconsistent_flow`)**, so none can confirm gate crossing and none should be reused as-is.

## Motion feedback (measured, not requested)

1. seq 5 — `move_relative(forward_m=0, lateral_m=+0.6)`: succeeded. Measured increment (0.0152, 0.5371) m, yaw delta +0.0041 rad, position error 0.065 m, 3.96 s sim time; stop clean (speed ≈0.013 m/s).
2. seq 9 — `move_relative(forward_m=0, lateral_m=+0.3)`: succeeded. Measured increment (0.0143, 0.2290) m, yaw delta +0.0041 rad, position error 0.072 m, 2.46 s sim time; stop clean (speed ≈0.018 m/s).

Resulting `local_odometry`: (0.0260, 0.7663) m, yaw 0.0195 rad at sim_time 7.42 s. Robot idle, not fallen, body height 0.363 m, pitch −0.0067 rad, roll −0.0239 rad — inside the posture limits. Both lateral shims under-delivered by roughly 10–25 %; expect the same on later relative moves and confirm with odometry.

## Visual notes

- frame-00004 (image saved with seq 9): green gate frame centred slightly right of the camera axis, posts reaching the ground, top bar well above the cabinet; brown drawer cabinet ahead-right and near (~1.5 m) with three dark slots on the face turned toward the robot; white plaques behind the opening with **C** leftmost and **B** just right of it legible, and a third plaque nearer the gate centreline whose letter could not be read.
- The saved image belongs to its named event, not to the present robot state.
