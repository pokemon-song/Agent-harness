---
path: "episodes/outbound_01_measurements.md"
revision: 3
scope: "roundtrip-82c2daea3ebf493a825a40d1d3d78685"
status: "confirmed"
evidence_ids: ["event-fc908ef9a21540a6a2f4ca18f8fea319","event-cda90a3aaa174c25be8a2177252b25c5","event-413a987d870546328879bcbee395cee1","event-109e14fabe2b4673b702ef5b12696f87","event-d455aa59fbf04c6e99badb3ddf8978ff","event-27c4b11529c04365834daa2770dcc4d1"]
evidence_max_seq: 15
observed_at: "2026-11-11T12:11:58.801286+00:00"
evidence: [{"id":"event-fc908ef9a21540a6a2f4ca18f8fea319","observed_at":"2026-11-11T12:10:44.361056+00:00","seq":10},{"id":"event-cda90a3aaa174c25be8a2177252b25c5","observed_at":"2026-11-11T12:11:00.219045+00:00","seq":11},{"id":"event-413a987d870546328879bcbee395cee1","observed_at":"2026-11-11T12:11:21.536622+00:00","seq":12},{"id":"event-109e14fabe2b4673b702ef5b12696f87","observed_at":"2026-11-11T12:11:33.255241+00:00","seq":13},{"id":"event-d455aa59fbf04c6e99badb3ddf8978ff","observed_at":"2026-11-11T12:11:46.100573+00:00","seq":14},{"id":"event-27c4b11529c04365834daa2770dcc4d1","observed_at":"2026-11-11T12:11:58.801286+00:00","seq":15}]
---
# Outbound leg — tracked points and motion feedback (frames 00002–00006)

Episode `65e33cbb0a944297a54e00cd0ef23872`; scope `roundtrip-82c2daea3ebf493a825a40d1d3d78685`; scene `composite_drawers_gate_b`. `relative_xy_m` is body-frame (x forward, y left) at the listed base pose; the mission column rotates `relative_xy` by that pose's yaw and adds `local_odometry` (origin = mission start, shared by both legs).

## Surface points

| point | image | base pose (x, y, yaw) | uv | range (m) | relative xy (m) | mission xy (m) |
|---|---|---|---|---|---|---|
| point-001 | frame-00002 | (0, 0, 0) | (330, 240) | 1.4316 | (1.431, −0.040) | (1.431, −0.040) |
| point-002 | frame-00002 | (0, 0, 0) | (310, 66) | 3.3060 | (3.304, +0.111) | (3.304, +0.111) |
| point-003 | frame-00002 | (0, 0, 0) | (400, 100) | 3.3717 | (3.288, −0.745) | (3.288, −0.745) |
| point-004 | frame-00003 | (0.0152, 0.5371, 0.0154) | (382, 220) | 1.6703 | (1.648, −0.269) | (1.664, 0.245) |
| point-005 | frame-00003 | (0.0152, 0.5371, 0.0154) | (362, 163) | 5.5898 | (5.546, −0.701) | (5.560, −0.197) |
| point-006 | frame-00003 | (0.0152, 0.5371, 0.0154) | (500, 285) | 1.5094 | (1.376, −0.620) | (1.393, −0.114) |
| point-007 | frame-00004 | (0.0260, 0.7663, 0.0195) | (515, 200) | 1.5493 | (1.395, −0.674) | (1.434, 0.120) |
| point-008 | frame-00005 | (0.5567, 0.7934, 0.0181) | (352, 120) | 2.7632 | (2.752, −0.244) | (3.313, 0.599) |
| point-008 | frame-00006 | (1.0866, 0.8225, 0.0171) | (366, 118) | 2.2388 | (2.222, −0.276) | (3.313, 0.585) |
| point-009 | frame-00006 | (1.0866, 0.8225, 0.0171) | (585, 150) | 2.6858 | (2.176, −1.575) | (3.289, −0.715) |
| point-010 | frame-00006 | (1.0866, 0.8225, 0.0171) | (393, 157) | 4.5729 | (4.469, −0.971) | (5.571, −0.072) |

Reading:
- point-007 lies on the cabinet front face (mission x ≈ 1.43, y ≈ +0.12); it was lost on frame-00005 (`optical_flow_lost`) once the robot moved past the cabinet.
- point-008 is the gate's **left post**: two re-observations 0.62 m apart both resolve to mission x = 3.313 (y 0.585–0.599), a good consistency check → **gate plane x ≈ 3.30 m**.
- point-009 is the **right post** → (3.289, −0.715). With point-008 this puts the opening between y ≈ +0.59 and y ≈ −0.72, centre ≈ **−0.06**, ~1.3 m between the sampled points.
- point-010 is the centre of plaque **B** → mission (5.571, −0.072): B sits directly behind the opening centre, 4.47 m ahead in x.
- Points 001–006 remain invalid (`inconsistent_flow`); 007 lost. Only 008, 009, 010 are live as of frame-00006 — keep them alive for the crossing check.

## Motion feedback (measured, not requested)

1. seq 5 — `move_relative(lateral +0.6)`: 0.537 m, yaw +0.0041, position error 0.065 m.
2. seq 9 — `move_relative(lateral +0.3)`: 0.229 m, yaw +0.0041, error 0.072 m.
3. seq 11 — `move_relative(forward 0.6)`: **0.5311 m** in 3.22 s, heading error 0.0013 rad, position error 0.071 m; clean stop (0.012 m/s).
4. seq 13 — `move_relative(forward 0.6)`: **0.5304 m** in 3.28 s, heading error 0.0010 rad, position error 0.072 m; clean stop (0.012 m/s).

Pattern: forward requests now reliably deliver ≈0.53 m per 0.60 m asked (**~11–12 % short**), so ask for ~1.13 m per desired metre or, better, iterate on odometry. Pose after seq 13 (sim_time 13.92 s): `local_odometry` (1.0866, 0.8225) m, yaw 0.0171 rad; idle, upright (height 0.358 m, pitch −0.0051, roll −0.0169). None of the requests were under-delivered by the planner's miscalibration alone — the base really is where odometry says.

## Visual notes
- frame-00006 (saved with seq 15, the current image): gate posts stand on the checker floor with a black backdrop; all three plaques are legible behind/around the opening — **C** leftmost (u ≈ 315), **B** just inside the opening's left edge (u ≈ 393), **A** to the right of B (u ≈ 470). The drawer cabinet is no longer visible (the robot is beside/past it on its left). C and A have no measured coordinates.
- Every saved image belongs to its named event, not to the present robot state.
