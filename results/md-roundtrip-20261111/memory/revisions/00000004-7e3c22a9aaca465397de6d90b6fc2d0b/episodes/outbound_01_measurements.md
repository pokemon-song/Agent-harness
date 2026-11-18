---
path: "episodes/outbound_01_measurements.md"
revision: 4
scope: "roundtrip-82c2daea3ebf493a825a40d1d3d78685"
status: "confirmed"
evidence_ids: ["event-95e6340cbf424365af4238e5cde6c63a","event-492fcacf6c1e4a05aa6dd78db1c124f0","event-81a3e2976f3a42e9bebc9360a7be43c0","event-4ca83b791abd4ae59fd43a807053ba90"]
evidence_max_seq: 19
observed_at: "2026-11-11T12:15:35.035162+00:00"
evidence: [{"id":"event-95e6340cbf424365af4238e5cde6c63a","observed_at":"2026-11-11T12:14:08.424535+00:00","seq":16},{"id":"event-492fcacf6c1e4a05aa6dd78db1c124f0","observed_at":"2026-11-11T12:14:20.486058+00:00","seq":17},{"id":"event-81a3e2976f3a42e9bebc9360a7be43c0","observed_at":"2026-11-11T12:14:44.836966+00:00","seq":18},{"id":"event-4ca83b791abd4ae59fd43a807053ba90","observed_at":"2026-11-11T12:15:35.035162+00:00","seq":19}]
---
# Outbound leg — tracked points and motion feedback (frames 00002–00009)

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
| point-011 | frame-00007 | (1.5655, 0.9354, 0.0203) | (392, 200) | 1.7730 | (1.7419, −0.3308) | (3.314, 0.640) |

Reading:
- point-007 lies on the cabinet front face (mission x ≈ 1.43, y ≈ +0.12); it was lost on frame-00005 (`optical_flow_lost`) once the robot moved past the cabinet. The cabinet has not been imaged since frame-00004.
- point-008 is the gate's **left post**: two re-observations 0.62 m apart both resolve to mission x = 3.313 → **gate plane x ≈ 3.30 m**.
- point-011 (frame-00007) is a green-frame surface point in that same plane, mission (3.314, 0.640) — within 0.05 m of point-008's y, so it is a **third independent confirmation of gate plane x ≈ 3.31**. It was lost on frame-00008 (`optical_flow_lost`).
- point-009 is the **right post** → (3.289, −0.715). With point-008 this puts the sampled opening edges at y ≈ +0.60…+0.64 and y ≈ −0.72, centre ≈ **−0.06**.
- point-010 is the centre of plaque **B** → mission (5.571, −0.072), straight behind the opening centre.
- Points 001–006 are invalid (`inconsistent_flow`); 007, 009, 011 lost. **As of frame-00009 no tracked point is live** — use odometry plus fresh perception, and keep 010's saved mission position as the B estimate.

## Motion feedback (measured, not requested)

1. seq 5 — `move_relative(lateral +0.6)`: 0.537 m, yaw +0.0041, position error 0.065 m.
2. seq 9 — `move_relative(lateral +0.3)`: 0.229 m, yaw +0.0041, error 0.072 m.
3. seq 11 — `move_relative(forward 0.6)`: **0.5311 m** in 3.22 s, heading error 0.0013 rad, position error 0.071 m.
4. seq 13 — `move_relative(forward 0.6)`: **0.5304 m** in 3.28 s, heading error 0.0010 rad, error 0.072 m.
5. seq 16 — `move_relative(forward 0.55, lateral +0.1)`: start-frame translation (0.4808, **+0.1047**) m in 2.96 s, heading error 0.0032 rad, error 0.069 m → pose (1.5655, 0.9354), yaw 0.0203.
6. seq 18 — `move_relative(forward 0.6)`: **0.5301 m** (lateral 0.0126) in 3.22 s, heading error 0.0038 rad, error 0.071 m → (2.0953, 0.9588), yaw 0.0164.
7. seq 19 — `move_relative(forward 0.6)`: **0.5299 m** in 3.22 s, heading error 0.0027 rad, error 0.0713 m → (2.6249, 0.9804), yaw 0.0137, idle, upright.

Pattern over 7 executed moves: forward requests deliver **0.530–0.531 m per 0.60 m asked (≈88 %, ~12 % short)**; lateral shims also under-deliver; heading error stays ≤0.004 rad, position error ≈0.07 m, clean stops (≤0.03 m/s). Ask ≈1.13 m per desired metre or iterate on odometry — never sum requests. All requests were delivered, so odometry is trustworthy.

## Visual notes
- frame-00006: gate posts on the checker floor against a black backdrop; all three plaques legible — **C** leftmost (u ≈ 315), **B** just inside the opening's left edge (u ≈ 393), **A** right of B (u ≈ 470). Left→right in mission y the order is C (+), B (−0.07), A (−).
- frame-00009 (saved with seq 19, the current image): the green frame's near **left upright fills the right ~1/4** of the image (u ≳ 480); **C** (u ≈ 330) and **B** (u ≈ 443, clipped by the upright) are still visible ahead, **A** is not. The camera sits at mission x ≈ 2.945 m, still ~0.37 m short of the gate plane x ≈ 3.30, so the visible gate matches odometry: the base has **not** crossed yet.
- Every saved image belongs to its named event, not to the present robot state.
