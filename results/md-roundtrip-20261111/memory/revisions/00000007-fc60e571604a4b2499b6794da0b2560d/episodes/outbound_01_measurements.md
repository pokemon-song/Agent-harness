---
path: "episodes/outbound_01_measurements.md"
revision: 7
scope: "roundtrip-82c2daea3ebf493a825a40d1d3d78685"
status: "confirmed"
evidence_ids: ["event-0cffb5888a264aebbcf50afd45486275","event-362cb971d0d9491f9a24d578c7999930"]
evidence_max_seq: 27
observed_at: "2026-11-11T12:18:50.445509+00:00"
evidence: [{"id":"event-0cffb5888a264aebbcf50afd45486275","observed_at":"2026-11-11T12:18:05.047834+00:00","seq":26},{"id":"event-362cb971d0d9491f9a24d578c7999930","observed_at":"2026-11-11T12:18:50.445509+00:00","seq":27}]
---
# Outbound leg — tracked points and motion feedback (frames 00002–00012)

Episode `65e33cbb0a944297a54e00cd0ef23872`; scope `roundtrip-82c2daea3ebf493a825a40d1d3d78685`; scene `composite_drawers_gate_b`. `relative_xy_m` is body-frame (x forward, y left) at the listed base pose; the mission column rotates it by that pose's yaw and adds `local_odometry` (origin = mission start, shared by both legs).

## Surface points

| point | image | base pose (x, y, yaw) | uv | range (m) | relative xy | mission xy |
|---|---|---|---|---|---|---|
| point-001 | frame-00002 | (0, 0, 0) | (330, 240) | 1.4316 | (1.431, −0.040) | (1.431, −0.040) |
| point-007 | frame-00004 | (0.0260, 0.7663, 0.0195) | (515, 200) | 1.5493 | (1.395, −0.674) | (1.434, 0.120) |
| point-008 | frame-00005 | (0.5567, 0.7934, 0.0181) | (352, 120) | 2.7632 | (2.752, −0.244) | (3.313, 0.599) |
| point-008 | frame-00006 | (1.0866, 0.8225, 0.0171) | (366, 118) | 2.2388 | (2.222, −0.276) | (3.313, 0.585) |
| point-009 | frame-00006 | (1.0866, 0.8225, 0.0171) | (585, 150) | 2.6858 | (2.176, −1.575) | (3.289, −0.715) |
| point-010 | frame-00006 | (1.0866, 0.8225, 0.0171) | (393, 157) | 4.5729 | (4.469, −0.971) | (5.571, −0.072) |
| point-011 | frame-00007 | (1.5655, 0.9354, 0.0203) | (392, 200) | 1.7730 | (1.742, −0.331) | (3.314, 0.640) |
| point-012 | frame-00009 | (2.6249, 0.9804, 0.0137) | (500, 170) | 0.8103 | (0.768, −0.259) | (3.396, 0.732) |
| point-013 | frame-00009 | (2.6249, 0.9804, 0.0137) | (440, 160) | 3.1032 | (2.934, −1.010) | (5.573, 0.011) |
| point-014 | frame-00009 | (2.6249, 0.9804, 0.0137) | (600, 180) | 0.7595 | (0.685, −0.328) | (3.314, 0.661) |
| point-015 | frame-00012 | (2.7122, 0.0226, 0.2799) | (165, 180) | 0.9199 | (0.878, +0.276) | (3.479, 0.530) |
| point-016 | frame-00012 | (2.7122, 0.0226, 0.2799) | (437, 140) | 2.8608 | (2.717, −0.895) | (5.571, −0.087) |

Points 002–006 were dropped (`inconsistent_flow`); point-001 is weak (used for the cabinet plane).

Reading:
- point-007 → cabinet front face, mission x ≈ 1.43, y ≈ +0.12; lost on frame-00005 once the robot drove past.
- point-008 = gate **left post**, re-observed 0.53 m apart at the same mission x = 3.313 → **gate plane x ≈ 3.30**.
- point-011 (frame-00007) and point-014 (frame-00009) both land at x = 3.314, y = 0.640 / 0.661: the left upright's surface spans y ≈ +0.59…+0.66 at x ≈ 3.31 — third and fourth independent fixes of that plane.
- point-009 = **right post** (3.289, −0.715); with the left-post samples the opening is y ≈ +0.60…−0.72, centre ≈ **−0.06**.
- point-012 is the left upright's robot-facing side face ≈ (3.40, +0.73): post ~0.13 m thick outside the +0.60 edge; its x is a silhouette-edge depth (~3.30–3.40), not a new plane.
- **point-015** (frame-00012, uv 165/180, picked on the near left upright's dark-green panel) = (3.479, +0.530): the upright's robot-facing material reaches x ≈ 3.48, so x ≈ 3.30 remains the plane the base must cross while the post body lies between ≈ 3.30 and 3.48. Not a new gate plane.
- **point-016** (frame-00012, uv 437/140, plaque B) = (5.571, −0.087). B's x is now fixed three ways: 5.571 (point-010), 5.573 (point-013), 5.571 (point-016) → **B ≈ (5.57, −0.05)**, y scatter ≈ 0.10 m.
- **Points 001–014 are invalid from frame-00010 on** (`optical_flow_lost` / `inconsistent_flow`). Their saved mission coordinates above are unaffected. Points **015 and 016 are the only live points**, both on frame-00012, and both describe frame-00012's pose only. Re-anchor after moving.

## Motion feedback (measured, not requested)

1. seq 5 — `move_relative(lateral +0.6)`: 0.537 m, position error 0.065 m.
2. seq 9 — `lateral +0.3`: 0.229 m, error 0.072 m.
3. seq 11 — `forward 0.6`: 0.5311 m in 3.22 s, error 0.071 m.
4. seq 13 — `forward 0.6`: 0.5304 m in 3.28 s, error 0.072 m.
5. seq 16 — `forward 0.55 + lateral +0.1`: (0.4808, +0.1047) m, error 0.069 m → (1.5655, 0.9354) yaw 0.0203.
6. seq 18 — `forward 0.6`: 0.5301 m, error 0.071 m → (2.0953, 0.9588) yaw 0.0164.
7. seq 19 — `forward 0.6`: 0.5299 m, error 0.0713 m → (2.6249, 0.9804) yaw 0.0137.
8. seq 23 — `lateral −0.6`: start-frame (0.0117, **−0.5267**) m, yaw +0.0322, error 0.074 m → (2.6438, 0.4539) yaw 0.0459.
9. seq 24 — `lateral −0.5`: (0.0239, **−0.4248**) m, yaw −0.0149, error 0.0789 m → (2.6872, 0.0306) yaw 0.0311.
10. seq 25 — `turn +0.3`: **yaw_delta 0.2488 rad** in 2.6 s, translation (0.0247, −0.0088) m, error 0.0262 m → **(2.7122, 0.0226) yaw 0.2799**, idle, upright.
11. seq 26–27 — two `track_point` calls on frame-00012 (point-015 at uv 165/180, point-016 at uv 437/140); no motion, pose unchanged.

Pattern over 10 moves: everything under-delivers — forward 0.530 per 0.60 asked (≈88 %), lateral ≈85–88 %, turns ≈83 %. Heading error ≤ 0.05 rad, position error 0.03–0.08 m, clean stops. Scale requests by ~1.15, verify on odometry, never sum requests.

## Visual notes
- frame-00006: gate posts on the checker floor against a black backdrop; all three plaques legible — **C** leftmost (u ≈ 315), **B** just inside the opening's left edge (u ≈ 393), **A** right of B (u ≈ 470). Left→right in mission y: C (+), B (≈0.0), A (−).
- frame-00009 (sha256 `5f2b202f…cce00231`): the near left upright fills the right ~1/4 (dark green u ≈ 475–530, bright green beyond); C and clipped B visible, A not; camera at mission x ≈ 2.945, short of the gate.
- frame-00012 (sha256 `a6a3a7ab…`): with yaw +0.28 the gate is only a left-edge sliver — bright green u ≈ 0–60, dark green panel u ≈ 60–175 — while **C (u ≈ 315), B (u ≈ 437), A (u ≈ 590)** stand on thin posts over the checker floor. Camera at ≈ (3.02, 0.11); base still 0.59 m short of the plane.