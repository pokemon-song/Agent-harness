---
path: "episodes/outbound_01_measurements.md"
revision: 6
scope: "roundtrip-82c2daea3ebf493a825a40d1d3d78685"
status: "confirmed"
evidence_ids: ["event-95e6340cbf424365af4238e5cde6c63a","event-492fcacf6c1e4a05aa6dd78db1c124f0","event-81a3e2976f3a42e9bebc9360a7be43c0","event-4ca83b791abd4ae59fd43a807053ba90","event-7b440b7d2ea14c81b6356c49833199c8","event-d297db4b3eac4131afc779c99b5c4f6d","event-2f71c67d0f4242619e02f008d9442ba1","event-be9853f75ce2410b871ec57563b6cf7c","event-2ef2575ead584f4dbb76425a229ba174","event-13f58dcaac764b238bf7b1d9cc536dbe"]
evidence_max_seq: 25
observed_at: "2026-11-11T12:18:01.192904+00:00"
evidence: [{"id":"event-95e6340cbf424365af4238e5cde6c63a","observed_at":"2026-11-11T12:14:08.424535+00:00","seq":16},{"id":"event-492fcacf6c1e4a05aa6dd78db1c124f0","observed_at":"2026-11-11T12:14:20.486058+00:00","seq":17},{"id":"event-81a3e2976f3a42e9bebc9360a7be43c0","observed_at":"2026-11-11T12:14:44.836966+00:00","seq":18},{"id":"event-4ca83b791abd4ae59fd43a807053ba90","observed_at":"2026-11-11T12:15:35.035162+00:00","seq":19},{"id":"event-7b440b7d2ea14c81b6356c49833199c8","observed_at":"2026-11-11T12:16:19.138130+00:00","seq":20},{"id":"event-d297db4b3eac4131afc779c99b5c4f6d","observed_at":"2026-11-11T12:16:38.200404+00:00","seq":21},{"id":"event-2f71c67d0f4242619e02f008d9442ba1","observed_at":"2026-11-11T12:17:23.010915+00:00","seq":22},{"id":"event-be9853f75ce2410b871ec57563b6cf7c","observed_at":"2026-11-11T12:17:36.485447+00:00","seq":23},{"id":"event-2ef2575ead584f4dbb76425a229ba174","observed_at":"2026-11-11T12:17:47.794616+00:00","seq":24},{"id":"event-13f58dcaac764b238bf7b1d9cc536dbe","observed_at":"2026-11-11T12:18:01.192904+00:00","seq":25}]
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

Points 002–006 (frames 00002/00003) were flagged `inconsistent_flow` and dropped; point-001 is kept only because its value was used for the cabinet plane, so treat it as weak.

Reading:
- point-007 → cabinet front face, mission x ≈ 1.43, y ≈ +0.12; lost on frame-00005 once the robot drove past.
- point-008 = gate **left post**, re-observed 0.53 m apart at the same mission x = 3.313 → **gate plane x ≈ 3.30**.
- point-011 (frame-00007) and new **point-014** (frame-00009) both land at x = 3.314, y = 0.640 / 0.661: the left upright's surface spans y ≈ +0.59…+0.66 at x ≈ 3.31 — third and fourth independent fixes of that plane.
- point-009 = **right post** (3.289, −0.715); with the left-post samples the opening is y ≈ +0.60…−0.72, centre ≈ **−0.06**.
- point-012 is the left upright's robot-facing side face ≈ (3.40, +0.73): post ~0.13 m thick outside the +0.60 edge; its x is a silhouette-edge depth (~3.30–3.40), not a new plane.
- point-010 (5.571, −0.072) and point-013 (5.573, +0.011), plaque B seen at 4.57 m and at 3.10 m, agree to 2 mm in x → **B ≈ (5.57, −0.03 ± 0.08)**.
- **All 14 points are invalid from frame-00010 on** (`optical_flow_lost` / `inconsistent_flow`), including 012–014. The gate and plaque positions above are saved mission coordinates; re-acquire points on fresh frames.

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

Pattern over 10 executed moves: everything under-delivers — forward 0.530 per 0.60 asked (≈88 %), lateral 0.527/0.60 and 0.425/0.50 (≈85–88 %), turn 0.249/0.30 (≈83 %). Heading error ≤ 0.05 rad, position error 0.03–0.08 m, clean stops. Scale requests by ~1.15, verify on odometry, never sum requests.

## Visual notes
- frame-00006: gate posts on the checker floor against a black backdrop; all three plaques legible — **C** leftmost (u ≈ 315), **B** just inside the opening's left edge (u ≈ 393), **A** right of B (u ≈ 470). Left→right in mission y: C (+), B (≈0.0), A (−).
- frame-00009 (sha256 `5f2b202f…cce00231`): the near left upright fills the right ~1/4 (dark green u ≈ 475–530, bright green beyond); C and clipped B visible, A not; camera at mission x ≈ 2.945, short of the gate.
- frame-00012 (sha256 `a6a3a7ab…`, saved with seq 25): with yaw +0.28 the gate is only a left-edge sliver — bright green u ≈ 0–60, dark green panel u ≈ 60–175 — while **C (u ≈ 315), B (u ≈ 440) and A (u ≈ 590) are all legible**, each on its own thin floor-standing post. Camera at ≈ (3.02, 0.11); base still 0.59 m short of the plane.
- Every saved image belongs to its named event, not to the present robot state.