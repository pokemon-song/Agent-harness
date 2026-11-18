---
path: "episodes/outbound_02_crossing_and_approach.md"
revision: 8
scope: "roundtrip-82c2daea3ebf493a825a40d1d3d78685"
status: "confirmed"
evidence_ids: ["event-6f187ce922a540af9857d18de1dd8afc","event-fb29311e322e47679c811cc540a223eb","event-d05e11b82fdf448f8e9449e5715dc75b","event-c870fa1b06be403790b229234b96718d","event-219a3f1f27b0496496ccbe7d193bab60","event-1dac0e0c118a49d29422ebd59239a882","event-5be891b0ea874edfaf016eafc2269951","event-8ddbd7c60a10451eb18f2427aeb45b73"]
evidence_max_seq: 35
observed_at: "2026-11-11T12:20:21.631921+00:00"
evidence: [{"id":"event-6f187ce922a540af9857d18de1dd8afc","observed_at":"2026-11-11T12:19:50.835635+00:00","seq":28},{"id":"event-fb29311e322e47679c811cc540a223eb","observed_at":"2026-11-11T12:19:56.795061+00:00","seq":29},{"id":"event-d05e11b82fdf448f8e9449e5715dc75b","observed_at":"2026-11-11T12:20:01.617932+00:00","seq":30},{"id":"event-c870fa1b06be403790b229234b96718d","observed_at":"2026-11-11T12:20:04.594848+00:00","seq":31},{"id":"event-219a3f1f27b0496496ccbe7d193bab60","observed_at":"2026-11-11T12:20:07.644334+00:00","seq":32},{"id":"event-1dac0e0c118a49d29422ebd59239a882","observed_at":"2026-11-11T12:20:10.869579+00:00","seq":33},{"id":"event-5be891b0ea874edfaf016eafc2269951","observed_at":"2026-11-11T12:20:13.564301+00:00","seq":34},{"id":"event-8ddbd7c60a10451eb18f2427aeb45b73","observed_at":"2026-11-11T12:20:21.631921+00:00","seq":35}]
---
# Outbound leg 2 — gate crossing, plaque B fixes, move calibration (frames 00013–00017)

Same scope/episode/scene as `episodes/outbound_01_measurements.md`. Mission xy below = `local_odometry` pose plus `relative_xy_m` rotated by that pose's yaw.

## Tracked points

| point | frame | uv | pose (x, y, yaw) | range m | rel (x, y) | conf | mission (x, y) |
|---|---|---|---|---|---|---|---|
| point-017 | 00013 | 345.0, 140.0 | 2.7362, 0.0373, 0.0044 | 2.8393 | 2.8329, −0.1911 | 1.00 | 5.570, −0.141 |
| point-017 | 00014 | 354.9, 147.3 | 3.2733, 0.0561, 0.0080 | 2.3052 | 2.2948, −0.2183 | 0.21 | 5.570, −0.144 |
| point-017 | 00015 | 355.4, 136.4 | 3.7570, −0.0052, 0.0168 | 1.8178 | 1.8104, −0.1639 | 0.37 | 5.570, −0.139 |
| point-018 | 00016 | 345.0, 175.0 | 4.2382, −0.0782, 0.0385 | 1.3376 | 1.3350, −0.0829 | 1.00 | 5.576, −0.110 |

All four follow the same white plaque **B** across 1.8 m of base travel; x agrees to 7 mm → B x = 5.57. y clusters −0.11…−0.14 here, while the pre-turn fixes point-010/013/016 gave −0.087…+0.011 → plaque centre ≈ (5.57, −0.10), y uncertain ±0.10. Every point (011–018) was invalid on frame-00017 (`optical_flow_lost` / `inconsistent_flow`), so nothing is live at pose (4.4448, −0.0907, 0.0523).

## Motion feedback seq 28–35
- **seq 28** `turn` −0.34 → measured −0.2755 rad (81 %), heading error 0.065, translation (0.0272, 0.0075), pos error 0.028 → (2.7362, 0.0373) yaw 0.0044.
- **seq 29** `track_point` frame-00013 (345,140) → point-017; no motion.
- **seq 30** `move_relative` forward 0.60 → 0.5372 m (90 %), yaw Δ 0.0036, pos error 0.065 → (3.2733, 0.0561) yaw 0.0080.
- **seq 31** forward 0.596 / lateral −0.07 → **failed `INVALID_ARGUMENTS` "Translation length exceeds 0.6 m"** (norm 0.6001); no motion. Keep ‖(f, l)‖ ≤ 0.6.
- **seq 32** forward 0.55 / lateral −0.09 → (0.4832, −0.0652) = 88 % / 72 %, pos error 0.071 → (3.7570, −0.0052) yaw 0.0168.
- **seq 33** forward 0.55 / lateral −0.10 → (0.4799, −0.0811) = 87 % / 81 %, pos error 0.073 → (4.2382, −0.0782) yaw 0.0385.
- **seq 34** `track_point` frame-00016 (345,175) → point-018; no motion.
- **seq 35** forward 0.28 / lateral −0.05 → (0.2059, −0.0205) = 74 % / 41 %, pos error 0.080 → (4.4448, −0.0907) yaw 0.0523.

Pattern over 14 successful moves: forward 0.55–0.60 m lands at 87–90 %, but 0.28 m lands at 74 %; laterals 0.09–0.10 m at 72–81 %, 0.05 m at 41 %; turns ≈ 81 %. Short requests are under-delivered disproportionately — plan ≥ 0.25–0.3 m per axis or expect repeats, and always read back odometry.

## Gate crossing (measured, not requested)
Seq 32 ran from (3.2733, 0.0561) to (3.7570, −0.0052), so the base crossed the frame plane x ≈ 3.30 at y ≈ **+0.05** (interpolated), ≈ 0.55 m inside the left inner edge and ≈ 0.77 m from the right post. By the end of seq 35 x = 4.4448 → **1.145 m past the plane** (≥ 0.8 m required) on a ≈ +x heading. No frame is visible in frame-00017, which is consistent with that but is not the proof; the odometry is.

## Visual note (frame-00017, current image)
Plaque **B** fills the image centre (u ≈ 275–430, v ≈ 10–190) on a thin grey post over a grey checker floor. At u ≤ 60 on the left edge only the right portion of another white plaque with black bars is visible — at larger mission y than B, consistent with C (identity not certain). No gate, cabinet or walls in view; background black.
