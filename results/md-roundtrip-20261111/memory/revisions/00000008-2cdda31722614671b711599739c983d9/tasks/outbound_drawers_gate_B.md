---
path: "tasks/outbound_drawers_gate_B.md"
revision: 8
scope: "roundtrip-82c2daea3ebf493a825a40d1d3d78685"
status: "confirmed"
evidence_ids: ["event-6cf0048c2b8346fab57fc80bfedc1e7c","event-6f187ce922a540af9857d18de1dd8afc","event-fb29311e322e47679c811cc540a223eb","event-d05e11b82fdf448f8e9449e5715dc75b","event-c870fa1b06be403790b229234b96718d","event-219a3f1f27b0496496ccbe7d193bab60","event-1dac0e0c118a49d29422ebd59239a882","event-5be891b0ea874edfaf016eafc2269951","event-8ddbd7c60a10451eb18f2427aeb45b73"]
evidence_max_seq: 35
observed_at: "2026-11-11T12:20:21.631921+00:00"
evidence: [{"id":"event-6cf0048c2b8346fab57fc80bfedc1e7c","observed_at":"2026-11-11T12:07:44.826444+00:00","seq":1},{"id":"event-6f187ce922a540af9857d18de1dd8afc","observed_at":"2026-11-11T12:19:50.835635+00:00","seq":28},{"id":"event-fb29311e322e47679c811cc540a223eb","observed_at":"2026-11-11T12:19:56.795061+00:00","seq":29},{"id":"event-d05e11b82fdf448f8e9449e5715dc75b","observed_at":"2026-11-11T12:20:01.617932+00:00","seq":30},{"id":"event-c870fa1b06be403790b229234b96718d","observed_at":"2026-11-11T12:20:04.594848+00:00","seq":31},{"id":"event-219a3f1f27b0496496ccbe7d193bab60","observed_at":"2026-11-11T12:20:07.644334+00:00","seq":32},{"id":"event-1dac0e0c118a49d29422ebd59239a882","observed_at":"2026-11-11T12:20:10.869579+00:00","seq":33},{"id":"event-5be891b0ea874edfaf016eafc2269951","observed_at":"2026-11-11T12:20:13.564301+00:00","seq":34},{"id":"event-8ddbd7c60a10451eb18f2427aeb45b73","observed_at":"2026-11-11T12:20:21.631921+00:00","seq":35}]
---
# Outbound leg: drawers → green gate → letter B

Scope `roundtrip-82c2daea3ebf493a825a40d1d3d78685`; scene `composite_drawers_gate_b`; episode `65e33cbb0a944297a54e00cd0ef23872`.

## Registered instruction (public announcement, seq 1, 2026-11-11T12:07:44Z)
Stages ordered (`stages_ordered = true`):
1. `avoid_drawers` — go around the front drawer cabinet so the **base origin** passes behind it with body clearance.
2. `pass_gate` — pass the base origin through the **green** frame's opening (inside it while crossing the frame plane), then continue at least **0.8 m** past it in the crossing direction.
3. `approach_b` — among white plaques A, B, C behind the gate choose **B**, move to ~**1.0 m** from B's plaque center, face B, stop, then call `finish`.

Limits: no contact with cabinet, frame, plaques or other scene objects; stages in order; skipping the gate straight to B is forbidden. Finish tolerances: distance 0.18 m, heading 0.25 rad, stopped speed ≤ 0.12 m/s, yaw rate ≤ 0.25 rad/s; reference body = base.
The front camera is 0.32 m ahead of the base, so the frame leaving the image does not prove crossing; confirm the crossing and its margin from `local_odometry` plus measured feedback, never from requested motion.

After `finish` only the outbound leg ends: planner context, notes and tracked points are cleared; robot, scene, sim time, pose and `local_odometry` are not reset. The return leg is planned from this memory plus fresh perception.

## Coordinates and frames
- coordinate_frame_id `roundtrip-82c2daea3ebf493a825a40d1d3d78685`; reference "continuous_mission_start; physical world and odometry origin are preserved".
- `local_odometry`: origin/axes fixed at the mission's initial base pose, shared by both legs.
- `tracked_points.relative_xy_m`: origin at the current base, x forward, y left, metres; yaw positive left. Body axes x forward, y left, z up; camera `front_rgbd_optical` at +0.32 m forward, +0.08 m up, optical x right / y down / z forward.

## Scene geometry (mission frame, x forward, y left)
- **Cabinet:** front plane x ≈ 1.43 (point-001 y ≈ −0.04, point-007 y ≈ +0.12). Depth and width unmeasured.
- **Green gate:** frame plane x ≈ **3.30** (point-008/011/014, x = 3.313–3.314). Sampled opening y ≈ +0.60 (left upright surface) to −0.72 (right post, point-009), centre ≈ y −0.06, width ~1.3 m. Solid post material reaches x ≈ 3.48 near y ≈ +0.5…+0.7 (point-012, point-015); cross at x ≈ 3.30 and keep y well below +0.5.
- **Measured crossing point ≈ (3.30, +0.05)** — interpolated from seq 32 odometry; ~0.55 m inside the left inner edge, ~0.77 m from the right post sample.
- **Plaque B ≈ (5.57, −0.10).** x fixed to ±0.006 m by four fixes: point-017 on frames 00013/14/15 → 5.570, 5.570, 5.570, and point-018 (frame-00016) → 5.576. Their y = −0.141, −0.144, −0.139, −0.110; earlier pre-turn fixes (point-010/013/016) gave y = −0.087…+0.011 → treat y as −0.10 ± 0.10. C is left of B (larger y), A right of B (smaller y); neither is tracked → coordinates unknown.

## Route status (through frame-00017, sim_time 54.2 s)
- **Pose:** `local_odometry` (4.4448, −0.0907) m, **yaw 0.0523 rad**; idle, upright, no contact reported. Camera at ≈ (4.76, −0.07).
- **`avoid_drawers`: satisfied on available evidence.** The base advanced x ≈ 1.1 → 4.44 m while 0.86–0.98 m left of the cabinet front-face points (y ≤ +0.12); cabinet rear extent still unmeasured, so the margin behind the body is unverified, but no move reported contact.
- **`pass_gate`: satisfied — measured, not requested.** Base crossed x ≈ 3.30 at ≈ (3.30, +0.05) during seq 32, inside the opening. Travel past the plane from odometry = 4.4448 − 3.30 = **1.145 m ≥ 0.8 m**, along ≈ +x (yaw 0.05). Frame is no longer in view (frame-00017), which alone would prove nothing.
- **`approach_b`: close to tolerance, needs one verification.** B is 1.125 m ahead and ~0.01–0.04 m right in mission frame → range ≈ **1.13 m** (inside the 0.82–1.18 m band, only ~0.05 m margin) and body-frame bearing ≈ **−0.09 rad** (inside 0.25 rad). Every point was lost on frame-00017, so re-anchor with a fresh `track_point` on the current image (B's plaque fills u ≈ 275–430, v ≈ 10–190) and confirm with measured geometry before `finish`. A short measured forward step (~0.10–0.15 m) would sit near the 1.0 m standoff with margin on both sides.

## Motion-feedback lessons (14 successful moves)
- `move_relative` rejects a combined translation with norm > 0.6 m: seq 31 (forward 0.596, lateral −0.07) failed `INVALID_ARGUMENTS` "Translation length exceeds 0.6 m" and did not move. Keep ‖(forward, lateral)‖ ≤ 0.6.
- Delivery: forward 0.55–0.60 m → 87–90 % (0.48–0.54 m); forward 0.28 m → 74 %; lateral 0.09–0.10 m → 72–81 %; lateral 0.05 m → 41 %; turn −0.34 rad → 0.2755 rad (81 %). Short requests under-deliver disproportionately → use steps ≥ 0.25–0.3 m per axis, scale requests ~1.15, verify each move on odometry.
- Heading error ≤ 0.05 rad per step; position error 0.065–0.080 m.

## Return leg (from this memory + fresh perception)
Start near (4.45, −0.09) yaw ≈ 0. Turn in verified increments to ≈ π, then run −x at y ≈ −0.09 through the gate opening (left inner surface y +0.60 → ≈ 0.69 m clearance; right post sample −0.72 → ≈ 0.63 m); at x ≈ 2.6 shift to y ≈ +0.9 to stay behind the cabinet front plane x ≈ 1.43 and beyond its front-face points (y ≤ +0.12), continue −x past it, then to ≈ (0, 0) ending yaw ≈ π. Cabinet rear extent unknown — re-measure on the way back.

Per-move and per-point detail: `episodes/outbound_01_measurements.md` (frames 00002–00012) and `episodes/outbound_02_crossing_and_approach.md` (frames 00013–00017).