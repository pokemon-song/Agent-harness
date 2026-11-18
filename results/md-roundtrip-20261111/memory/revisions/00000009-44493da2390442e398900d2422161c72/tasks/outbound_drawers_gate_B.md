---
path: "tasks/outbound_drawers_gate_B.md"
revision: 9
scope: "roundtrip-82c2daea3ebf493a825a40d1d3d78685"
status: "confirmed"
evidence_ids: ["event-31a406a1ef9d4b99864b678c1621555b","event-7005a7cc2003473b8a6f01647fd11116","event-d4ac6bca96134320b4105a992426de74","event-bf3eda93cc924dbf975306bd1d41f8d2"]
evidence_max_seq: 39
observed_at: "2026-11-11T12:20:57.077475+00:00"
evidence: [{"id":"event-31a406a1ef9d4b99864b678c1621555b","observed_at":"2026-11-11T12:20:29.461473+00:00","seq":36},{"id":"event-7005a7cc2003473b8a6f01647fd11116","observed_at":"2026-11-11T12:20:35.632132+00:00","seq":37},{"id":"event-d4ac6bca96134320b4105a992426de74","observed_at":"2026-11-11T12:20:55.978438+00:00","seq":38},{"id":"event-bf3eda93cc924dbf975306bd1d41f8d2","observed_at":"2026-11-11T12:20:57.077475+00:00","seq":39}]
---
# Outbound leg: drawers → green gate → letter B

Scope `roundtrip-82c2daea3ebf493a825a40d1d3d78685`; scene `composite_drawers_gate_b`; episode `65e33cbb0a944297a54e00cd0ef23872`.

## Registered instruction (public announcement, seq 1, 2026-11-11T12:07:44Z)
Stages ordered (`stages_ordered = true`):
1. `avoid_drawers` — go around the front drawer cabinet so the **base origin** passes behind it with body clearance.
2. `pass_gate` — pass the base origin through the **green** frame's opening (inside it while crossing the frame plane), then continue at least **0.8 m** past it in the crossing direction.
3. `approach_b` — among white plaques A, B, C behind the gate choose **B**, move to ~**1.0 m** from B's plaque center, face B, stop, then call `finish`.

Limits: no contact with cabinet, frame, plaques or other scene objects; stages in order; skipping the gate straight to B is forbidden. Finish tolerances: distance 0.18 m, heading 0.25 rad, stopped speed ≤ 0.12 m/s, yaw rate ≤ 0.25 rad/s; reference body = base.
The front camera is 0.32 m ahead of the base, so the frame leaving the image does not prove crossing; confirm crossing and margin from `local_odometry` plus measured feedback, never from requested motion.

`finish` ends only the outbound leg: planner context, notes and tracked points are cleared; robot, scene, sim time, pose and `local_odometry` are NOT reset. The return leg is planned from this memory plus fresh perception.

## Coordinates and frames
- coordinate_frame_id `roundtrip-82c2daea3ebf493a825a40d1d3d78685`; "continuous_mission_start; physical world and odometry origin are preserved".
- `local_odometry`: origin/axes fixed at the mission's initial base pose, shared by both legs.
- `tracked_points.relative_xy_m`: origin at the current base, x forward, y left, metres; yaw positive left. Camera `front_rgbd_optical` at +0.32 m forward, +0.08 m up; optical x right / y down / z forward.

## Scene geometry (mission frame, x forward, y left)
- **Cabinet:** front plane x ≈ 1.43 (point-001 y ≈ −0.04, point-007 y ≈ +0.12). Depth and width still unmeasured.
- **Green gate:** frame plane x ≈ **3.30** (point-008/011/014: x = 3.313–3.314). Opening at that plane: left inner surface y ≈ +0.60, right post sample y ≈ −0.72, centre ≈ y −0.06, width ~1.3 m. Solid post material reaches x ≈ 3.48 near y ≈ +0.5…+0.7 (point-012/015) — cross at x ≈ 3.30 and keep y well below +0.5.
- **Measured crossing point ≈ (3.30, +0.05)** (interpolated from seq 32 odometry): ~0.55 m inside the left inner edge, ~0.77 m from the right post sample.
- **Plaque B ≈ (5.571, −0.10)**. x fixed to ±0.006 m by five fixes (point-017 on frames 00013/14/15 → 5.570 ×3; point-018 frame-00016 → 5.576; point-019 frame-00019 back-solved → 5.5715). y fixes scatter −0.144 … −0.056; treat y as **−0.10 ± 0.06**. C is left of B (larger y), A right of B (smaller y); neither tracked → coordinates unknown.

## Outbound outcome — `finish` declared (seq 39, sim 57.28 s)
Final pose `local_odometry` **(4.5662, −0.0922) m, yaw 0.0589 rad**; idle, upright, no contact reported; stopped speed 0.0049 m/s, yaw rate 0.0043 rad/s (both inside limits).
- **`avoid_drawers`: satisfied on available evidence.** Base ran x ≈ 0.03 → 4.57 m at y ≈ +0.77…+0.98 through the bypass and then y ≈ 0.0 at the far end; the lane stayed 0.86–0.98 m left of the cabinet front-face points (y ≤ +0.12) and 0.65–0.80 m from the front plane. Rear extent of the body behind the cabinet is still unverified by measurement, but no move reported contact.
- **`pass_gate`: satisfied — measured.** Base crossed x ≈ 3.30 at ≈ (3.30, +0.05) during seq 32; distance past the plane at finish = 4.5662 − 3.30 = **1.266 m ≥ 0.8 m**, along ≈ +x (yaw 0.05).
- **`approach_b`: satisfied on the measured state immediately before `finish`.** B range **1.0059 m** (tolerance ±0.18, i.e. 0.93 m margin), body-frame bearing **−0.023 rad** (tolerance 0.25), robot stopped. `finish` returned "Declaration recorded; evaluator runs separately" — the task-level verdict is not known here.

## Motion-feedback lessons (15 successful moves)
- `move_relative` rejects combined translation with norm > 0.6 m: seq 31 (forward 0.596, lateral −0.07) → `INVALID_ARGUMENTS` "Translation length exceeds 0.6 m", no motion. Keep ‖(forward, lateral)‖ ≤ 0.6.
- Delivery ratio, measured vs requested: forward 0.55–0.60 → 87–90 %; forward 0.28 → 74 %; forward 0.18 → 68 % (0.1225 m, seq 37); lateral 0.09–0.10 → 72–81 %; lateral 0.05 → 41 %; lateral 0.04 → 30 % (0.0118 m, seq 37); turn −0.34 rad → 81 %. Short requests under-deliver badly → use steps ≥ 0.25–0.30 m per axis, scale requests ~1.15, verify every move on odometry.
- Heading error ≤ 0.05 rad per step; position error 0.065–0.080 m.

## Return leg (predicted — replan with fresh perception)
Start exactly at the finish pose above, yaw ≈ 0.0589 (facing +x); tracked points are cleared.
1. Turn in verified increments to yaw ≈ π. +π (left) and −π (right) are equivalent; prefer several ≤ 0.34 rad steps and check odometry each time.
2. Run −x at y ≈ −0.09 from x ≈ 4.57 to ≈ 2.7 — this is the proven outbound lane, so no new clearance question. Gate crossing at y ≈ −0.09 leaves ≈ 0.69 m to the left inner surface (+0.60) and ≈ 0.63 m to the right post sample (−0.72).
3. At x ≈ 2.6 shift to y ≈ +0.90…0.95 (mission +y is the robot's right when facing −x), then continue −x past the cabinet front plane x ≈ 1.43; that lane is 0.65–0.80 m clear of the front face and was travelled outbound without contact.
4. Near x ≈ 0.6 step down to y ≈ +0.8, then to ≈ (0, 0) ending yaw ≈ π.
Cabinet rear extent is still unmeasured — re-measure behind it on the way back rather than assuming.

Per-move and per-point detail: `episodes/outbound_01_measurements.md` (frames 00002–00012), `episodes/outbound_02_crossing_and_approach.md` (frames 00013–00017), `episodes/outbound_03_finish_return_start.md` (frames 00017–00019).