---
path: "tasks/outbound_drawers_gate_B.md"
revision: 6
scope: "roundtrip-82c2daea3ebf493a825a40d1d3d78685"
status: "confirmed"
evidence_ids: ["event-95e6340cbf424365af4238e5cde6c63a","event-492fcacf6c1e4a05aa6dd78db1c124f0","event-81a3e2976f3a42e9bebc9360a7be43c0","event-4ca83b791abd4ae59fd43a807053ba90","event-7b440b7d2ea14c81b6356c49833199c8","event-d297db4b3eac4131afc779c99b5c4f6d","event-2f71c67d0f4242619e02f008d9442ba1","event-be9853f75ce2410b871ec57563b6cf7c","event-2ef2575ead584f4dbb76425a229ba174","event-13f58dcaac764b238bf7b1d9cc536dbe"]
evidence_max_seq: 25
observed_at: "2026-11-11T12:18:01.192904+00:00"
evidence: [{"id":"event-95e6340cbf424365af4238e5cde6c63a","observed_at":"2026-11-11T12:14:08.424535+00:00","seq":16},{"id":"event-492fcacf6c1e4a05aa6dd78db1c124f0","observed_at":"2026-11-11T12:14:20.486058+00:00","seq":17},{"id":"event-81a3e2976f3a42e9bebc9360a7be43c0","observed_at":"2026-11-11T12:14:44.836966+00:00","seq":18},{"id":"event-4ca83b791abd4ae59fd43a807053ba90","observed_at":"2026-11-11T12:15:35.035162+00:00","seq":19},{"id":"event-7b440b7d2ea14c81b6356c49833199c8","observed_at":"2026-11-11T12:16:19.138130+00:00","seq":20},{"id":"event-d297db4b3eac4131afc779c99b5c4f6d","observed_at":"2026-11-11T12:16:38.200404+00:00","seq":21},{"id":"event-2f71c67d0f4242619e02f008d9442ba1","observed_at":"2026-11-11T12:17:23.010915+00:00","seq":22},{"id":"event-be9853f75ce2410b871ec57563b6cf7c","observed_at":"2026-11-11T12:17:36.485447+00:00","seq":23},{"id":"event-2ef2575ead584f4dbb76425a229ba174","observed_at":"2026-11-11T12:17:47.794616+00:00","seq":24},{"id":"event-13f58dcaac764b238bf7b1d9cc536dbe","observed_at":"2026-11-11T12:18:01.192904+00:00","seq":25}]
---
# Outbound leg: drawers → green gate → letter B

Scope `roundtrip-82c2daea3ebf493a825a40d1d3d78685`; scene `composite_drawers_gate_b`; episode `65e33cbb0a944297a54e00cd0ef23872`.

## Registered instruction (public announcement, seq 1, 2026-11-11T12:07:44Z)
Stages ordered (`stages_ordered = true`):
1. `avoid_drawers` — go around the front drawer cabinet so the **base origin** passes behind it with body clearance.
2. `pass_gate` — then pass the base origin through the **green** frame's opening (inside it while crossing the frame plane) and continue at least **0.8 m** past it in the crossing direction.
3. `approach_b` — among white plaques A, B, C behind the gate choose **B**, move to ~**1.0 m** from B's plaque center, face B, stop, then call `finish`.

Limits: no contact with cabinet, frame, plaques or any scene object; stages in order; skipping the gate straight to B is forbidden. Finish tolerances: distance 0.18 m, heading 0.25 rad, speed ≤ 0.12 m/s, yaw rate ≤ 0.25 rad/s; reference body = base.
The front camera is 0.32 m ahead of the base, so the gate leaving the image does not prove the base crossed it; confirm crossing and the margin from `local_odometry` plus measured feedback, never from requested motion.

After `finish` only the outbound leg ends: the planner's conversation, notes and tracked points are cleared, but robot, scene, sim time, pose and `local_odometry` are not reset. The return leg is planned from this memory plus fresh perception.

## Coordinates and frames
- coordinate_frame_id `roundtrip-82c2daea3ebf493a825a40d1d3d78685`; reference "continuous_mission_start; physical world and odometry origin are preserved".
- `local_odometry`: origin/axes fixed at the mission's initial base pose, preserved across planner-context resets, shared by both legs.
- `tracked_points.relative_xy_m`: origin at the current base, x forward, y left, metres; yaw positive left. Body axes x forward, y left, z up; camera `front_rgbd_optical` at +0.32 m forward, +0.08 m up, optical x right / y down / z forward.

## Scene geometry (mission frame, x forward, y left)
- **Cabinet:** front plane x ≈ 1.43 (point-001 y ≈ −0.04, point-007 y ≈ +0.12). Depth and width unmeasured.
- **Green gate:** frame plane x ≈ **3.30**, fixed four ways — point-008 twice (x = 3.313, y 0.585/0.599), point-011 (3.314, 0.640), point-014 (3.314, 0.661). The left upright's surface therefore spans y ≈ +0.59…+0.66 at x ≈ 3.31; right post point-009 at (3.289, −0.715). Sampled opening y ≈ +0.60…−0.72, centre ≈ **y = −0.06**, sampled width ~1.3 m (clear inner width less). point-012 (3.40, +0.73) is the near left upright's robot-facing side face → the post is ~0.13 m thick outside the +0.60 edge; its x is a silhouette-edge depth (~3.30–3.40), not a new plane.
- **Plaque B:** ≈ **(5.57, −0.03)** — point-010 (5.571, −0.072) and point-013 (5.573, +0.011) agree to 2 mm in x and 8 cm in y, so B sits behind the opening centre. C is leftmost in the image (larger y), A right of B (smaller y); neither tracked → coordinates unknown.

## Route status (through frame-00012, sim_time 38.56 s)
- **Pose:** `local_odometry` (2.7122, 0.0226) m, **yaw 0.2799 rad** (left of mission +x); idle, upright, height 0.358 m. Camera at ≈ (3.02, 0.11), still 0.28 m short of the gate plane.
- **Executed, 10 successful moves:** lateral +0.6, +0.3; forward 0.6 ×4; diagonal forward 0.55 / lateral +0.1; lateral −0.6 (**0.527 m**); lateral −0.5 (**0.425 m**); turn +0.3 (**0.2488 rad**). Every axis under-delivers by ~12–17 %: forward ≈ 0.530 m per 0.60 asked, lateral 85–88 %, turns ~83 %. Heading error ≤ 0.05 rad, position error 0.03–0.08 m, clean stops. Scale requests by ~1.15 and re-check odometry; never sum requests.
- **Stage `avoid_drawers`: likely satisfied (not directly re-confirmed).** The base advanced from x ≈ 1.1 to 2.62 m while 0.86–0.98 m **left** of the cabinet front-face points (y ≤ +0.12), so it passed behind the body with lateral clearance; no move reported contact. Cabinet rear extent still unmeasured, so treat the margin beyond the front plane as unverified.
- **Stage `pass_gate`: laterally aligned, not yet crossed.** Base y = 0.023 m is ~0.08 m from the opening centre (y ≈ −0.06) and ~0.57 m inside the left inner edge → alignment done. Base x = 2.712 m is still **0.59 m short** of x ≈ 3.30. Heading is 0.28 rad left of the opening axis, so running straight on it would cross the plane near **(3.30, +0.20)** — inside the opening but drifting toward the left post and away from B. Recommendation: turn right ~0.3 rad (request ≈ 0.34 given 83 % turn delivery) to face +x again, then drive forward until measured travel ≥ 1.39 m (0.59 m to the plane + 0.8 m margin → x ≥ 4.10); split into two legs and verify on odometry.
- **Stage `approach_b`: not started.** B is ≈ 2.86 m ahead/right; after realignment a 1.0 m standoff is predicted near **(4.57, −0.03) facing +x**, which also satisfies x ≥ 4.10.
- **Live tracked points: none.** Points 001–014 are all invalid from frame-00010 on (`optical_flow_lost` / `inconsistent_flow`), including 012–014 picked on frame-00009. Use the saved mission coordinates above (gate plane, opening edges, B) plus odometry and re-anchor with fresh `track_point` calls on the next usable frame.

**Return-leg note:** from an ideal finish at ≈(4.57, −0.03) yaw ≈ 0, turn ~180°, run back −x through the same opening (cross x ≈ 3.30 near y ≈ −0.06), then shift left (+y) to ≈ +0.9 m to clear the cabinet on its left side before the cabinet front plane x ≈ 1.43, finishing near (0, 0) with yaw ≈ π. Cabinet rear extent unknown — re-measure on the way back.

Detailed point table and per-move feedback: `episodes/outbound_01_measurements.md`.