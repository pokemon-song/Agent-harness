---
path: "tasks/outbound_drawers_gate_B.md"
revision: 5
scope: "roundtrip-82c2daea3ebf493a825a40d1d3d78685"
status: "confirmed"
evidence_ids: ["event-95e6340cbf424365af4238e5cde6c63a","event-492fcacf6c1e4a05aa6dd78db1c124f0","event-81a3e2976f3a42e9bebc9360a7be43c0","event-4ca83b791abd4ae59fd43a807053ba90","event-7b440b7d2ea14c81b6356c49833199c8","event-d297db4b3eac4131afc779c99b5c4f6d"]
evidence_max_seq: 21
observed_at: "2026-11-11T12:16:38.200404+00:00"
evidence: [{"id":"event-95e6340cbf424365af4238e5cde6c63a","observed_at":"2026-11-11T12:14:08.424535+00:00","seq":16},{"id":"event-492fcacf6c1e4a05aa6dd78db1c124f0","observed_at":"2026-11-11T12:14:20.486058+00:00","seq":17},{"id":"event-81a3e2976f3a42e9bebc9360a7be43c0","observed_at":"2026-11-11T12:14:44.836966+00:00","seq":18},{"id":"event-4ca83b791abd4ae59fd43a807053ba90","observed_at":"2026-11-11T12:15:35.035162+00:00","seq":19},{"id":"event-7b440b7d2ea14c81b6356c49833199c8","observed_at":"2026-11-11T12:16:19.138130+00:00","seq":20},{"id":"event-d297db4b3eac4131afc779c99b5c4f6d","observed_at":"2026-11-11T12:16:38.200404+00:00","seq":21}]
---
# Outbound leg: drawers → green gate → letter B

Scope `roundtrip-82c2daea3ebf493a825a40d1d3d78685`; scene `composite_drawers_gate_b`; episode `65e33cbb0a944297a54e00cd0ef23872`.

## Registered instruction (public announcement, seq 1, 2026-11-11T12:07:44Z)
Ordered stages (stages_ordered = true):

1. `avoid_drawers` — go around the front drawer cabinet so the **base origin** passes behind it with body clearance.
2. `pass_gate` — then pass the base origin through the **green** rectangular frame's opening (inside the opening while crossing the frame plane) and continue at least **0.8 m** past it in the crossing direction.
3. `approach_b` — among white plaques A, B, C behind the gate choose **B**, move to ~**1.0 m** from B's plaque center, face B, stop, then call `finish`.

Limits: no contact with cabinet, frame, plaques or any scene object. Stages in order; skipping the gate and heading straight for B is forbidden. Finish tolerances: distance 0.18 m, heading 0.25 rad, stopped speed ≤ 0.12 m/s, stopped yaw rate ≤ 0.25 rad/s, reference body = base.

Instruction note: the front camera is 0.32 m ahead of the base origin, so the gate leaving the image does not prove the base passed it. Track frame surface points while visible and confirm crossing plus the 0.8 m margin from local_odometry and actual motion feedback — never substitute requested motion for measured displacement.

After `finish` only the outbound leg ends: the planner's conversation, notes and tracked points are cleared, but the robot, scene, sim time, pose and local_odometry do **not** reset. The return leg is planned from this memory plus fresh perception, retracing the outbound route.

## Coordinates and frames
- coordinate_frame_id `roundtrip-82c2daea3ebf493a825a40d1d3d78685`; coordinate_reference "continuous_mission_start; physical world and odometry origin are preserved".
- `local_odometry`: origin and horizontal axes fixed at the mission's initial base pose, preserved across planner-context resets (shared by outbound and return legs).
- `tracked_points.relative_xy_m`: origin at the current base body reference, x forward, y left, metres; yaw positive left.
- Body axes x forward, y left, z up; camera front_rgbd_optical at +0.32 m forward, +0.08 m up in body, optical x right / y down / z forward.

## Scene geometry in the mission frame (x forward, y left)
- **Cabinet:** front plane x ≈ 1.43 m (point-001 at y ≈ −0.04, point-007 at y ≈ +0.12). Depth and full width unmeasured; not imaged since frame-00004.
- **Green gate:** frame plane x ≈ **3.30 m**, confirmed three ways — point-008 twice (x = 3.313, y 0.585–0.599) and point-011 (3.314, 0.640). Right post point-009 at (3.289, −0.715) → opening between roughly y = +0.60 and y = −0.72, centre ≈ **y = −0.06**; sampled width ~1.3 m, clear inner width less. point-012 (frame-00009) is the near **left upright's** robot-facing side face at ≈ (3.40, **+0.73**): that post is only ~0.13 m thick outside the y ≈ +0.60 edge, and its x is a silhouette-edge depth (~3.30–3.40), not a new plane.
- **Plaque B:** mission ≈ **(5.57, −0.03)** — point-010 (5.571, −0.072) and point-013 on the same plaque from 3.10 m (5.573, +0.011) agree to 2 mm in x and 8 cm in y, so B sits directly behind the opening centre at x ≈ 5.57 m. C is leftmost in the image (larger y), A right of B (smaller y), but neither was tracked → coordinates unknown.

## Route status (through frame-00009, sim_time 23.32 s)
- **Pose:** `local_odometry` (2.6249, 0.9804) m, yaw 0.0137 rad; idle, upright (height 0.358 m). Net ≈2.62 m forward, ≈0.98 m left of the start line. Base ≈0.69 m short of the gate plane; the camera (x ≈ 2.945) ~0.37 m short, matching the gate still filling the image's right edge.
- **Executed (all succeeded):** lateral +0.6, lateral +0.3, forward 0.6, forward 0.6, forward 0.55 + lateral +0.1, forward 0.6, forward 0.6. Measured increments 0.537, 0.229, 0.5311, 0.5304, (0.4808 + 0.1047 diag.), 0.5301, 0.5299 m. Every forward request yields ≈0.530 m per 0.60 m asked (~12 % short); lateral shims also under-deliver. Confirm with odometry; scale requests by ≈1.13.
- **Stage `avoid_drawers`: likely satisfied, not directly re-confirmed.** Base is 1.19 m past the cabinet front plane while ~0.86 m to its **left** (base y 0.98 vs front-face points y ≤ +0.12), so it passes behind the body with lateral clearance; cabinet rear extent unmeasured.
- **Stage `pass_gate`: not started.** No crossing, no 0.8 m margin. **Lateral problem:** base y = +0.98 m is ≈1.04 m **left** of the opening centre (y ≈ −0.06) and only ~0.25 m left of the near left upright's outer face (y ≈ +0.73) — the base is outside the opening and would hit the frame if it drove straight on. Shift **right (−y)** to y ≈ 0 m (opening clear for y ≈ −0.35…+0.35) before/while crossing x ≈ 3.30, then continue to at least x ≈ 4.10. The seq 16 lateral +0.1 shim was leftward and widened the offset from 0.87 to ~1.04 m; the needed rightward shim (~1.0 m) must itself be scaled up.
- **Stage `approach_b`: not started.** B is ≈3.10 m away (bearing ≈ −0.32 rad, ~18° right). A 1.0 m standoff straight down the opening axis predicts a stop near **(4.57, −0.03) facing +x (yaw ≈ 0)**, which also satisfies x ≥ 4.10.
- **Live tracked points (frame-00009): point-012 (left upright side face), point-013 (plaque B).** All of 001–011 invalid. Re-verify after the next move; if they drop, use odometry plus fresh perception — point-010/013 give B's saved mission position (~5.57, −0.03).

**Return-leg note:** from an ideal finish at ≈(4.57, −0.03) yaw ≈ 0, turn ~180°, run back −x through the same opening (cross x ≈ 3.30 near y ≈ −0.06), then shift left (+y) to ≈ +0.9 m to clear the cabinet on its left side before the cabinet front plane x ≈ 1.43, finishing near (0, 0) with yaw ≈ π. Cabinet rear extent unknown — re-measure on the way back.

Detailed point table and per-move feedback: `episodes/outbound_01_measurements.md`.
