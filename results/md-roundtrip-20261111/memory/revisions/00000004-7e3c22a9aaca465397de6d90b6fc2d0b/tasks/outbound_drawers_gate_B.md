---
path: "tasks/outbound_drawers_gate_B.md"
revision: 4
scope: "roundtrip-82c2daea3ebf493a825a40d1d3d78685"
status: "confirmed"
evidence_ids: ["event-95e6340cbf424365af4238e5cde6c63a","event-492fcacf6c1e4a05aa6dd78db1c124f0","event-81a3e2976f3a42e9bebc9360a7be43c0","event-4ca83b791abd4ae59fd43a807053ba90"]
evidence_max_seq: 19
observed_at: "2026-11-11T12:15:35.035162+00:00"
evidence: [{"id":"event-95e6340cbf424365af4238e5cde6c63a","observed_at":"2026-11-11T12:14:08.424535+00:00","seq":16},{"id":"event-492fcacf6c1e4a05aa6dd78db1c124f0","observed_at":"2026-11-11T12:14:20.486058+00:00","seq":17},{"id":"event-81a3e2976f3a42e9bebc9360a7be43c0","observed_at":"2026-11-11T12:14:44.836966+00:00","seq":18},{"id":"event-4ca83b791abd4ae59fd43a807053ba90","observed_at":"2026-11-11T12:15:35.035162+00:00","seq":19}]
---
# Outbound leg: drawers → green gate → letter B

Scope `roundtrip-82c2daea3ebf493a825a40d1d3d78685`; scene `composite_drawers_gate_b`; episode `65e33cbb0a944297a54e00cd0ef23872`.

## Registered instruction (public announcement, seq 1, 2026-11-11T12:07:44Z)
Ordered stages (stages_ordered = true):

1. `avoid_drawers` — go around the front drawer cabinet so the **base origin** passes behind the cabinet body and leaves body clearance.
2. `pass_gate` — then pass the base origin through the opening of the **green** rectangular door frame; while crossing, the base origin must be inside the opening; continue at least **0.8 m** past the frame plane in the crossing direction.
3. `approach_b` — among the white letter plaques A, B, C behind the gate, select **B**, move to about **1.0 m** from B's plaque center, face B, stop, then call `finish`.

Limits: no contact with cabinet, frame, plaques or any scene object (collision-free required). Stages must be done in order; it is forbidden to skip the gate and head straight for B. Tolerances at finish: distance 0.18 m, heading 0.25 rad, stopped speed ≤ 0.12 m/s, stopped yaw rate ≤ 0.25 rad/s, reference body = base.

Note in the instruction: the front camera sits 0.32 m ahead of the base origin, so the gate leaving the image does not prove the base has passed it. Track reliable surface points on the frame while it is visible and confirm crossing plus the 0.8 m extra from local_odometry and actual motion feedback — do not substitute requested motion for measured displacement.

After `finish` only the outbound leg ends: the planner's conversation, temporary notes and tracked points are cleared, but the robot, scene, simulation time, pose and local_odometry do **not** reset. The return leg must be planned from this memory plus fresh perception, retracing the outbound route.

## Coordinates and frames
- coordinate_frame_id `roundtrip-82c2daea3ebf493a825a40d1d3d78685`; coordinate_reference "continuous_mission_start; physical world and odometry origin are preserved".
- `local_odometry`: origin and horizontal axes fixed at the mission's initial base pose, preserved across planner-context resets (shared by outbound and return legs).
- `tracked_points.relative_xy_m`: relative to the current base body reference, x forward, y left, metres; yaw positive left.
- Body axes x forward, y left, z up; camera frame front_rgbd_optical at +0.32 m forward, +0.08 m up in body, optical x right / y down / z forward.

## Scene geometry in the mission frame (x forward, y left)
- **Cabinet:** front plane x ≈ 1.43 m (point-001 at y ≈ −0.04, point-007 at y ≈ +0.12). Depth and full width still unmeasured; the cabinet has not been imaged since frame-00004.
- **Green gate:** frame plane x ≈ **3.30 m**, now confirmed three ways — point-008 twice (x = 3.313, y 0.585–0.599) and point-011 (3.314, 0.640). Right post point-009 at (3.289, −0.715) → opening between roughly y = +0.60 and y = −0.72, centre ≈ **y = −0.06**; sampled width ~1.3 m, clear inner width a little less.
- **Plaque B:** centre at mission ≈ **(5.571, −0.072)** (point-010, range 4.57 m), directly behind the opening centre. C is leftmost in the image (larger y) and A right of B (smaller y), but neither was tracked, so their coordinates are unknown.

## Route status (through frame-00009, sim_time 23.32 s)
- **Pose:** `local_odometry` (2.6249, 0.9804) m, yaw 0.0137 rad; idle, upright (height 0.358 m, pitch 0.0043, roll −0.0033). Net ≈2.62 m forward, ≈0.98 m left of the start line. Base is ≈0.69 m short of the gate plane; the camera (x ≈ 2.945) is ~0.37 m short, which matches the gate still filling the right edge of the current image.
- **Executed (all succeeded):** lateral +0.6, lateral +0.3, forward 0.6, forward 0.6, forward 0.55 + lateral +0.1, forward 0.6, forward 0.6. Measured increments: 0.537, 0.229, 0.5311, 0.5304, (0.4808 + 0.1047 diag.), 0.5301, 0.5299 m. Every forward request delivers ≈0.530 m per 0.60 m asked (≈12 % short); lateral shims also under-deliver. Confirm with odometry; scale requests by ≈1.13.
- **Stage `avoid_drawers`: likely satisfied, not directly re-confirmed.** The base is 1.19 m past the cabinet front plane while staying ~0.86 m to its **left** (base y 0.98 vs front-face points y ≤ +0.12), so it passes behind the cabinet body with lateral clearance; the cabinet rear extent is still unmeasured, so verify if the cabinet comes back into view.
- **Stage `pass_gate`: not started.** No crossing, no 0.8 m overshoot. **Lateral problem:** the base y = +0.98 m is ≈1.04 m **left** of the opening centre (y ≈ −0.06) and ~0.34 m left of the sampled left-post face, i.e. the base is currently outside the opening and would hit the frame if it drove straight on. It must shift **right (−y)** to roughly y ≈ 0 m (opening is clear for y ≈ −0.35…+0.35) before/while crossing x ≈ 3.30, then keep going to at least x ≈ 4.10. The seq 16 lateral shim (+0.1) was leftward and enlarged this offset from 0.87 to ~1.04 m.
- **Stage `approach_b`: not started.** B is 3.13 m from the current base (bearing ≈28° to the right). A 1.0 m standoff reached straight down the opening axis predicts a stop near **(4.57, −0.07) facing +x (yaw ≈ 0)**, which also satisfies gate plane + 0.8 m (x ≥ 4.10).
- **Live tracked points: none.** All points (001–011) are invalid as of frame-00009; continue from odometry and fresh perception. point-010's saved mission position (5.571, −0.072) remains the best B estimate but is no longer tracked → treat as needing visual re-confirmation.

**Return-leg note:** with an ideal finish at ≈(4.57, −0.07) yaw ≈ 0, the return must first turn ~180°, run back −x through the same opening (cross x ≈ 3.30 near y ≈ −0.06), then shift left (+y) to ≈ +0.9 m to clear the cabinet on its left side before reaching the cabinet front plane x ≈ 1.43, and finish near (0, 0) with yaw ≈ π. The cabinet rear extent is still unknown — re-measure on the way back.

Detailed point table and per-move feedback: `episodes/outbound_01_measurements.md`.
