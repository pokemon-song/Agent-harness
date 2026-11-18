---
path: "episodes/outbound_03_finish_return_start.md"
revision: 9
scope: "roundtrip-82c2daea3ebf493a825a40d1d3d78685"
status: "confirmed"
evidence_ids: ["event-31a406a1ef9d4b99864b678c1621555b","event-7005a7cc2003473b8a6f01647fd11116","event-d4ac6bca96134320b4105a992426de74","event-bf3eda93cc924dbf975306bd1d41f8d2"]
evidence_max_seq: 39
observed_at: "2026-11-11T12:20:57.077475+00:00"
evidence: [{"id":"event-31a406a1ef9d4b99864b678c1621555b","observed_at":"2026-11-11T12:20:29.461473+00:00","seq":36},{"id":"event-7005a7cc2003473b8a6f01647fd11116","observed_at":"2026-11-11T12:20:35.632132+00:00","seq":37},{"id":"event-d4ac6bca96134320b4105a992426de74","observed_at":"2026-11-11T12:20:55.978438+00:00","seq":38},{"id":"event-bf3eda93cc924dbf975306bd1d41f8d2","observed_at":"2026-11-11T12:20:57.077475+00:00","seq":39}]
---
# Outbound leg, frames 00017–00019: B re-fix, final approach, finish, return start state

Scope `roundtrip-82c2daea3ebf493a825a40d1d3d78685`; scene `composite_drawers_gate_b`; episode `65e33cbb0a944297a54e00cd0ef23872`; sim 54.2 → 57.28 s.

## Sequence in this window
1. **seq 36 — `track_point` (frame-00017, u 330, v 78).** Re-anchored plaque B as **point-019**: valid, confidence 1.0, range **1.1274 m**, bearing −0.0176 rad, relative_xy (1.1273, −0.0198). All older points (011–018) were already invalid at that frame (`optical_flow_lost`, `inconsistent_flow`, `geometry_jump`), so point-019 is the only live B fix and it is the one that carried through to `finish`.
2. **seq 37 — `move_relative` forward 0.18 m, lateral −0.04 m.** Succeeded but under-delivered: measured translation in start frame **(0.1225, −0.0118) m** (68 % / 30 % of request), yaw delta +0.0046 rad, heading error 0.0046 rad, position error 0.064 m, elapsed 2.48 s; stop ok (0.011 m/s, 0.0007 rad/s). After it, odometry (4.5677, −0.0961) yaw 0.0569, point-019 range **1.0045 m**, bearing −0.0171.
3. **seq 38 — `get_geometry`:** returns the same point table; point-019 still the only valid point, 1.0045 m at frame-00018.
4. **seq 39 — `finish`:** recorded as a declaration, evaluator separate. Pre-finish observation (frame-00019): pose **(4.5662, −0.0922) m, yaw 0.0589 rad**, idle, upright (`fallen` false, body height 0.3631 m, pitch −0.036 rad), speed 0.0049 m/s, yaw rate 0.0043 rad/s. point-019 on frame-00019: range **1.0059 m**, bearing **−0.023 rad**, confidence 0.999.

## Measured pose ladder
| stop | x (m) | y (m) | yaw (rad) | note |
| --- | --- | --- | --- | --- |
| frame-00017 start | 4.4448 | −0.0907 | 0.0523 | idle before re-fix |
| frame-00018 | 4.5677 | −0.0961 | 0.0569 | after forward 0.18 / lateral −0.04 |
| frame-00019 (finish) | 4.5662 | −0.0922 | 0.0589 | 15 mm backlash during stop; final state |

Total travel past the gate plane x = 3.30: 1.266 m (requirement 0.8 m).

## B estimate from the finish observation
Back-solving point-019's mission offset through yaw 0.0589 from (4.5662, −0.0922) gives B ≈ **(5.5715, −0.0561)**, i.e. x agrees with the 5.570–5.576 cluster of the earlier fixes, while y lands near the high end of the −0.144 … −0.056 spread. Standing value for planning: **B ≈ (5.571, −0.10), y ≈ ±0.06**; the plaque filled u ≈ 275–430, v ≈ 10–190 on frame-00017.

## Standing state handed to the return planner
- Base at (4.5662, −0.0922) m, yaw 0.0589 rad, facing +x toward B; ~1.0 m from B (do not drive into the plaque on the return).
- No contact with cabinet, gate, plaque or anything else reported on any of the 15 moves; posture within limits throughout.
- All tracked points cleared by `finish`; `local_odometry` origin/axes unchanged, so these mission coordinates remain valid for the return leg.
- Reusable lessons from this window: short moves under-deliver (0.18 m request → 0.12 m), so overshoot modestly (~1.15–1.4×) and re-read odometry; a single freshly re-anchored point (point-019) survived where flow-tracked structural points did not, so re-anchor on the nearest high-contrast surface at each waypoint.

Route and stage analysis live in `tasks/outbound_drawers_gate_B.md`; earlier frames in `episodes/outbound_01_measurements.md` and `episodes/outbound_02_crossing_and_approach.md`.