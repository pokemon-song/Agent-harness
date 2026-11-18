---
path: "MEMORY.md"
revision: 9
scope: "roundtrip-82c2daea3ebf493a825a40d1d3d78685"
status: "confirmed"
evidence_ids: ["event-bf3eda93cc924dbf975306bd1d41f8d2","event-7005a7cc2003473b8a6f01647fd11116"]
evidence_max_seq: 39
observed_at: "2026-11-11T12:20:57.077475+00:00"
evidence: [{"id":"event-bf3eda93cc924dbf975306bd1d41f8d2","observed_at":"2026-11-11T12:20:57.077475+00:00","seq":39},{"id":"event-7005a7cc2003473b8a6f01647fd11116","observed_at":"2026-11-11T12:20:35.632132+00:00","seq":37}]
---
# Memory index

- `tasks/outbound_drawers_gate_B.md` — outbound task identity, stage rules, tolerances, shared coordinate frame (roundtrip-82c2daea3ebf493a825a40d1d3d78685), scene geometry, outbound result and the return-leg route.
- `episodes/outbound_01_measurements.md` — frames 00002–00012: point table and per-move feedback (cabinet, gate, plaque fixes).
- `episodes/outbound_02_crossing_and_approach.md` — frames 00013–00017: gate crossing, plaque B fixes and latest motion-feedback ratios.
- `episodes/outbound_03_finish_return_start.md` — frames 00017–00019: plaque B re-fix (point-019), final approach, `finish` declaration and the exact start state handed to the return leg.

Scene `composite_drawers_gate_b`: drawers cabinet → green door frame → letter plaques A/B/C (target B). The outbound leg ended at `finish`; the return leg is planned from `tasks/outbound_drawers_gate_B.md` plus fresh perception.