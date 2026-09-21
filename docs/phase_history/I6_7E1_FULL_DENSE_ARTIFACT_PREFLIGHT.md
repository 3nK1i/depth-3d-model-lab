# I6.7E1 — Read-only full dense artifact qualification

## Authoritative baseline

I6.7D frozen source: `i6_7d_closed_20260919_121022_265721`; complete source SHA256 `d3cf16e1a3a96371b2f42d976b957dc0e93d99d2f71f638a25c5f8d4cdf3da6d`. I6.7A–D, One Start controllers, provider adapters and UI remain unchanged in this increment. No Git/Gitea gate yet.

## Purpose and strict boundary

The I6.7B real report identifies a complete COLMAP MVS `dense/fused.ply` with 1,686,160 vertices (reported) and SHA256 `b3a3934ff502a0d42c154ded3f26f21b64e960c2ea2f5f588404eae01503ace6`. The I6.7D viewer shows a separate provisional sample of 4,096 vertices. The uploaded report and source checkpoint do not contain the complete PLY bytes; only the user's workstation can independently validate the full artifact.

E1 introduces a **read-only candidate-artifact preflight**, not final-result admission. It scans every PLY vertex in bounded chunks (up to 65,536 binary vertices subject to an 8 MiB byte budget, or 1,024 ASCII vertices), detects full-population invalid XYZ/RGB, truncation and identity mismatch, computes full-file SHA256, and records full vertex count and bounds. It refuses geometry outside the expected output root and direct symlinked files. The real smoke CLI checks the frozen report SHA, explicit COLMAP provider identity, PyCOLMAP frame binding, completed result, and no forbidden progressive/metric/Auto authority. A successful E1 result is evidence for the later typed sparse-to-dense final-admission policy, not authorization to render, persist, measure, or certify the geometry.

## Deferred from E1

- Production One Start dense execution/progress wiring, selection policy and cancellation.
- Typed sparse/session/run/request bindings and independent final-admission receipt.
- NPZ full-population validation for other concrete providers; this first real evidence probe is explicitly limited to the existing COLMAP MVS PLY, without changing provider availability.
- Full-density GPU buffering, thread-safe asynchronous load, navigation/LOD and 2060/4090/5090 performance qualification.
- Persistent canonical dense-scene promotion, measurement/metric scale, mesh/Auto.

No rerun of PatchMatch, fusion or SfM is needed to qualify the existing frozen PLY. Before applying E1, verify SHA256 and frozen-source anchors. On the workstation, run focused tests then read-only real artifact probe. The result cannot be called admitted or full-density rendered until later I6.7E/F gates prove it.
