# I4.3F3 — Production validation and adaptive-selection closure

## Purpose

F3 closes I4.3F after the F1 explainable planner and F2 production wiring have both passed on the authoritative workstation. It is validation/closure focused, with one narrow hardening correction: the low-level `KeyframeSelectionPolicy` default is now `None` rather than the historical `250`, so no implicit fixed cap can re-enter through the core API. Explicit engineer limits remain supported.

## Qualification coverage

F3 validates:

- small, medium and large finite sources;
- constrained 6 GB, RTX-2060-class and RTX-5090-class resource profiles;
- monotonic resource-aware expansion with no fixed cap;
- all ten I4.3 dense-provider implementations;
- provider-specific minimum views and relative work pressure;
- temporal source ordering for RollingDepth and Video Depth Anything;
- deterministic repeated planning;
- explicit pre-SfM and post-SfM engineer overrides;
- immutable authoritative SfM during provider projection/preparation;
- production CLI/controller/core defaults remain adaptive;
- F1/F2 predecessor evidence remains fail-closed.

The synthetic resource profiles are architecture/planning qualification inputs, not measured runtime or memory guarantees. Real provider quality/runtime comparison remains I4.3G/I5.

## Authority locks

F3 does not select or promote a provider, implement Auto, authorize hybrid fusion, establish physical metric scale, authorize measurements/metrology, or generate a surface mesh. PyCOLMAP remains the sole authoritative global camera/reconstruction frame and `permanent_global_view_cap` remains `None`.

## Closure

A passing F3 binding sets `i4_3f_closed=true` and authorizes I4.3G controlled cross-provider scenario qualification. It does not identify a universal provider winner.
