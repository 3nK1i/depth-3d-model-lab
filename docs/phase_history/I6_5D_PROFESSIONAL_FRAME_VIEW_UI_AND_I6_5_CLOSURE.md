# I6.5D — Professional Frame/View UI and I6.5 Closure

## Scope

I6.5D productizes the frozen I6.5A-C frame/view planning contracts in the normal
Reconstruction UI. It adds one compact `FRAME & VIEW PLAN` disclosure group and
an application-owned resource handoff. It does not execute source sampling,
mutate worker/SfM settings, start reconstruction, select dense providers, enable
production Auto, or add progressive SfM visualization.

## Product semantics

The UI keeps these facts separate:

1. **Request** — `Adaptive`, `All eligible`, or `Manual limit`.
2. **Source** — source provenance and known source-view count.
3. **Planned** — deterministic effective pre-SfM target.
4. **Resource** — I4.3F2-derived planning heuristic, explicitly not a runtime guarantee.
5. **Hardware** — startup GPU/RAM observation supplied by the application layer.
6. **Plan ID** — deterministic I6.5B planning provenance SHA-256.

A manual limit is a maximum, not a guaranteed execution count. `All eligible`
is never silently capped or substituted with Adaptive because of resource
pressure; the UI surfaces a warning instead.

## Resource ownership

`ReconstructionController` remains hardware-probe-free. `MainWindow` reuses the
existing `SettingsPanel.hardware_info` startup GPU snapshot and reads Linux
`MemAvailable` through `app/reconstruction_frame_view_resources.py`. The result
is converted to the frozen `AdaptiveSelectionResourceSnapshot` contract and
explicitly installed on the controller before the Reconstruction page is bound.

The snapshot is planning provenance only. It is not continuous telemetry and it
is not an empirical guarantee that a given number of views will fit at runtime.
Execution resource preflight remains authoritative later in the workflow.

## UI behavior

The `FRAME & VIEW PLAN` group is appended after the frozen I6.4D primary stack:

1. DIGITAL TWIN WORKSPACE
2. BUILD & PROCESSING
3. CANONICAL SCENE
4. RULER METRIC ESTIMATE
5. RECONSTRUCTION ADVISOR
6. DENSE RGB POINTS
7. PROVIDER & AUTO
8. FRAME & VIEW PLAN

`Apply Plan` records explicit planning intent only. `Reset` restores the default
Adaptive request. Changing source preserves the user's request while forcing a
new source-bound effective plan and provenance digest.

## Authority and non-goals

- Planned views are not registered cameras.
- PyCOLMAP remains the sole canonical camera/global-frame authority.
- No metric scale or certified metrology is established.
- No provider selection/activation/execution is added.
- Production Auto remains unchanged and unqualified.
- No retry-until-pass or silent replan-after-failure behavior is added.
- Progressive SfM visualization remains I6.6.

## Qualification

I6.5D closure requires focused pure/static tests, real Qt UI tests, I6.5C/B and
I6.4D/page regressions, the complete unit suite, compile/shell checks, scope
checks, and manual UI validation on the normal application.
