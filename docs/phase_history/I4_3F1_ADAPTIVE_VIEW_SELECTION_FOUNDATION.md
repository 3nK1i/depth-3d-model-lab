# I4.3F1 — Adaptive Frame/View Selection Foundation

I4.3F is broader than the provider-adapter slices because it affects both the
pre-SfM frame plan and the post-SfM dense-provider view plan.  The work is split
internally so the validated provider chain is not rewritten in one step.

## Scope of F1

F1 freezes the provider-neutral, post-SfM adaptive view-selection contract and
planner.  It is execution-free with respect to dense backends and does not wire
selection into the production worker yet.

The planner consumes the authoritative `OfflineSfmResult` and evaluates only
registered PyCOLMAP views.  Selection evidence includes:

- SfM registration / sparse landmark support
- pairwise sparse-landmark overlap
- camera-baseline diversity
- image sharpness
- image resolution
- source temporal/span coverage
- available RAM
- available/free VRAM
- estimated relative provider work
- provider-specific minimum-view / temporal requirements

All registered views remain eligible.  If the estimated work fits the current
resource snapshot, the planner keeps every registered view.  If resource
pressure is present, it selects a deterministic subset while preserving source
span, overlap/connectivity, baseline diversity and quality.  The resource
capacity is a transparent planning heuristic, not a benchmark or a claimed
memory model; real runtime envelopes remain I4.3G/I5 evidence.

## Provider-specific behavior

Every concrete I4.3 provider has an explicit selection-requirement record.
Temporal providers preserve source ordering and are marked stateful.  MVS and
multi-view providers prioritize overlap/baseline support more strongly than
independent single-image depth providers.  No provider winner is selected.

## Engineer visibility / override

F1 supports explicit engineer overrides:

- use all registered views;
- use an explicit registered-keyframe subset.

Overrides are visible in plan provenance and cannot violate a provider's
minimum-view requirement.  They are not permanent production caps.

Every view receives a selected/excluded decision and reason codes.  The plan
records source/selected estimated work, resource observations and the dynamic
resource-derived target count when reduction is necessary.

## Hard locks preserved

- PyCOLMAP remains sole camera/global-frame authority.
- The authoritative SfM result is not modified.
- `permanent_global_view_cap = None`.
- Auto provider selection is not implemented.
- No provider is selected or promoted by F1.
- Hybrid remains unauthorized.
- `metric_scale_claimed = false`.
- `measurement_authoritative = false`.
- `certified_metrology = false`.
- `surface_mesh_generated = false`.

## Deliberately not performed in F1

F1 does **not** yet:

- replace the production offline worker's historical frame/keyframe controls;
- subset a real provider run;
- modify any D/B/C/E provider adapter;
- implement provider Auto selection;
- run a cross-provider quality comparison;
- change filtering, persistence, UI or controller behavior.

Those production wiring changes belong to the next narrow I4.3F slice after F1
is validated on the authoritative workstation.

## Validation

The automated matrix covers:

- all ten provider requirement records;
- no permanent global view cap;
- all-view selection when resources fit;
- dynamic reduction under resource pressure;
- provider-specific relative cost behavior;
- overlap/baseline/quality/source-span evidence;
- deterministic selection;
- temporal source-order preservation;
- engineer all-view and explicit-subset overrides;
- unknown/undersized override rejection;
- missing sharpness evidence fallback;
- large registered-view stress sets;
- predecessor authority/cap refusal;
- metric/measurement/metrology/mesh authority refusal.

The real binding smoke is execution-free and requires the passing I4.3A
architecture snapshot plus the passing I4.3E6 closure/binding snapshot.  It
records the current workstation RAM/VRAM snapshot and authorizes F2 only if the
predecessor locks remain intact.
