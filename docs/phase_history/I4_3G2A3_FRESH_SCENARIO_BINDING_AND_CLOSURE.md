# I4.3G2A.3 — Fresh scenario binding and G2A closure gate

## Purpose

G1 froze three fresh MVS-vs-DA3 scenario semantics but deliberately did not
invent filesystem bindings:

- `g2_complete_room_mvs_da3`
- `g2_low_texture_mvs_da3`
- `g2_weak_overlap_mvs_da3`

G2A.3 converts those logical scenarios into exact, content-addressed source
bindings before any G2B SfM/MVS/DA3 execution is allowed.

The step is intentionally read-only with respect to reconstruction. It may read
source pixels for diagnostics and write evidence/contact sheets, but it does
not start SfM, MVS, DA3, Auto, hybrid logic, or provider-quality evaluation.

## Semantic binding policy

A path or directory name is never accepted as semantic evidence.

### Complete room

Complete-room semantics require an explicit engineer confirmation after visual
review of a generated contact sheet. No pixel heuristic attempts to infer that
a capture is a complete room.

South Building remains only a discovered real-scene candidate. Its existing
use elsewhere in the project does not automatically qualify it as the G2
complete-room capture.

### Low texture

The original G2A.3 log-compressed aggregate-ratio gate was retired by the
pre-provider construct-validation amendment `I4.3G2A.3-A1`. Known low-texture
positive controls demonstrated that ratios of the compressed aggregate proxy
do not represent percentage texture reduction faithfully. The legacy proxy is
therefore retained for diagnostics only. See
`I4_3G2A3_A1_LOW_TEXTURE_CONSTRUCT_VALIDATION_AMENDMENT.md`.

A1 keeps the original `0.75` material-reduction criterion but applies it to raw
component ratios. A candidate qualifies only when edge-density ratio is at
most `0.75` and either Laplacian-variance ratio or contrast-standard-deviation
ratio is also at most `0.75`. For a controlled subset, G2A.3 chooses the
contiguous window with the lowest deterministic component-gate qualification
margin. A separate explicit real low-texture directory uses the same component
gate against the complete-room reference. The semantic label still requires
explicit engineer confirmation.

### Weak overlap

By default, G2A.3 derives an ordered, increased-spacing subset from the
confirmed complete-room capture. Frames selected for the controlled low-texture
subset are excluded when possible so weak overlap is not intentionally
confounded with low texture.

The preflight proxy requires:

- source order preserved;
- capture-spacing factor at least 2;
- enough ORB feature connectivity to avoid a trivially disconnected subset;
- explicit engineer confirmation.

ORB evidence is only a cheap pre-provider connectivity proxy. It is **not**
claimed as authoritative overlap measurement. G2B must revalidate weak-overlap
behavior from the authoritative PyCOLMAP SfM/covisibility evidence. Failure to
produce a valid authoritative SfM remains a legitimate scenario result; G2A.3
does not weaken SfM gates to make the scenario pass.

A separate weak-overlap capture may be supplied; its cheap overlap proxy is
compared with the complete-room reference, but authoritative SfM revalidation
is still mandatory in G2B.

## Frozen evidence

Each binding records:

- exact source directory;
- exact ordered image list;
- per-file size and SHA256;
- aggregate asset-set SHA256;
- semantic confirmation state;
- derivation policy;
- objective diagnostic values;
- visual-review contact sheet and SHA256;
- explicit statement that semantic labels were not inferred from path names;
- unchanged authority claims.

When a G2A.3 snapshot is consumed by the final G2A preflight, source files and
contact sheets are rehashed and the low-texture/weak-overlap diagnostics are
recomputed. A hand-edited `objective_proxy_qualified=true`, changed source
image, changed subset, changed policy, or changed review artifact fails closed.

## Authority locks

Always false in G2A.3:

- real provider execution started
- SfM started
- MVS started
- DA3 started
- provider selected/promoted
- Auto policy implemented
- hybrid authorized
- metric scale claimed
- measurement authoritative
- certified metrology
- surface mesh generated

PyCOLMAP remains the sole camera/frame authority and the permanent global view
cap remains `None`.

## Closure rule

G2A.3 being `ready` does not itself authorize G2B. The normal G2A preflight must
consume the frozen G2A.3 snapshot and reverify:

- TempleRing source + I4.2 authority chain;
- recovered House historical chain;
- all three fresh source identities and diagnostics.

Only the combined G2A snapshot may report:

- `fresh_scenario_bindings_ready = 4`
- `historical_binding_verified = true`
- `all_g2_sources_bound = true`
- `i4_3g2b_real_execution_authorized = true`

## Automated gate

```bash
bash scripts/run_i4_3g2a3_tests.sh all
```

The gate includes focused G2A.3 tests plus the complete G2A -> G1 -> immediate
F3/I4.3A predecessor regression.
