# I4.3G2B — MVS vs DA3 production comparison

G2B consumes the frozen I4.3G1 methodology and the final closed I4.3G2A
source-binding authorization. It does not reopen source selection or the A1
low-texture amendment.

## Execution contract

For each fresh static G2 scenario, the runner:

1. verifies exact G1 and final G2A SHA256 identities;
2. verifies the frozen source asset identity;
3. probes the isolated PyCOLMAP MVS and DA3 runtimes;
4. creates one authoritative PyCOLMAP SfM/BA result;
5. persists that SfM result before dense execution;
6. derives separate I4.3F adaptive provider plans from the same authoritative
   SfM result;
7. executes COLMAP MVS and DA3 sequentially using those projected subsets;
8. freezes per-provider G1-compatible production-native evidence;
9. defers matched-view/manual A/B judgement until native evidence exists.

House is `historical_read_only` and the runner refuses a new House execution.

For weak overlap, authoritative PyCOLMAP co-visibility connectivity is persisted
and must qualify before either dense provider starts. A connectivity failure is
preserved as the scenario outcome; matching/source/provider thresholds are not
relaxed to erase it.

## Authority

Always preserved:

- `camera_pose_authority_owner = pycolmap_sfm`
- `permanent_global_view_cap = null`
- `provider_selected = false`
- `provider_promoted = false`
- `auto_policy_implemented = false`
- `hybrid_authorized = false`
- `metric_scale_claimed = false`
- `measurement_authoritative = false`
- `certified_metrology = false`
- `surface_mesh_generated = false`

DA3 model-predicted extrinsics remain provider evidence and are never consumed
as canonical camera poses.

## Runtime preflight

`--mode preflight` is deliberately execution-free: source files are hashed and
both isolated environments are probed, but SfM, MVS and DA3 are not started.
Heavy production-native execution is authorized only when this preflight passes.
