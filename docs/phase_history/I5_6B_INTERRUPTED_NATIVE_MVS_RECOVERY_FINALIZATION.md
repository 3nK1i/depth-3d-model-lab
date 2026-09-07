# I5.6B interrupted-parent native-MVS recovery finalization

This narrow correction finalizes one interrupted I5.6B replay without rerunning SfM, PatchMatch, StereoFusion, or any dense provider.

The original parent runner was interrupted after scenarios 1-5 had written normal provider summaries and scenario records. The sixth scenario (`castle_dynamic_proxy`) continued in the already-launched external PyCOLMAP worker and produced all 19 geometric depth maps, all 19 geometric normal maps, `fused.ply`, and `fused.ply.vis`, but the worker terminated before writing its normal `result.json` and before the adapter could write `i4_3b_summary.json` or the I5.6B scenario record.

Recovery is admitted only when the surviving frozen artifacts prove all of the following:

- the exact I5.6A-admitted `castle_dynamic_proxy` scenario is being recovered;
- the saved adaptive plan and worker request preserve the production COLMAP MVS configuration;
- geometric consistency remains enabled and production defaults are unchanged;
- all 19 requested geometric depth and normal outputs are present exactly;
- the fused PLY has a positive vertex count and `fused.ply.vis` contains exactly the same number of point-support records;
- all registered-image and authoritative sparse-model SHA256 values from the original worker request still match;
- the stale worker state truthfully remains at started-but-not-finalized `stereo_fusion`;
- the normal worker result, provider summary, and scenario record are absent before recovery;
- scenarios 1-5 are reused from their existing immutable records and artifact bindings.

The recovery binds both the original replay-test receipt used before heavy execution and a fresh recovery-validation receipt for the recovery code path. It does not fabricate the missing worker result or rewrite the stale worker state. It writes a recovery-specific provider summary, an immutable native-artifact manifest, the missing scenario-6 record with explicit recovery provenance, and the normal top-level I5.6B replay evidence. Missing wall-time and peak-resource metrics are left unavailable rather than estimated.

The recovered scenario may carry `runtime_status=pass` only because its complete native MVS outputs and authority integrity are independently validated. The evidence explicitly records that the provider adapter did not return normally and that the execution-result representation was reconstructed from existing native artifacts. The worker termination cause remains unresolved.

I5.6C is authorized only after this recovery finalization passes and the resulting top-level I5.6B evidence is SHA-frozen.
