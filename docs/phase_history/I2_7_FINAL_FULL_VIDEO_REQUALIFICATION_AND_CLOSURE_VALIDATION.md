# I2.7 — Final Full-Video Requalification & I2 Closure

I2.7 validates the I2.6 promoted `d12_sparse` formal F/G scene. It introduces no new filtering policy and does not rerun SfM, DA3, dense fusion, or persistence generation.

## Automated gate

`run_i2_7_final_requalification.py` verifies:

- I2.6 pass and exact manually-reviewed counterfactual equivalence.
- promoted F/G point-count and exact geometry agreement.
- exact supported-voxel accounting and `weak_manifold_attachment` provenance.
- original I2 upstream registration/alignment/selection quality gates.
- recomputed 26-neighbour largest-component health.
- recomputed peripheral/far/weak diagnostics.
- promoted full/robust envelope improvement with robust-scene preservation.
- persistent bundle integrity, relative references, self-contained state, and relocation proof.
- arbitrary SfM scale and unpromoted metric/measurement/mesh authority.

A passing automated run is **not** I2 closure. It yields `READY_FOR_FINAL_MANUAL_REVIEW`.

## Final manual gate

Review fixed RGB/region views, then run `run_i2_7_ui_sanity.py` to verify Extended/Core switching, Dense RGB Scene Layer OFF/ON, Fit, Unload and reload through the existing Reconstruction workspace.

Only after those pass may `run_i2_7_finalize_closure.py --confirm-fixed-view-review-pass` write `i2_7_closure.json` with I2 complete and authorization to start I3. Surface-mesh authorization remains false until the full I campaign reaches mesh readiness.
