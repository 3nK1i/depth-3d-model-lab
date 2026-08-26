# Phase F — Confidence / Consistency / Artifact Filtering

## Scope

Phase F formalizes the confidence/consistency model first proven by E2 and applies it to the passing E3 expanded dense scene. It consumes saved E3 selected-view clouds and the E3 voxel-fused all-cloud; it does not rerun PyCOLMAP or DA3.

The reusable filtering contract lives in `core/reconstruction/dense_confidence_filtering.py` and is dependency-light. F computes robust per-view voxel dispersion, multi-view support, local 26-neighbour/radius-2 topology, high-support anchoring, connected-component size, explicit artifact-reason bits, and a view-count-normalized confidence score.

## Outputs

- `f_filtered_extended.npz/.ply` — primary filtered dense RGB geometry.
- `f_filtered_core.npz/.ply` — stricter high-confidence geometry.
- `f_rejected_supported_artifacts.npz` — rejected >=2-view voxels with reason codes.
- `f_attached_single_view_fringe_diagnostic.npz` — low-authority attached single-view coverage.
- `f_detached_single_view_artifacts_diagnostic.npz` — detached single-view artifact candidates.
- `f_coverage_candidate_diagnostic.npz` — Extended + attached fringe, diagnostic only.
- `f_summary.json` — exact accounting, confidence, topology, artifact reasons, policy and authority.

## Required gates

- Source E3 is `PASS`, capacity-clean and explicitly authorizes F.
- Reconstructed per-view voxel counts exactly match the E3 fused support counts.
- Core and Extended products satisfy minimum size/fraction gates.
- Core is a strict subset of Extended.
- The supported scene retains a sufficiently dominant connected component.
- Metric scale, canonical persistence, measurement authority and mesh authority remain unpromoted.

A passing F run authorizes only the next milestone: **G — Persistent Canonical Dense RGB Geometry**.

## Validation commands

```bash
python -m py_compile \
  core/reconstruction/dense_confidence_filtering.py \
  scripts/run_f_confidence_artifact_filtering.py \
  scripts/run_f_confidence_artifact_preview.py \
  tests/unit/test_f_confidence_artifact_filtering.py

bash scripts/run_f_tests.sh quick
bash scripts/run_f_tests.sh affected
```

Real scene:

```bash
rm -rf /tmp/eia_f_confidence_artifact_filtering
python scripts/run_f_confidence_artifact_filtering.py \
  --e3-dir /tmp/eia_e3_qualified_scene_fusion_expansion \
  --output-dir /tmp/eia_f_confidence_artifact_filtering
```

Recommended visual checks:

```bash
python scripts/run_f_confidence_artifact_preview.py --f-dir /tmp/eia_f_confidence_artifact_filtering --cloud extended --color-mode rgb --overlay-sparse --show
python scripts/run_f_confidence_artifact_preview.py --f-dir /tmp/eia_f_confidence_artifact_filtering --cloud core --color-mode rgb --overlay-sparse --show
python scripts/run_f_confidence_artifact_preview.py --f-dir /tmp/eia_f_confidence_artifact_filtering --cloud extended --color-mode confidence --overlay-sparse --show
python scripts/run_f_confidence_artifact_preview.py --f-dir /tmp/eia_f_confidence_artifact_filtering --cloud rejected --color-mode artifact --overlay-sparse --show
```
