# Phase 9.3J — Real-Scene End-to-End Qualification / Phase 9.3 Closure

## Scope

Phase 9.3J closes the interactive Canonical Digital Twin campaign without changing the locked roadmap or authority model. It adds two qualification paths:

1. a real RTX-2060 South Building campaign that runs the existing Phase 9.3H production DA3 worker through the isolated `eia-da3` runtime, canonicalizes the resulting relative geometry, persists the canonical scene, and verifies read-back integrity/provenance; and
2. an additive, uncertainty-aware Measure & Inspect → canonical registration bridge that emits 3D dimension endpoints only when exact source-frame, canonical-object, camera-pose, matching source measurement-box fit method, metric-scale, and pose/registration-uncertainty evidence are all present.

Phase 7.5 measurement values and source uncertainty remain authoritative. Canonical registration never rewrites them. Calibration remains optional for application use.

## Authority rules

- Canonical dense/surface providers own canonical-geometry authority after canonicalization.
- DA3 remains supporting relative depth/pose evidence and never creates metric scale.
- A Phase 9.2 reviewed mask remains 2D semantic evidence unless real reconstruction/depth association exists.
- Measurement registration is additive presentation/provenance. It never makes semantics or camera poses measurement authority.
- Missing metric scale, source-frame identity, canonical semantic association, pose uncertainty, or registration evidence fails closed as `UNAVAILABLE`/`PROVISIONAL`; no 3D metric dimension line is produced.
- `certified_metrology=false` throughout Phase 9.3J.
- Gaussian and NeRF are not executed in this phase; their interactive qualification remains Phase 9.4/9.5.

## Measurement registration states

- `QUALIFIED`: exact observation-ID semantic binding, exact source-frame SHA-256 → persisted camera binding, scene-bound qualified metric scale with scale uncertainty, and quantified pose translation/rotation uncertainty. Genuine canonical dimension endpoints may be rendered.
- `PROVISIONAL`: a plausible explicit transform/metric chain exists but required uncertainty evidence is incomplete. No dimension geometry is rendered.
- `UNAVAILABLE`: required binding/metric evidence is absent. Original Measure & Inspect workflow remains fully usable.
- `REJECTED`: reserved for explicit invalid registration evidence; no dimension geometry is rendered.

The registered dimension uncertainty combines the original source-expanded measurement uncertainty with scale uncertainty. Pose translation/rotation uncertainty contributes separately to endpoint placement uncertainty. Rigid pose uncertainty does not overwrite the original measured dimension.

## Real DA3 qualification

Default real dataset:

`/home/eiatech/Desktop/phase8_3c_south_building/south-building/images`

Qualified isolated runtime:

- Python: `/home/eiatech/anaconda3/envs/eia-da3/bin/python`
- checkout: `/home/eiatech/Desktop/eia-da3-backends/depth-anything-3`

The real qualification harness runs `scripts/run_phase9_3h_da3_canonical_job.py` from the production `depth` interpreter. That worker probes/launches DA3 only through the qualified external interpreter, validates runtime-lock/import-source/revision/GPU evidence, then canonicalizes and persists the output in `depth`.

A successful real campaign must prove:

- `da3_runtime_probed=true`
- `da3_backend_started=true`
- qualified runtime lock valid
- the imported `DepthAnything3` implementation originates from the qualified checkout
- measured GPU identity and positive finite peak VRAM are present
- the RTX-2060/Turing compatibility patch is reported when the real GPU is an RTX 2060
- persistent scene/checksum/file references read back successfully
- canonical surface is non-empty
- persisted DA3 camera poses and exact source-frame fingerprints are present
- canonical geometry authority remains true after canonicalization
- scale remains arbitrary / `unit_to_metre=null` unless separate real metric evidence is supplied
- semantics/QC are unavailable rather than fabricated when the new DA3 scene does not contain them
- no DA3/Gaussian/NeRF heavy module is imported into the production qualification process

Measured elapsed time and peak VRAM are real evidence for the hardware that actually ran the campaign. No RTX 4090/5090 values may be extrapolated.

## Physical calibration / metric authority

Real physical checkerboard/ChArUco calibration is optional and may remain explicitly deferred if no real capture/profile is supplied. Its absence must not fail the DA3/canonical qualification or disable Measure & Inspect/Reconstruction.

A real DA3 scene at arbitrary scale therefore reports canonical measurement registration as `UNAVAILABLE`. This is expected, not a failure. Qualified 3D measurement placement becomes available only when the same scene has real scene-bound metric authority, an exact semantic observation match, a matching source-frame camera pose, and quantified pose/registration uncertainty.

## Automated gates

- `./scripts/run_phase9_3j_tests.sh quick`
- `./scripts/run_phase9_3j_tests.sh integration`
- `./scripts/run_phase9_3j_tests.sh backward`
- `python scripts/run_phase9_3j_measurement_registration_smoke.py`
- `python scripts/run_phase9_3j_measurement_registration_preview.py --show` (real Qt/VisPy, deterministic qualified evidence)
- phase closure: `./scripts/run_phase9_3j_tests.sh full`

`full` is justified at this phase-closure gate and runs the complete current-tree unit, UI, integration, and stress suites. It does not substitute for the real DA3 campaign.


### Qualified measurement-placement preview

Before the real arbitrary-scale DA3 campaign, validate the qualified UI path using deterministic explicit evidence:

```bash
python scripts/run_phase9_3j_measurement_registration_preview.py --show
```

This preview must show the same source measurement in Measure & Inspect and Reconstruction, with a genuine canonical dimension segment only because the fixture contains exact source-frame/object/camera/metric/pose-uncertainty evidence. It must not launch any backend.

## Real campaign commands

```bash
rm -rf /tmp/phase9_3j_real_scene
python scripts/run_phase9_3j_real_scene_qualification.py \
  --output-dir /tmp/phase9_3j_real_scene
```

Expected summary:

`/tmp/phase9_3j_real_scene/phase9_3j_qualification_summary.json`

The summary distinguishes real execution, persistent-scene evidence, authority, measurement-registration availability, runtime isolation, and deferred later-phase work.

Then load the produced scene without launching any backend:

```bash
python scripts/run_phase9_3j_real_scene_preview.py \
  --qualification-summary /tmp/phase9_3j_real_scene/phase9_3j_qualification_summary.json \
  --show
```

## Closure gate

Phase 9.3J may be committed/closed only after:

- focused automated gates are green;
- the full current-tree regression is green;
- the real RTX-2060 DA3 qualification summary reports `PASS` (not validation-only);
- the real persistent scene is manually loaded in the application and canonical mesh/cameras/navigation/resource cleanup are verified;
- arbitrary/unqualified scale and unavailable semantics/QC are presented truthfully when applicable;
- no heavy-runtime isolation violation occurs;
- any unavailable real physical calibration/metric-registration validation is recorded as deferred, never falsely passed.

No branch merge, release tag, Gaussian execution, or NeRF execution belongs to this Phase 9.3J commit. Version/release identity remains unchanged until the final Phase 9 release gate.
