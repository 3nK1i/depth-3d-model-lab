# I4.1.1 — Orbit Alignment & Coverage Diagnosis

## Purpose

Diagnose the I4.1 Horse capture after the ordinary I2 baseline-quality prerequisite returned `DEGRADED`.
This step is evidence-only: it does not rerun SfM/DA3/fusion, alter F/G, relax thresholds, claim metric scale,
authorize measurement, or generate/authorize a mesh.

## Required source state

- I4.1 completed with `status=degraded`.
- `failed_stage=object_orbit_baseline_quality`.
- I1 baseline itself passed and produced F/G persistent Dense RGB geometry.
- I2 baseline-quality summary exists and is degraded.

## Diagnostics

I4.1.1 records:

- exact failed ordinary-I2 gate names;
- separation of room/capture coverage gates from geometry/contract gates;
- I2 alignment denominator (accepted/candidates) and current I4 denominator (accepted/registered);
- accepted/rejected alignment orders, contiguous rejection clusters and reason histogram;
- accepted scale evolution, quantiles and >=1.5x adjacent transitions;
- registered, alignment-accepted and selected camera orbit metrics using the same scale-invariant I4.1 orbit kernel;
- selected-orbit gate subset independently from the ordinary I2 prerequisite;
- factual failure classification and whether a policy-boundary reassessment is merely a candidate.

A diagnosis never authorizes a threshold change or I4.2.

## Validation gates

Run:

```bash
bash scripts/run_i4_1_1_tests.sh focused
bash scripts/run_i4_1_1_tests.sh quick
bash scripts/run_i4_1_1_tests.sh integration
bash scripts/run_i4_1_1_tests.sh stress
bash scripts/run_i4_1_1_tests.sh affected
```

## Real Horse diagnosis

```bash
rm -rf /tmp/eia_i4_1_1_orbit_alignment_coverage_diagnosis
python scripts/run_i4_1_1_orbit_alignment_coverage_diagnosis.py \
  --i4-dir /tmp/eia_i4_object_orbit_campaign \
  --output-dir /tmp/eia_i4_1_1_orbit_alignment_coverage_diagnosis
```

Do not make a policy decision until the terminal diagnosis is reviewed. Fixed-view camera-set previews are available afterward via `run_i4_1_1_orbit_diagnosis_preview.py`.
