# I6.7F — South Building fusion-only recovery (separate evidence)

## Scope

The original South Building COLMAP MVS run was terminated by the kernel OOM killer at stereo fusion. Its original PyCOLMAP request and state are pinned by SHA256 in the accompanying recovery script. This standalone recovery tool **does not** modify the original worker, original SfM, original depth/normal maps, original state, or failed run ID. It does not rerun undistortion, PatchMatch, SfM, bundle adjustment, or any neural provider; it calls only geometric `pycolmap.stereo_fusion()` on the existing dense workspace and writes a PLY plus visibility file under `dense_runs/<original-run-id>/fusion_recoveries/<new-recovery-id>/`.

This is **not** the application's normal final-admission flow. Even a verified fusion-only PLY must subsequently pass a separately qualified adapter/envelope/full admission gate before it may appear in the authoritative full-density viewer. Do not move this PLY into the original run or manufacture `final_dense_admission.json`.

## Memory and execution policy

The previously failed run used `use_cache=False`, unrestricted `num_threads=-1`, and full-image geometric fusion. Recovery preserves `use_cache=False`, all scientific thresholds, all 87 views, and `max_image_size=-1`, but requests **two** CPU fusion threads (configurable to 1–4). The supervisor samples worker RSS, system available RAM, and free disk space every second. Defaults: stop worker if its RSS reaches 8 GiB, system available RAM falls below 3 GiB, disk drops below 10 GiB, or 240 minutes elapse. This sampling **reduces** but cannot eliminate OOM risk or guarantee successful completion; memory can spike between samples. The first attempt should start after closing RAM-heavy applications, in a normal terminal rather than VS Code.

## Limitations of the inspection material

The inspection ZIP contains the original worker and admission/verification code but not the full `MvsProductionProviderAdapter`, the typed E2A handoff code, or the recovery launcher's original source. The recovery tool therefore stops at independently checking a fusion-only point cloud and writes `final_dense_result_admitted: false`. It does not assert a provider envelope, persistent scene, or metric scale. Structural checks verify the exact 348 COLMAP depth/normal files, dimensions, and untruncated byte lengths, plus pinned source sparse and registered-image SHA256. The original per-map cryptographic hashes were not recorded at stereo generation; structural checks do not prove scientific validity of every pixel. The original PyCOLMAP sparse/dense pose and point geometry comparison and PatchMatch registered source/reference configuration are also revalidated before fusion.

## Usage

Unzip only the new recovery script, tests, and documentation into the existing repo after checking the package SHA256. Run `python -m pytest -q tests/test_i6_7f_south_fusion_only_recovery.py` if pytest is available and run `--check-only` with original pinned request, state, baseline worker, and a new UUID. Do not proceed if any checksum or preflight fails. In the launch command, use `nohup setsid python -u` outside VS Code and preserve the supervisor log/recovery ID. Never launch two recovery attempts concurrently; never overwrite an existing recovery directory.
