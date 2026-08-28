"""I4.3G2A MVS-vs-DA3 source/evidence binding preflight.

G2A is a read-only qualification preflight.  It binds the frozen G1 logical
G2 scenarios to concrete source identities where those identities are already
known, discovers candidate image directories for scenarios that still need an
engineer-approved semantic binding, and refuses to start MVS/DA3/SfM work.

The preflight deliberately does not infer that an arbitrary directory is a
"complete room", "low texture", or "weak overlap" capture merely from its
name.  Those semantics are qualification inputs and must be bound explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from core.reconstruction.errors import ReconstructionValidationError

I43G2A_SCHEMA_VERSION = 1
I43G2A_PHASE = "I4.3G2A"

_G2_SCENARIOS = (
    "g2_complete_room_mvs_da3",
    "g2_temple_ring_mvs_da3",
    "g2_house_mvs_da3_historical",
    "g2_low_texture_mvs_da3",
    "g2_weak_overlap_mvs_da3",
)
_FRESH_BINDING_IDS = (
    "g2_complete_room_mvs_da3",
    "g2_low_texture_mvs_da3",
    "g2_weak_overlap_mvs_da3",
)
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
_EXPECTED_G1_METRICS = 21
_EXPECTED_G1_PROVIDERS = 10
_EXPECTED_G1_G2_SCENARIOS = 5
_EXPECTED_HOUSE_HISTORICAL_SHA256 = "0fbc46280ec38e571a92c15f48eda9f6103735e6a8915c80798b58a7f2330b50"


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    directory: str
    image_count: int
    asset_set_sha256: str
    files: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "directory": self.directory,
            "image_count": self.image_count,
            "asset_set_sha256": self.asset_set_sha256,
            "files": [dict(item) for item in self.files],
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I4.3G2A requires mapping section: {label}")
    return value




def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    source = path.expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read {label}: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"{label} must contain a JSON object: {source}")
    return payload


def _validate_i4_2f_closure_chain(path: Path) -> dict[str, Any]:
    closure_path = path.expanduser().resolve()
    closure = _read_json_object(closure_path, label="I4.2F closure")
    if closure.get("phase") != "I4.2F" or closure.get("status") != "pass" or closure.get("pass") is not True:
        raise ReconstructionValidationError("TempleRing binding requires a passing I4.2F closure.")
    decision = _require_mapping(closure.get("decision"), label="I4.2F decision")
    authority = _require_mapping(closure.get("authority"), label="I4.2F authority")
    if decision.get("i4_2_closed") is not True or decision.get("i4_2f_closed") is not True or decision.get("i4_3_provider_architecture_authorized") is not True:
        raise ReconstructionValidationError("TempleRing binding requires I4.2F closure and I4.3 authorization.")
    forbidden = (
        "authoritative_sparse_modified", "sfm_rerun", "bundle_adjustment_rerun", "pose_refinement",
        "patch_match_rerun", "stereo_fusion_rerun", "production_filter_policy_changed",
        "dense_provider_promoted", "metric_scale_claimed", "measurement_authoritative",
        "certified_metrology", "surface_mesh_generated",
    )
    drifted = [key for key in forbidden if bool(authority.get(key, False))]
    if drifted:
        raise ReconstructionValidationError(f"I4.2F authority drift: {drifted}")
    return closure


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value.lower())


def _validate_recovered_house_closure(
    *,
    upstream: Mapping[str, Any],
    i4_2a_summary: Path,
    i4_2a_summary_sha256: str,
    verification_method_prefix: str,
    extra_provenance: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate a persistent semantic recovery of the deleted House closure.

    The original I4.1.9A-A3.1 closure lived under ``/tmp`` and its exact bytes
    were later unavailable.  I4.2A deliberately preserved that original SHA
    while also pointing at a persistent recovered semantic copy with a distinct
    byte hash.  G2A must preserve both identities instead of pretending the
    recovered file is byte-identical to the deleted original.
    """

    recovered_path_raw = upstream.get("path")
    recovered_sha = upstream.get("sha256")
    verdict = upstream.get("verdict")
    classification = upstream.get("classification")
    if (
        not isinstance(recovered_path_raw, str)
        or not recovered_path_raw.strip()
        or not _valid_sha256(recovered_sha)
        or verdict != "mvs_better_but_imperfect"
    ):
        return None

    recovered_path = Path(recovered_path_raw).expanduser().resolve()
    if not recovered_path.is_file() or _sha256_file(recovered_path) != recovered_sha:
        return None

    recovered = _read_json_object(recovered_path, label="recovered House I4.1.9A-A3.1 closure")
    if (
        recovered.get("phase") != "I4.1.9A-A3.1"
        or recovered.get("schema_version") != 1
        or recovered.get("status") != "pass"
        or recovered.get("pass") is not True
    ):
        return None

    manual = recovered.get("manual_review")
    if not isinstance(manual, Mapping):
        return None
    if manual.get("confirmed") is not True or manual.get("verdict") != verdict:
        return None
    recovered_classification = manual.get("classification")
    if classification is not None and recovered_classification != classification:
        return None

    recovery = recovered.get("recovery_provenance")
    if not isinstance(recovery, Mapping):
        return None
    if recovery.get("artifact_role") != "recovered_semantic_copy_of_previously_passed_I4.1.9A_A3.1_closure":
        return None
    if recovery.get("original_closure_sha256_recorded_by_I4.2A") != _EXPECTED_HOUSE_HISTORICAL_SHA256:
        return None
    if recovery.get("original_bytes_available") is not False:
        return None

    source_a3 = recovered.get("source_a3")
    if not isinstance(source_a3, Mapping):
        return None
    required_source_assertions = (
        "controlled_comparison_revalidated",
        "independent_auto_fit_forbidden",
        "same_authoritative_sfm_frame",
        "same_house_input",
        "shared_camera_parameters",
    )
    if any(source_a3.get(key) is not True for key in required_source_assertions):
        return None

    authority = recovered.get("authority")
    if not isinstance(authority, Mapping) or authority.get("single_authoritative_sfm_frame") is not True:
        return None
    forbidden_authority = (
        "bundle_adjustment_rerun",
        "camera_geometry_modified",
        "certified_metrology",
        "da3_rerun",
        "dense_provider_promoted",
        "house_k_txt_consumed",
        "measurement_authoritative",
        "metric_scale_claimed",
        "mvs_rerun",
        "patch_match_rerun",
        "pose_refinement",
        "sfm_rerun",
        "stereo_fusion_rerun",
        "surface_mesh_generated",
    )
    if any(bool(authority.get(key, False)) for key in forbidden_authority):
        return None

    return {
        "verification_method": "persistent_recovered_house_closure_semantic_chain",
        "i4_2a_chain_method": verification_method_prefix,
        "i4_2a_summary": str(i4_2a_summary),
        "i4_2a_summary_sha256": i4_2a_summary_sha256,
        "historical_house_closure_sha256": _EXPECTED_HOUSE_HISTORICAL_SHA256,
        "historical_house_original_closure_sha256": _EXPECTED_HOUSE_HISTORICAL_SHA256,
        "historical_house_recovered_closure": str(recovered_path),
        "historical_house_recovered_closure_sha256": recovered_sha,
        "original_bytes_available": False,
        "verdict": verdict,
        "classification": recovered_classification,
        **dict(extra_provenance),
    }


def _historical_house_from_i4_2f_chain(i4_2f_closure: Mapping[str, Any]) -> dict[str, Any] | None:
    """Resolve frozen House A/B provenance through the persistent I4.2F chain.

    I4.2F has existed with two equivalent provenance layouts:

    1. ``frozen_summaries`` embedded directly in ``i4_2f_closure.json``;
    2. the closure's SHA-pinned ``i4_2f_evidence_manifest.json``, where the
       same records live under ``summaries`` (or ``frozen_summaries``).

    G2A supports both without trusting an unverified adjacent file.  It also
    understands the project's explicit recovery contract for the deleted
    original House closure: the recovered semantic copy has its own byte hash
    and must separately preserve the original deleted-closure hash in
    ``recovery_provenance``.
    """

    summary_records: list[tuple[str, Mapping[str, Any], dict[str, Any]]] = []

    direct = i4_2f_closure.get("frozen_summaries")
    if isinstance(direct, Mapping):
        i42a = direct.get("i4_2a_summary")
        if isinstance(i42a, Mapping):
            summary_records.append(("closure", i42a, {}))

    evidence_ref = i4_2f_closure.get("evidence_manifest")
    if isinstance(evidence_ref, Mapping):
        raw_manifest_path = evidence_ref.get("path")
        expected_manifest_sha = evidence_ref.get("sha256")
        if (
            isinstance(raw_manifest_path, str)
            and raw_manifest_path.strip()
            and _valid_sha256(expected_manifest_sha)
        ):
            manifest_path = Path(raw_manifest_path).expanduser().resolve()
            if manifest_path.is_file() and _sha256_file(manifest_path) == expected_manifest_sha:
                manifest = _read_json_object(manifest_path, label="I4.2F evidence manifest")
                for section_name in ("summaries", "frozen_summaries"):
                    section = manifest.get(section_name)
                    if not isinstance(section, Mapping):
                        continue
                    i42a = section.get("i4_2a_summary")
                    if isinstance(i42a, Mapping):
                        summary_records.append((
                            "manifest",
                            i42a,
                            {
                                "i4_2f_evidence_manifest": str(manifest_path),
                                "i4_2f_evidence_manifest_sha256": expected_manifest_sha,
                            },
                        ))
                        break

    # Each summary record contains two byte-addressed routes: the closure-local
    # frozen copy and the persistent source copy from I4.2B.  Prefer frozen
    # bytes, but accept the source copy when cleanup removed the redundant
    # closure-local file.
    for record_origin, i42a, extra_provenance in summary_records:
        candidates = (
            ("frozen_copy", "frozen_sha256", f"persistent_i4_2f_{record_origin}_frozen_i4_2a_chain"),
            ("source", "source_sha256", f"persistent_i4_2f_{record_origin}_source_i4_2a_chain"),
        )
        for path_key, sha_key, method in candidates:
            raw_path = i42a.get(path_key)
            expected_sha = i42a.get(sha_key)
            if not isinstance(raw_path, str) or not raw_path.strip() or not _valid_sha256(expected_sha):
                continue
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file() or _sha256_file(path) != expected_sha:
                continue

            payload = _read_json_object(path, label="I4.2A summary from I4.2F chain")
            upstream = payload.get("upstream_house_mvs_closure")
            if not isinstance(upstream, Mapping):
                continue

            verdict = upstream.get("verdict")
            classification = upstream.get("classification")
            upstream_sha = upstream.get("sha256")
            if verdict != "mvs_better_but_imperfect" or not _valid_sha256(upstream_sha):
                continue

            # Legacy/original-byte provenance: I4.2A itself is content-addressed
            # by I4.2F and preserves the original deleted closure SHA/verdict.
            # Keep this path for backward compatibility without claiming the
            # deleted bytes still exist.
            if upstream_sha == _EXPECTED_HOUSE_HISTORICAL_SHA256:
                return {
                    "verification_method": method,
                    "i4_2a_summary": str(path),
                    "i4_2a_summary_sha256": expected_sha,
                    "historical_house_closure_sha256": upstream_sha,
                    "historical_house_original_closure_sha256": upstream_sha,
                    "original_bytes_available": False,
                    "verdict": verdict,
                    "classification": classification,
                    "original_path": upstream.get("path"),
                    **extra_provenance,
                }

            # Persistent recovered semantic copy: byte identity is intentionally
            # different from the deleted original.  Verify the recovered bytes,
            # then require explicit recovery provenance that points back to the
            # original I4.2A-recorded closure SHA and preserves the same verdict.
            recovered = _validate_recovered_house_closure(
                upstream=upstream,
                i4_2a_summary=path,
                i4_2a_summary_sha256=expected_sha,
                verification_method_prefix=method,
                extra_provenance=extra_provenance,
            )
            if recovered is not None:
                return recovered

    return None

def validate_i4_3g1_for_g2(i4_3g1_snapshot: Mapping[str, Any]) -> None:
    """Fail closed unless the exact frozen G1 contract authorizes G2."""

    if i4_3g1_snapshot.get("phase") != "I4.3G1" or i4_3g1_snapshot.get("status") != "pass" or i4_3g1_snapshot.get("pass") is not True:
        raise ReconstructionValidationError("I4.3G2A requires a passing I4.3G1 snapshot.")

    coverage = _require_mapping(i4_3g1_snapshot.get("coverage"), label="coverage")
    decision = _require_mapping(i4_3g1_snapshot.get("decision"), label="decision")
    authority = _require_mapping(i4_3g1_snapshot.get("authority"), label="authority")
    scope = _require_mapping(i4_3g1_snapshot.get("scope"), label="scope")

    if coverage.get("g2_scenarios") != _EXPECTED_G1_G2_SCENARIOS:
        raise ReconstructionValidationError("I4.3G2A requires exactly five frozen G2 scenarios.")
    if coverage.get("metric_count") != _EXPECTED_G1_METRICS:
        raise ReconstructionValidationError("I4.3G2A refuses G1 metric-registry drift.")
    if coverage.get("provider_implementation_count") != _EXPECTED_G1_PROVIDERS:
        raise ReconstructionValidationError("I4.3G2A refuses G1 provider-registry drift.")
    if decision.get("i4_3g1_framework_closed") is not True or decision.get("i4_3g2_mvs_da3_qualification_authorized") is not True:
        raise ReconstructionValidationError("I4.3G2A requires G1 closure and explicit G2 authorization.")
    if decision.get("global_scalar_score_forbidden") is not True or decision.get("universal_provider_winner_assumed") is not False:
        raise ReconstructionValidationError("I4.3G2A requires the frozen no-leaderboard/no-winner policy.")
    if scope.get("scenario_matrix_frozen") is not True or scope.get("metric_registry_frozen") is not True or scope.get("decision_rules_frozen") is not True:
        raise ReconstructionValidationError("I4.3G2A requires the frozen G1 methodology sections.")

    scenarios = i4_3g1_snapshot.get("scenario_matrix")
    if not isinstance(scenarios, Sequence) or isinstance(scenarios, (str, bytes)):
        raise ReconstructionValidationError("I4.3G2A requires the frozen G1 scenario matrix.")
    g2_ids = [
        item.get("scenario_id")
        for item in scenarios
        if isinstance(item, Mapping) and item.get("substep") == "I4.3G2" and item.get("required") is True
    ]
    if tuple(g2_ids) != _G2_SCENARIOS:
        raise ReconstructionValidationError(f"I4.3G2A G2 scenario-order drift: {g2_ids}")

    if authority.get("camera_pose_authority_owner") != "pycolmap_sfm":
        raise ReconstructionValidationError("I4.3G2A requires PyCOLMAP camera/frame authority.")
    if authority.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError("I4.3G2A refuses a permanent global view cap.")
    forbidden = (
        "authoritative_sfm_modified",
        "provider_selected",
        "provider_promoted",
        "auto_policy_implemented",
        "hybrid_authorized",
        "metric_scale_claimed",
        "measurement_authoritative",
        "certified_metrology",
        "surface_mesh_generated",
    )
    drifted = [key for key in forbidden if bool(authority.get(key, False))]
    if drifted:
        raise ReconstructionValidationError(f"I4.3G2A refuses G1 authority drift: {drifted}")


def build_source_identity(directory: Path, *, minimum_images: int = 3) -> SourceIdentity:
    """Build a deterministic, content-addressed identity for one image directory."""

    root = directory.expanduser().resolve()
    if not root.is_dir():
        raise ReconstructionValidationError(f"Source image directory does not exist: {root}")
    images = sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES),
        key=lambda path: path.name.casefold(),
    )
    if len(images) < int(minimum_images):
        raise ReconstructionValidationError(
            f"Source image directory requires at least {minimum_images} supported images: {root} (found {len(images)})"
        )
    records: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in images:
        sha = _sha256_file(path)
        size = int(path.stat().st_size)
        record = {"name": path.name, "size_bytes": size, "sha256": sha}
        records.append(record)
        aggregate.update(path.name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(sha.encode("ascii"))
        aggregate.update(b"\n")
    return SourceIdentity(str(root), len(records), aggregate.hexdigest(), tuple(records))


def discover_image_directories(
    roots: Iterable[Path],
    *,
    minimum_images: int = 3,
    max_depth: int = 4,
    maximum_candidates: int = 100,
) -> tuple[dict[str, Any], ...]:
    """Read-only discovery of image directories; no semantic classification is inferred."""

    candidates: list[dict[str, Any]] = []
    visited: set[str] = set()
    skip_names = {".git", ".cache", "node_modules", "site-packages", "__pycache__", "checkpoints"}

    for raw_root in roots:
        root = raw_root.expanduser().resolve()
        if not root.is_dir():
            continue
        stack: list[tuple[Path, int]] = [(root, 0)]
        while stack and len(candidates) < maximum_candidates:
            current, depth = stack.pop()
            resolved = str(current.resolve())
            if resolved in visited:
                continue
            visited.add(resolved)
            try:
                entries = list(current.iterdir())
            except OSError:
                continue
            image_files = [p for p in entries if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES]
            if len(image_files) >= minimum_images:
                sample = sorted((p.name for p in image_files), key=str.casefold)[:5]
                candidates.append({
                    "directory": resolved,
                    "image_count": len(image_files),
                    "sample_names": sample,
                    "semantic_binding_inferred": False,
                })
            if depth >= max_depth:
                continue
            subdirs = [p for p in entries if p.is_dir() and not p.is_symlink() and p.name not in skip_names and not p.name.startswith(".")]
            for child in sorted(subdirs, key=lambda p: p.name.casefold(), reverse=True):
                stack.append((child, depth + 1))

    candidates.sort(key=lambda item: (-int(item["image_count"]), str(item["directory"])))
    return tuple(candidates[:maximum_candidates])


def _explicit_binding(scenario_id: str, directory: Path | None) -> dict[str, Any]:
    if directory is None:
        return {
            "scenario_id": scenario_id,
            "status": "needs_explicit_source_binding",
            "ready_for_real_execution": False,
            "semantic_binding_inferred": False,
            "source_identity": None,
        }
    identity = build_source_identity(directory)
    return {
        "scenario_id": scenario_id,
        "status": "bound",
        "ready_for_real_execution": True,
        "semantic_binding_inferred": False,
        "binding_method": "explicit_cli_source_directory",
        "source_identity": identity.as_dict(),
        "authoritative_sfm_binding": "build_once_or_bind_existing_in_g2b_then_share_between_mvs_and_da3",
    }


def build_g2a_preflight_snapshot(
    *,
    i4_3g1_snapshot: Mapping[str, Any],
    temple_ring_dir: Path | None,
    i4_2f_closure: Path | None,
    house_dir: Path | None,
    house_historical_closure: Path | None,
    complete_room_dir: Path | None = None,
    low_texture_dir: Path | None = None,
    weak_overlap_dir: Path | None = None,
    fresh_binding_snapshot: Mapping[str, Any] | None = None,
    discovery_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    validate_i4_3g1_for_g2(i4_3g1_snapshot)

    fresh_by_id: dict[str, dict[str, Any]] = {}
    if fresh_binding_snapshot is not None:
        # Import lazily so the G2A.3 module can reuse the base G2A source
        # identity helpers without creating an import cycle at module load.
        from core.reconstruction.mvs_da3_fresh_scenario_binding import (
            validate_fresh_scenario_binding_snapshot,
        )

        fresh_by_id = validate_fresh_scenario_binding_snapshot(
            fresh_binding_snapshot,
            verify_files=True,
        )

    def fresh_or_explicit(scenario_id: str, directory: Path | None) -> dict[str, Any]:
        frozen = fresh_by_id.get(scenario_id)
        if frozen is None:
            return _explicit_binding(scenario_id, directory)
        return {
            "scenario_id": scenario_id,
            "status": "bound",
            "ready_for_real_execution": True,
            "semantic_binding_inferred": False,
            "binding_method": "frozen_i4_3g2a3_fresh_scenario_binding",
            "source_identity": dict(frozen["source_identity"]),
            "semantic_confirmation": dict(frozen["semantic_confirmation"]),
            "diagnostics": dict(frozen["diagnostics"]),
            "derivation": dict(frozen["derivation"]),
            "visual_review": dict(frozen["visual_review"]),
            "authoritative_sfm_binding": frozen["authoritative_sfm_binding"],
        }

    bindings: list[dict[str, Any]] = [
        fresh_or_explicit("g2_complete_room_mvs_da3", complete_room_dir),
    ]

    # TempleRing is a previously qualified source/authority chain, but G2 must
    # still run fresh provider executions.  Bind only when both source and
    # persistent I4.2F closure evidence are actually present.
    i42f_payload: dict[str, Any] | None = None
    temple_binding: dict[str, Any] = {
        "scenario_id": "g2_temple_ring_mvs_da3",
        "status": "needs_binding",
        "ready_for_real_execution": False,
        "semantic_binding_inferred": False,
    }
    if temple_ring_dir is not None and i4_2f_closure is not None:
        try:
            source = build_source_identity(temple_ring_dir, minimum_images=47)
            closure = i4_2f_closure.expanduser().resolve()
            if source.image_count != 47:
                raise ReconstructionValidationError(f"TempleRing binding requires exactly 47 images; found {source.image_count}.")
            if not closure.is_file():
                raise ReconstructionValidationError(f"TempleRing I4.2F closure is missing: {closure}")
            i42f_payload = _validate_i4_2f_closure_chain(closure)
            temple_binding = {
                "scenario_id": "g2_temple_ring_mvs_da3",
                "status": "bound",
                "ready_for_real_execution": True,
                "semantic_binding_inferred": False,
                "binding_method": "frozen_i4_2_temple_ring_authority_chain",
                "source_identity": source.as_dict(),
                "authority_evidence": {"path": str(closure), "sha256": _sha256_file(closure)},
                "authoritative_sfm_binding": "reuse_or_rehydrate_frozen_i4_2_authoritative_sfm_without_pose_refinement",
            }
        except ReconstructionValidationError as exc:
            temple_binding["binding_error"] = str(exc)
    bindings.append(temple_binding)

    historical: dict[str, Any] = {
        "scenario_id": "g2_house_mvs_da3_historical",
        "status": "needs_historical_evidence_binding",
        "ready_for_real_execution": False,
        "historical_read_only": True,
        "semantic_binding_inferred": False,
        "new_real_execution_allowed": False,
        "expected_closure_sha256": _EXPECTED_HOUSE_HISTORICAL_SHA256,
    }
    if house_dir is not None:
        try:
            source = build_source_identity(house_dir, minimum_images=68)
            if source.image_count != 68:
                raise ReconstructionValidationError(f"Historical House binding requires exactly 68 images; found {source.image_count}.")
            historical["source_identity"] = source.as_dict()
        except ReconstructionValidationError as exc:
            historical["source_binding_error"] = str(exc)
    direct_verified = False
    if house_historical_closure is not None:
        closure = house_historical_closure.expanduser().resolve()
        if closure.is_file():
            actual_sha = _sha256_file(closure)
            historical["historical_evidence"] = {
                "verification_method": "direct_historical_closure",
                "path": str(closure),
                "sha256": actual_sha,
            }
            direct_verified = actual_sha == _EXPECTED_HOUSE_HISTORICAL_SHA256
        else:
            historical["historical_evidence_missing_path"] = str(closure)

    chained = _historical_house_from_i4_2f_chain(i42f_payload) if i42f_payload is not None else None
    if not direct_verified and chained is not None:
        historical["historical_evidence"] = chained
        direct_verified = True
    historical["historical_evidence_verified"] = bool(direct_verified and historical.get("source_identity") is not None)
    if historical["historical_evidence_verified"]:
        historical["status"] = "verified_historical_read_only"
    historical["ready_for_real_execution"] = False
    bindings.append(historical)

    bindings.extend((
        fresh_or_explicit("g2_low_texture_mvs_da3", low_texture_dir),
        fresh_or_explicit("g2_weak_overlap_mvs_da3", weak_overlap_dir),
    ))

    by_id = {item["scenario_id"]: item for item in bindings}
    if tuple(by_id) != _G2_SCENARIOS:
        # Reorder without changing the frozen G1 order.
        bindings = [by_id[item] for item in _G2_SCENARIOS]

    fresh_ready = [
        scenario_id
        for scenario_id in (*_FRESH_BINDING_IDS, "g2_temple_ring_mvs_da3")
        if by_id[scenario_id].get("ready_for_real_execution") is True
    ]
    historical_ready = by_id["g2_house_mvs_da3_historical"].get("historical_evidence_verified") is True
    all_bound = len(fresh_ready) == 4 and historical_ready
    candidates = discover_image_directories(discovery_roots)

    return {
        "schema_version": I43G2A_SCHEMA_VERSION,
        "phase": I43G2A_PHASE,
        "status": "ready" if all_bound else "needs_binding",
        "pass": True,
        "scope": {
            "source_evidence_binding_preflight_only": True,
            "real_provider_execution_started": False,
            "mvs_started": False,
            "da3_started": False,
            "sfm_started": False,
            "quality_results_consumed": False,
            "g1_methodology_modified": False,
            "semantic_scenario_labels_inferred_from_paths": False,
        },
        "scenario_bindings": bindings,
        "discovered_candidate_directories": list(candidates),
        "binding_summary": {
            "g2_scenario_count": 5,
            "fresh_scenario_bindings_ready": len(fresh_ready),
            "historical_binding_verified": historical_ready,
            "all_g2_sources_bound": all_bound,
            "missing_explicit_fresh_bindings": [
                scenario_id for scenario_id in _FRESH_BINDING_IDS if by_id[scenario_id].get("ready_for_real_execution") is not True
            ],
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
        "decision": {
            "i4_3g2a_preflight_complete": True,
            "i4_3g2b_real_execution_authorized": all_bound,
            "source_binding_required_before_heavy_execution": True,
            "provider_selected": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "next": (
            "I4.3G2B real production-native MVS/DA3 executions."
            if all_bound
            else "Bind the missing complete-room/low-texture/weak-overlap source identities, then authorize I4.3G2B."
        ),
    }


__all__ = [
    "I43G2A_SCHEMA_VERSION",
    "I43G2A_PHASE",
    "SourceIdentity",
    "validate_i4_3g1_for_g2",
    "build_source_identity",
    "discover_image_directories",
    "build_g2a_preflight_snapshot",
]
