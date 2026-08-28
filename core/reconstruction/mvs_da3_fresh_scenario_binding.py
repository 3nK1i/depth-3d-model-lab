"""I4.3G2A.3 fresh-scenario binding and evidence freeze.

This module binds the three still-unresolved G2 scenarios without executing
SfM, MVS, DA3, Auto, or hybrid logic.  A complete-room source requires an
explicit engineer semantic confirmation.  Low-texture and weak-overlap
bindings additionally require objective, pre-provider diagnostics and an
explicit confirmation; they may either use separate real captures or
controlled subsets derived from the confirmed complete-room source.

The binding artifact is content-addressed and is later consumed by the normal
I4.3G2A preflight.  Provider-quality results are not consumed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import ceil, isfinite, log1p
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from core.reconstruction.errors import ReconstructionValidationError

I43G2A3_SCHEMA_VERSION = 2
I43G2A3_PHASE = "I4.3G2A.3"
LOW_TEXTURE_AMENDMENT_ID = "I4.3G2A.3-A1"
LOW_TEXTURE_METHOD = "raw_component_ratio_gate_v2"
LEGACY_LOW_TEXTURE_METHOD = "log1p_aggregate_proxy_ratio_v1"

_COMPLETE_ROOM_ID = "g2_complete_room_mvs_da3"
_LOW_TEXTURE_ID = "g2_low_texture_mvs_da3"
_WEAK_OVERLAP_ID = "g2_weak_overlap_mvs_da3"
_FRESH_IDS = (_COMPLETE_ROOM_ID, _LOW_TEXTURE_ID, _WEAK_OVERLAP_ID)
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})


@dataclass(frozen=True, slots=True)
class FreshBindingPolicy:
    """Frozen G2A.3 diagnostic policy; never tuned from provider results."""

    minimum_source_images: int = 12
    minimum_subset_images: int = 8
    weak_overlap_minimum_subset_images: int = 6
    maximum_subset_images: int = 24
    low_texture_fraction: float = 0.30
    # Retained only to preserve the retired v1 gate in provenance.  A1 no
    # longer uses the ratio of the log-compressed aggregate proxy to qualify
    # low-texture sources.
    low_texture_max_median_ratio_to_reference: float = 0.75
    # A1 applies the same 0.75 material-reduction criterion to raw component
    # ratios where percentage comparisons are mathematically meaningful.
    low_texture_max_component_ratio_to_reference: float = 0.75
    weak_overlap_minimum_stride: int = 2
    weak_overlap_max_median_ratio_to_reference_for_separate_capture: float = 0.70
    weak_overlap_min_good_matches_per_pair: int = 8
    weak_overlap_min_connected_pair_fraction: float = 0.80
    weak_overlap_minimum_pairs: int = 5
    orb_nfeatures: int = 1400
    orb_ratio_test: float = 0.75
    diagnostic_max_dimension: int = 720
    maximum_stride: int = 8

    def as_dict(self) -> dict[str, Any]:
        return {
            "minimum_source_images": self.minimum_source_images,
            "minimum_subset_images": self.minimum_subset_images,
            "weak_overlap_minimum_subset_images": self.weak_overlap_minimum_subset_images,
            "maximum_subset_images": self.maximum_subset_images,
            "low_texture_fraction": self.low_texture_fraction,
            "low_texture_max_median_ratio_to_reference": self.low_texture_max_median_ratio_to_reference,
            "low_texture_max_component_ratio_to_reference": self.low_texture_max_component_ratio_to_reference,
            "weak_overlap_minimum_stride": self.weak_overlap_minimum_stride,
            "weak_overlap_max_median_ratio_to_reference_for_separate_capture": self.weak_overlap_max_median_ratio_to_reference_for_separate_capture,
            "weak_overlap_min_good_matches_per_pair": self.weak_overlap_min_good_matches_per_pair,
            "weak_overlap_min_connected_pair_fraction": self.weak_overlap_min_connected_pair_fraction,
            "weak_overlap_minimum_pairs": self.weak_overlap_minimum_pairs,
            "orb_nfeatures": self.orb_nfeatures,
            "orb_ratio_test": self.orb_ratio_test,
            "diagnostic_max_dimension": self.diagnostic_max_dimension,
            "maximum_stride": self.maximum_stride,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticFrame:
    name: str
    width: int
    height: int
    texture_proxy: float
    laplacian_variance: float
    contrast_stddev: float
    edge_density: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "texture_proxy": self.texture_proxy,
            "laplacian_variance": self.laplacian_variance,
            "contrast_stddev": self.contrast_stddev,
            "edge_density": self.edge_density,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value.lower())


def _image_paths(directory: Path) -> tuple[Path, ...]:
    root = directory.expanduser().resolve()
    if not root.is_dir():
        raise ReconstructionValidationError(f"G2A.3 image directory does not exist: {root}")
    images = tuple(sorted(
        (p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES),
        key=lambda p: p.name.casefold(),
    ))
    return images


def _aggregate_identity(root: Path, images: Sequence[Path]) -> dict[str, Any]:
    if not images:
        raise ReconstructionValidationError("G2A.3 source identity cannot be empty.")
    aggregate = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for path in images:
        resolved = path.expanduser().resolve()
        if resolved.parent != root:
            raise ReconstructionValidationError("G2A.3 subset image must remain inside its declared source directory.")
        sha = _sha256_file(resolved)
        size = int(resolved.stat().st_size)
        record = {"name": resolved.name, "size_bytes": size, "sha256": sha}
        records.append(record)
        aggregate.update(resolved.name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(sha.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "directory": str(root),
        "image_count": len(records),
        "asset_set_sha256": aggregate.hexdigest(),
        "files": records,
    }


def build_exact_source_identity(directory: Path, *, names: Sequence[str] | None = None, minimum_images: int = 3) -> dict[str, Any]:
    root = directory.expanduser().resolve()
    all_images = _image_paths(root)
    by_name = {path.name: path for path in all_images}
    if names is None:
        chosen = all_images
    else:
        if len(set(names)) != len(tuple(names)):
            raise ReconstructionValidationError("G2A.3 subset contains duplicate image names.")
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ReconstructionValidationError(f"G2A.3 subset references missing images: {missing[:5]}")
        chosen = tuple(by_name[name] for name in names)
    if len(chosen) < int(minimum_images):
        raise ReconstructionValidationError(
            f"G2A.3 source requires at least {minimum_images} images; found {len(chosen)} in {root}"
        )
    return _aggregate_identity(root, chosen)


def _read_gray(path: Path, *, max_dimension: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2 or image.size <= 0:
        raise ReconstructionValidationError(f"G2A.3 cannot decode image: {path}")
    height, width = image.shape[:2]
    scale = min(1.0, float(max_dimension) / float(max(height, width)))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def _texture_diagnostic(path: Path, *, max_dimension: int) -> DiagnosticFrame:
    gray = _read_gray(path, max_dimension=max_dimension)
    laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    edges = cv2.Canny(gray, 60, 180)
    edge_density = float(np.mean(edges > 0))
    proxy = float(log1p(max(laplacian, 0.0)) + log1p(max(contrast, 0.0)) + 3.0 * edge_density)
    if not all(isfinite(value) and value >= 0.0 for value in (laplacian, contrast, edge_density, proxy)):
        raise ReconstructionValidationError(f"G2A.3 produced invalid texture diagnostics for {path}")
    height, width = gray.shape[:2]
    return DiagnosticFrame(
        name=path.name,
        width=int(width),
        height=int(height),
        texture_proxy=proxy,
        laplacian_variance=laplacian,
        contrast_stddev=contrast,
        edge_density=edge_density,
    )


def analyze_texture(directory: Path, *, policy: FreshBindingPolicy = FreshBindingPolicy()) -> tuple[DiagnosticFrame, ...]:
    images = _image_paths(directory)
    if len(images) < policy.minimum_source_images:
        raise ReconstructionValidationError(
            f"G2A.3 texture analysis requires at least {policy.minimum_source_images} images; found {len(images)}"
        )
    return tuple(_texture_diagnostic(path, max_dimension=policy.diagnostic_max_dimension) for path in images)


def _component_medians(frames: Sequence[DiagnosticFrame]) -> dict[str, float]:
    if not frames:
        raise ReconstructionValidationError("G2A.3 low-texture component analysis cannot use an empty frame set.")
    return {
        "laplacian_variance": float(median(item.laplacian_variance for item in frames)),
        "contrast_stddev": float(median(item.contrast_stddev for item in frames)),
        "edge_density": float(median(item.edge_density for item in frames)),
    }


def _positive_reference_ratio(candidate: float, reference: float, *, label: str) -> float:
    if not isfinite(reference) or reference <= 0.0:
        raise ReconstructionValidationError(f"G2A.3 complete-room {label} reference is degenerate.")
    ratio = float(candidate / reference)
    if not isfinite(ratio) or ratio < 0.0:
        raise ReconstructionValidationError(f"G2A.3 produced invalid {label} ratio.")
    return ratio


def _low_texture_amendment_record(policy: FreshBindingPolicy) -> dict[str, Any]:
    return {
        "amendment_id": LOW_TEXTURE_AMENDMENT_ID,
        "status": "active",
        "predecessor_method": LEGACY_LOW_TEXTURE_METHOD,
        "predecessor_threshold": policy.low_texture_max_median_ratio_to_reference,
        "predecessor_retired_for_qualification": True,
        "reason": (
            "Known low-texture positive controls showed that ratios of the log1p-compressed "
            "aggregate proxy materially under-represent raw reductions in Laplacian variance, "
            "contrast, and edge density."
        ),
        "replacement_method": LOW_TEXTURE_METHOD,
        "replacement_threshold": policy.low_texture_max_component_ratio_to_reference,
        "qualification_rule": (
            "edge_density_ratio <= threshold AND "
            "(laplacian_variance_ratio <= threshold OR contrast_stddev_ratio <= threshold)"
        ),
        "legacy_proxy_retained_for_diagnostics_only": True,
        "provider_quality_results_consumed_before_amendment": False,
        "real_provider_execution_started_before_amendment": False,
        "sfm_started_before_amendment": False,
        "mvs_started_before_amendment": False,
        "da3_started_before_amendment": False,
    }


def _low_texture_component_diagnostic(
    *,
    reference_frames: Sequence[DiagnosticFrame],
    candidate_frames: Sequence[DiagnosticFrame],
    policy: FreshBindingPolicy,
    selection: str,
    contiguous_source_order_preserved: bool,
) -> dict[str, Any]:
    reference_components = _component_medians(reference_frames)
    candidate_components = _component_medians(candidate_frames)
    ratios = {
        name: _positive_reference_ratio(candidate_components[name], reference_components[name], label=name)
        for name in ("laplacian_variance", "contrast_stddev", "edge_density")
    }
    threshold = float(policy.low_texture_max_component_ratio_to_reference)
    lap_ok = ratios["laplacian_variance"] <= threshold
    contrast_ok = ratios["contrast_stddev"] <= threshold
    edge_ok = ratios["edge_density"] <= threshold
    supporting_ratio = min(ratios["laplacian_variance"], ratios["contrast_stddev"])
    qualification_margin = max(ratios["edge_density"], supporting_ratio)
    qualified = bool(edge_ok and (lap_ok or contrast_ok))

    reference_proxy = float(median(item.texture_proxy for item in reference_frames))
    candidate_proxy = float(median(item.texture_proxy for item in candidate_frames))
    legacy_ratio = _positive_reference_ratio(candidate_proxy, reference_proxy, label="legacy texture proxy")

    return {
        "selection": selection,
        "subset_image_count": len(candidate_frames),
        "methodology": LOW_TEXTURE_METHOD,
        "methodology_amendment_id": LOW_TEXTURE_AMENDMENT_ID,
        "reference_component_medians": reference_components,
        "candidate_component_medians": candidate_components,
        "component_ratios_to_reference": ratios,
        "maximum_component_ratio": threshold,
        "laplacian_variance_qualified": bool(lap_ok),
        "contrast_stddev_qualified": bool(contrast_ok),
        "edge_density_qualified": bool(edge_ok),
        "qualification_margin_ratio": float(qualification_margin),
        "objective_proxy_qualified": qualified,
        "contiguous_source_order_preserved": bool(contiguous_source_order_preserved),
        # Backward-compatible diagnostic aliases.  They are deliberately not
        # used by A1 for low-texture qualification.
        "full_source_median_texture_proxy": reference_proxy,
        "subset_median_texture_proxy": candidate_proxy,
        "median_ratio_to_reference": legacy_ratio,
        "maximum_qualified_ratio": policy.low_texture_max_median_ratio_to_reference,
        "legacy_texture_proxy": {
            "methodology": LEGACY_LOW_TEXTURE_METHOD,
            "reference_median": reference_proxy,
            "candidate_median": candidate_proxy,
            "median_ratio_to_reference": legacy_ratio,
            "maximum_qualified_ratio": policy.low_texture_max_median_ratio_to_reference,
            "authoritative_for_qualification": False,
        },
    }


def _choose_low_texture_names(
    frames: Sequence[DiagnosticFrame], *, policy: FreshBindingPolicy
) -> tuple[tuple[str, ...], dict[str, Any]]:
    count = len(frames)
    subset_count = max(policy.minimum_subset_images, int(ceil(count * policy.low_texture_fraction)))
    subset_count = min(policy.maximum_subset_images, subset_count, count)
    if subset_count < policy.minimum_subset_images:
        raise ReconstructionValidationError("G2A.3 cannot form the minimum low-texture subset.")

    best_start = 0
    best_diag: dict[str, Any] | None = None
    best_key: tuple[float, float, float, int] | None = None
    for start in range(0, count - subset_count + 1):
        window = frames[start:start + subset_count]
        diag = _low_texture_component_diagnostic(
            reference_frames=frames,
            candidate_frames=window,
            policy=policy,
            selection="lowest_component_gate_margin_contiguous_texture_window",
            contiguous_source_order_preserved=True,
        )
        ratios = diag["component_ratios_to_reference"]
        key = (
            float(diag["qualification_margin_ratio"]),
            float(ratios["edge_density"]),
            min(float(ratios["laplacian_variance"]), float(ratios["contrast_stddev"])),
            start,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_start = start
            best_diag = diag

    if best_diag is None:
        raise ReconstructionValidationError("G2A.3 could not evaluate a low-texture contiguous subset.")
    selected = tuple(item.name for item in frames[best_start:best_start + subset_count])
    return selected, best_diag


def _orb_descriptor(path: Path, *, policy: FreshBindingPolicy) -> tuple[int, np.ndarray | None]:
    gray = _read_gray(path, max_dimension=policy.diagnostic_max_dimension)
    orb = cv2.ORB_create(nfeatures=int(policy.orb_nfeatures))
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return len(keypoints or ()), descriptors


def _pair_overlap(left: tuple[int, np.ndarray | None], right: tuple[int, np.ndarray | None], *, ratio_test: float) -> tuple[float, int]:
    left_count, left_desc = left
    right_count, right_desc = right
    denominator = min(left_count, right_count)
    if denominator <= 0 or left_desc is None or right_desc is None or len(left_desc) < 2 or len(right_desc) < 2:
        return 0.0, 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = matcher.knnMatch(left_desc, right_desc, k=2)
    good = 0
    for pair in pairs:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < float(ratio_test) * second.distance:
            good += 1
    return float(good / denominator), int(good)


def _overlap_sequence(names: Sequence[str], descriptor_by_name: Mapping[str, tuple[int, np.ndarray | None]], *, policy: FreshBindingPolicy) -> dict[str, Any]:
    ratios: list[float] = []
    good_counts: list[int] = []
    for left_name, right_name in zip(names, names[1:]):
        ratio, good = _pair_overlap(
            descriptor_by_name[left_name], descriptor_by_name[right_name], ratio_test=policy.orb_ratio_test
        )
        ratios.append(ratio)
        good_counts.append(good)
    if not ratios:
        raise ReconstructionValidationError("G2A.3 overlap diagnostic requires at least two images.")
    return {
        "pair_count": len(ratios),
        "median_pair_overlap_proxy": float(median(ratios)),
        "minimum_pair_overlap_proxy": float(min(ratios)),
        "median_good_matches": float(median(good_counts)),
        "minimum_good_matches": int(min(good_counts)),
        "connected_pair_fraction": float(
            sum(count >= policy.weak_overlap_min_good_matches_per_pair for count in good_counts) / len(good_counts)
        ),
    }


def _bounded_stride_names(names: Sequence[str], *, stride: int, offset: int, maximum_images: int) -> tuple[str, ...]:
    selected = tuple(names[offset::stride])
    if len(selected) <= maximum_images:
        return selected
    indices = np.linspace(0, len(selected) - 1, num=maximum_images, dtype=np.int64)
    return tuple(selected[int(index)] for index in indices)


def analyze_and_choose_weak_overlap(
    directory: Path,
    *,
    policy: FreshBindingPolicy = FreshBindingPolicy(),
    excluded_names: Iterable[str] = (),
) -> tuple[tuple[str, ...], dict[str, Any]]:
    images = _image_paths(directory)
    if len(images) < policy.minimum_source_images:
        raise ReconstructionValidationError(
            f"G2A.3 weak-overlap analysis requires at least {policy.minimum_source_images} images; found {len(images)}"
        )
    names = tuple(path.name for path in images)
    excluded = set(excluded_names)
    descriptors = {path.name: _orb_descriptor(path, policy=policy) for path in images}
    baseline = _overlap_sequence(names, descriptors, policy=policy)
    baseline_median = float(baseline["median_pair_overlap_proxy"])
    if baseline_median <= 0.0:
        raise ReconstructionValidationError("G2A.3 complete-room adjacent-overlap reference is degenerate.")

    candidates: list[tuple[float, int, int, tuple[str, ...], dict[str, Any]]] = []
    segments: list[tuple[str, ...]] = []
    current: list[str] = []
    for name in names:
        if name in excluded:
            if current:
                segments.append(tuple(current))
                current = []
            continue
        current.append(name)
    if current:
        segments.append(tuple(current))
    for segment_index, segment in enumerate(segments):
        if len(segment) < policy.weak_overlap_minimum_subset_images:
            continue
        max_stride = min(policy.maximum_stride, max(2, len(segment) // policy.weak_overlap_minimum_subset_images))
        for stride in range(2, max_stride + 1):
            for offset in range(stride):
                selected = _bounded_stride_names(
                    segment, stride=stride, offset=offset, maximum_images=policy.maximum_subset_images
                )
                if len(selected) < policy.weak_overlap_minimum_subset_images:
                    continue
                diag = _overlap_sequence(selected, descriptors, policy=policy)
                if int(diag["pair_count"]) < policy.weak_overlap_minimum_pairs:
                    continue
                relative = float(diag["median_pair_overlap_proxy"]) / baseline_median
                connected = (
                    float(diag["connected_pair_fraction"]) >= policy.weak_overlap_min_connected_pair_fraction
                    and float(diag["median_good_matches"]) >= policy.weak_overlap_min_good_matches_per_pair
                )
                diag = {
                    **diag,
                    "stride": stride,
                    "offset": offset,
                    "source_segment_index": segment_index,
                    "excluded_confounded_names": sorted(excluded),
                    "reference_median_pair_overlap_proxy": baseline_median,
                    "median_ratio_to_reference": relative,
                    "maximum_qualified_ratio": policy.weak_overlap_max_median_ratio_to_reference_for_separate_capture,
                    "minimum_required_good_matches_per_pair": policy.weak_overlap_min_good_matches_per_pair,
                    "minimum_connected_pair_fraction": policy.weak_overlap_min_connected_pair_fraction,
                    "minimum_required_stride": policy.weak_overlap_minimum_stride,
                    "capture_spacing_factor": stride,
                    "connectivity_proxy_qualified": connected,
                    "objective_proxy_qualified": bool(
                        connected and stride >= policy.weak_overlap_minimum_stride
                    ),
                    "source_order_preserved": True,
                }
                candidates.append((relative, -int(diag["minimum_good_matches"]), stride, selected, diag))
    if not candidates:
        raise ReconstructionValidationError("G2A.3 could not form any weak-overlap subset candidate.")
    candidates.sort(key=lambda item: (not item[4]["objective_proxy_qualified"], -item[2], item[0], item[1]))
    _, _, _, selected, diagnostic = candidates[0]
    return selected, {"selection": "ordered_strided_subset_with_orb_overlap_proxy", "reference": baseline, **diagnostic}


def _contact_sheet(directory: Path, names: Sequence[str], target: Path, *, title: str) -> dict[str, Any]:
    root = directory.expanduser().resolve()
    by_name = {path.name: path for path in _image_paths(root)}
    sampled = tuple(names)
    if len(sampled) > 12:
        indices = np.linspace(0, len(sampled) - 1, num=12, dtype=np.int64)
        sampled = tuple(sampled[int(index)] for index in indices)
    cells: list[np.ndarray] = []
    cell_w, cell_h = 320, 220
    for name in sampled:
        image = cv2.imread(str(by_name[name]), cv2.IMREAD_COLOR)
        if image is None:
            raise ReconstructionValidationError(f"G2A.3 cannot build contact sheet from {by_name[name]}")
        h, w = image.shape[:2]
        scale = min(cell_w / max(1, w), (cell_h - 28) / max(1, h))
        resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        y = 22 + max(0, (cell_h - 22 - resized.shape[0]) // 2)
        x = max(0, (cell_w - resized.shape[1]) // 2)
        cell[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.putText(cell, name[:36], (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(cell)
    columns = 3
    rows: list[np.ndarray] = []
    for start in range(0, len(cells), columns):
        row_cells = cells[start:start + columns]
        while len(row_cells) < columns:
            row_cells.append(np.zeros_like(cells[0]))
        rows.append(np.hstack(row_cells))
    canvas = np.vstack(rows)
    header = np.zeros((42, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    canvas = np.vstack((header, canvas))
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), canvas):
        raise ReconstructionValidationError(f"G2A.3 could not write contact sheet: {target}")
    return {"path": str(target.resolve()), "sha256": _sha256_file(target), "sample_names": list(sampled)}


def _binding_record(
    *,
    scenario_id: str,
    identity: Mapping[str, Any],
    confirmation: bool,
    confirmation_label: str,
    diagnostics: Mapping[str, Any],
    derivation: Mapping[str, Any],
    review: Mapping[str, Any],
    objective_required: bool,
) -> dict[str, Any]:
    objective_ok = diagnostics.get("objective_proxy_qualified") is True if objective_required else True
    qualified = bool(confirmation and objective_ok)
    return {
        "scenario_id": scenario_id,
        "status": "bound" if qualified else "needs_confirmation_or_qualification",
        "ready_for_real_execution": qualified,
        "semantic_binding_inferred": False,
        "semantic_confirmation": {
            "required": True,
            "confirmed": bool(confirmation),
            "label": confirmation_label,
        },
        "source_identity": dict(identity),
        "diagnostics": dict(diagnostics),
        "derivation": dict(derivation),
        "visual_review": dict(review),
        "authoritative_sfm_binding": "build_once_in_g2b_then_share_same_authoritative_sfm_between_mvs_and_da3",
    }


def build_fresh_scenario_binding_snapshot(
    *,
    complete_room_dir: Path,
    output_dir: Path,
    confirm_complete_room: bool,
    confirm_low_texture: bool,
    confirm_weak_overlap: bool,
    low_texture_dir: Path | None = None,
    weak_overlap_dir: Path | None = None,
    policy: FreshBindingPolicy = FreshBindingPolicy(),
) -> dict[str, Any]:
    """Create the frozen G2A.3 binding artifact without starting reconstruction."""

    out = output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    complete_root = complete_room_dir.expanduser().resolve()
    complete_images = _image_paths(complete_root)
    if len(complete_images) < policy.minimum_source_images:
        raise ReconstructionValidationError(
            f"Complete-room candidate needs >= {policy.minimum_source_images} images; found {len(complete_images)}"
        )
    complete_names = tuple(path.name for path in complete_images)
    complete_identity = build_exact_source_identity(
        complete_root, names=complete_names, minimum_images=policy.minimum_source_images
    )
    complete_texture = analyze_texture(complete_root, policy=policy)
    complete_review = _contact_sheet(
        complete_root, complete_names, out / "visual_review" / "complete_room_contact_sheet.jpg",
        title="G2A.3 complete-room candidate — engineer confirmation required",
    )
    complete_diag = {
        "image_count": len(complete_names),
        "median_texture_proxy": float(median(item.texture_proxy for item in complete_texture)),
        "objective_proxy_qualified": True,
        "note": "Complete-room semantics are not inferred from pixels or path names; explicit engineer visual confirmation is mandatory.",
    }
    complete_binding = _binding_record(
        scenario_id=_COMPLETE_ROOM_ID,
        identity=complete_identity,
        confirmation=confirm_complete_room,
        confirmation_label="complete_room_static_multiview_capture",
        diagnostics=complete_diag,
        derivation={"mode": "entire_explicit_source_directory"},
        review=complete_review,
        objective_required=False,
    )

    if low_texture_dir is None:
        low_names, low_diag = _choose_low_texture_names(complete_texture, policy=policy)
        low_root = complete_root
        low_derivation = {
            "mode": "controlled_natural_subset_from_complete_room",
            "parent_asset_set_sha256": complete_identity["asset_set_sha256"],
            "policy_frozen_before_provider_results": True,
        }
    else:
        low_root = low_texture_dir.expanduser().resolve()
        low_frames = analyze_texture(low_root, policy=policy)
        low_names = tuple(item.name for item in low_frames)
        low_diag = _explicit_low_texture_diagnostic(
            reference_frames=complete_texture,
            candidate_frames=low_frames,
            policy=policy,
        )
        low_derivation = {"mode": "separate_explicit_real_capture", "policy_frozen_before_provider_results": True}
    low_identity = build_exact_source_identity(low_root, names=low_names, minimum_images=policy.minimum_subset_images)
    low_review = _contact_sheet(
        low_root, low_names, out / "visual_review" / "low_texture_contact_sheet.jpg",
        title="G2A.3 low-texture candidate — objective proxy + engineer confirmation",
    )
    low_binding = _binding_record(
        scenario_id=_LOW_TEXTURE_ID,
        identity=low_identity,
        confirmation=confirm_low_texture,
        confirmation_label="static_multiview_low_texture",
        diagnostics=low_diag,
        derivation=low_derivation,
        review=low_review,
        objective_required=True,
    )

    if weak_overlap_dir is None:
        weak_exclusions = low_names if low_root == complete_root else ()
        try:
            weak_names, weak_diag = analyze_and_choose_weak_overlap(
                complete_root,
                policy=policy,
                excluded_names=weak_exclusions,
            )
        except ReconstructionValidationError as exc:
            weak_names = tuple(complete_names[: policy.weak_overlap_minimum_subset_images])
            weak_diag = {
                "selection": "no_qualified_controlled_weak_overlap_candidate",
                "objective_proxy_qualified": False,
                "diagnostic_error": str(exc),
                "authoritative_sfm_overlap_must_be_revalidated_in_g2b": True,
            }
        weak_root = complete_root
        weak_derivation = {
            "mode": "controlled_ordered_strided_subset_from_complete_room",
            "parent_asset_set_sha256": complete_identity["asset_set_sha256"],
            "authoritative_sfm_overlap_must_be_revalidated_in_g2b": True,
            "policy_frozen_before_provider_results": True,
        }
    else:
        weak_root = weak_overlap_dir.expanduser().resolve()
        weak_images = _image_paths(weak_root)
        weak_names = tuple(path.name for path in weak_images)
        weak_diag = _explicit_weak_overlap_diagnostic(
            reference_dir=complete_root,
            candidate_dir=weak_root,
            policy=policy,
        )
        weak_derivation = {
            "mode": "separate_explicit_real_capture",
            "authoritative_sfm_overlap_must_be_revalidated_in_g2b": True,
            "policy_frozen_before_provider_results": True,
        }
    weak_identity = build_exact_source_identity(
        weak_root,
        names=weak_names,
        minimum_images=policy.weak_overlap_minimum_subset_images,
    )
    weak_review = _contact_sheet(
        weak_root, weak_names, out / "visual_review" / "weak_overlap_contact_sheet.jpg",
        title="G2A.3 weak-overlap candidate — proxy must be revalidated by SfM in G2B",
    )
    weak_binding = _binding_record(
        scenario_id=_WEAK_OVERLAP_ID,
        identity=weak_identity,
        confirmation=confirm_weak_overlap,
        confirmation_label="static_multiview_weak_overlap",
        diagnostics=weak_diag,
        derivation=weak_derivation,
        review=weak_review,
        objective_required=True,
    )

    bindings = [complete_binding, low_binding, weak_binding]
    ready = all(item["ready_for_real_execution"] is True for item in bindings)
    return {
        "schema_version": I43G2A3_SCHEMA_VERSION,
        "phase": I43G2A3_PHASE,
        "status": "ready" if ready else "needs_confirmation_or_qualification",
        "pass": True,
        "policy": policy.as_dict(),
        "methodology_amendments": {
            "low_texture": _low_texture_amendment_record(policy),
        },
        "scope": {
            "fresh_scenario_binding_only": True,
            "source_pixels_read_for_diagnostics": True,
            "provider_quality_results_consumed": False,
            "real_provider_execution_started": False,
            "sfm_started": False,
            "mvs_started": False,
            "da3_started": False,
            "semantic_labels_inferred_from_paths": False,
        },
        "fresh_bindings": bindings,
        "binding_summary": {
            "required_fresh_bindings": 3,
            "ready_fresh_bindings": sum(item["ready_for_real_execution"] is True for item in bindings),
            "all_fresh_bindings_ready": ready,
            "missing_or_unqualified": [item["scenario_id"] for item in bindings if item["ready_for_real_execution"] is not True],
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
            "i4_3g2a3_fresh_binding_ready": ready,
            "g2b_authorized_by_this_artifact_alone": False,
            "final_g2a_preflight_must_reverify_all_sources": True,
        },
        "next": (
            "Rerun I4.3G2A with this frozen fresh-binding snapshot; only that combined preflight may authorize G2B."
            if ready
            else "Review the contact sheets/objective diagnostics and provide the missing engineer confirmations or a better explicit source."
        ),
    }


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"G2A.3 requires mapping: {label}")
    return value


def _verify_identity(identity: Mapping[str, Any]) -> None:
    directory = Path(str(identity.get("directory", ""))).expanduser().resolve()
    files = identity.get("files")
    if not directory.is_dir() or not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
        raise ReconstructionValidationError("G2A.3 source identity is malformed or missing its source directory.")
    names: list[str] = []
    for record in files:
        if not isinstance(record, Mapping):
            raise ReconstructionValidationError("G2A.3 source identity contains a malformed file record.")
        name = record.get("name")
        expected_sha = record.get("sha256")
        expected_size = record.get("size_bytes")
        if not isinstance(name, str) or not _valid_sha256(expected_sha) or not isinstance(expected_size, int):
            raise ReconstructionValidationError("G2A.3 source identity file record is incomplete.")
        path = (directory / name).resolve()
        if path.parent != directory or not path.is_file():
            raise ReconstructionValidationError(f"G2A.3 bound source file is missing: {path}")
        if int(path.stat().st_size) != expected_size or _sha256_file(path) != expected_sha:
            raise ReconstructionValidationError(f"G2A.3 bound source file changed after binding: {path}")
        names.append(name)
    rebuilt = build_exact_source_identity(directory, names=names, minimum_images=3)
    if rebuilt["asset_set_sha256"] != identity.get("asset_set_sha256") or rebuilt["image_count"] != identity.get("image_count"):
        raise ReconstructionValidationError("G2A.3 aggregate source identity mismatch.")


def _identity_names(identity: Mapping[str, Any]) -> tuple[str, ...]:
    files = identity.get("files")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
        raise ReconstructionValidationError("G2A.3 source identity files are malformed.")
    names: list[str] = []
    for record in files:
        if not isinstance(record, Mapping) or not isinstance(record.get("name"), str):
            raise ReconstructionValidationError("G2A.3 source identity contains an invalid file name.")
        names.append(str(record["name"]))
    return tuple(names)


def _verify_visual_review(review: Mapping[str, Any]) -> None:
    path_raw = review.get("path")
    expected_sha = review.get("sha256")
    if not isinstance(path_raw, str) or not _valid_sha256(expected_sha):
        raise ReconstructionValidationError("G2A.3 visual-review evidence is incomplete.")
    path = Path(path_raw).expanduser().resolve()
    if not path.is_file() or _sha256_file(path) != expected_sha:
        raise ReconstructionValidationError(f"G2A.3 visual-review evidence changed or is missing: {path}")


def _explicit_low_texture_diagnostic(
    *, reference_frames: Sequence[DiagnosticFrame], candidate_frames: Sequence[DiagnosticFrame], policy: FreshBindingPolicy
) -> dict[str, Any]:
    return _low_texture_component_diagnostic(
        reference_frames=reference_frames,
        candidate_frames=candidate_frames,
        policy=policy,
        selection="entire_explicit_low_texture_source_directory",
        contiguous_source_order_preserved=True,
    )


def _explicit_weak_overlap_diagnostic(
    *, reference_dir: Path, candidate_dir: Path, policy: FreshBindingPolicy
) -> dict[str, Any]:
    reference_images = _image_paths(reference_dir)
    candidate_images = _image_paths(candidate_dir)
    reference_names = tuple(path.name for path in reference_images)
    candidate_names = tuple(path.name for path in candidate_images)
    reference_desc = {path.name: _orb_descriptor(path, policy=policy) for path in reference_images}
    candidate_desc = {path.name: _orb_descriptor(path, policy=policy) for path in candidate_images}
    reference_diag = _overlap_sequence(reference_names, reference_desc, policy=policy)
    candidate_core = _overlap_sequence(candidate_names, candidate_desc, policy=policy)
    reference_median = float(reference_diag["median_pair_overlap_proxy"])
    ratio = float(candidate_core["median_pair_overlap_proxy"]) / reference_median if reference_median > 0 else float("inf")
    connected = (
        float(candidate_core["connected_pair_fraction"]) >= policy.weak_overlap_min_connected_pair_fraction
        and float(candidate_core["median_good_matches"]) >= policy.weak_overlap_min_good_matches_per_pair
    )
    return {
        "selection": "entire_explicit_weak_overlap_source_directory",
        "reference": reference_diag,
        **candidate_core,
        "reference_median_pair_overlap_proxy": reference_median,
        "median_ratio_to_reference": ratio,
        "maximum_qualified_ratio": policy.weak_overlap_max_median_ratio_to_reference_for_separate_capture,
        "minimum_required_good_matches_per_pair": policy.weak_overlap_min_good_matches_per_pair,
        "minimum_connected_pair_fraction": policy.weak_overlap_min_connected_pair_fraction,
        "connectivity_proxy_qualified": connected,
        "objective_proxy_qualified": bool(connected and ratio <= policy.weak_overlap_max_median_ratio_to_reference_for_separate_capture),
        "source_order_preserved": True,
    }


def validate_fresh_scenario_binding_snapshot(snapshot: Mapping[str, Any], *, verify_files: bool = True) -> dict[str, dict[str, Any]]:
    """Fail closed unless a frozen G2A.3 artifact can safely feed final G2A."""

    if snapshot.get("phase") != I43G2A3_PHASE or snapshot.get("status") != "ready" or snapshot.get("pass") is not True:
        raise ReconstructionValidationError("Final G2A requires a ready, passing I4.3G2A.3 fresh-binding snapshot.")
    if snapshot.get("schema_version") != I43G2A3_SCHEMA_VERSION:
        raise ReconstructionValidationError("I4.3G2A.3 schema version drift detected.")
    policy = FreshBindingPolicy()
    if snapshot.get("policy") != policy.as_dict():
        raise ReconstructionValidationError("I4.3G2A.3 frozen diagnostic policy drift detected.")
    amendments = _require_mapping(snapshot.get("methodology_amendments"), label="methodology_amendments")
    if amendments.get("low_texture") != _low_texture_amendment_record(policy):
        raise ReconstructionValidationError("I4.3G2A.3 low-texture methodology amendment drift detected.")
    summary = _require_mapping(snapshot.get("binding_summary"), label="binding_summary")
    if summary.get("required_fresh_bindings") != 3 or summary.get("ready_fresh_bindings") != 3 or summary.get("all_fresh_bindings_ready") is not True:
        raise ReconstructionValidationError("I4.3G2A.3 fresh bindings are incomplete.")
    scope = _require_mapping(snapshot.get("scope"), label="scope")
    forbidden_scope = ("real_provider_execution_started", "sfm_started", "mvs_started", "da3_started", "provider_quality_results_consumed")
    if any(bool(scope.get(key, False)) for key in forbidden_scope):
        raise ReconstructionValidationError("I4.3G2A.3 scope drifted into execution or provider-result consumption.")
    authority = _require_mapping(snapshot.get("authority"), label="authority")
    if authority.get("camera_pose_authority_owner") != "pycolmap_sfm" or authority.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError("I4.3G2A.3 authority drift.")
    forbidden_authority = (
        "authoritative_sfm_modified", "provider_selected", "provider_promoted", "auto_policy_implemented",
        "hybrid_authorized", "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    )
    if any(bool(authority.get(key, False)) for key in forbidden_authority):
        raise ReconstructionValidationError("I4.3G2A.3 forbidden authority claim detected.")

    raw_bindings = snapshot.get("fresh_bindings")
    if not isinstance(raw_bindings, Sequence) or isinstance(raw_bindings, (str, bytes)):
        raise ReconstructionValidationError("I4.3G2A.3 fresh_bindings must be a sequence.")
    by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_bindings:
        if not isinstance(raw, Mapping):
            raise ReconstructionValidationError("I4.3G2A.3 binding entry must be an object.")
        scenario_id = raw.get("scenario_id")
        if scenario_id not in _FRESH_IDS or scenario_id in by_id:
            raise ReconstructionValidationError("I4.3G2A.3 scenario ID set/order is invalid.")
        confirmation = _require_mapping(raw.get("semantic_confirmation"), label=f"{scenario_id}.semantic_confirmation")
        if raw.get("status") != "bound" or raw.get("ready_for_real_execution") is not True or confirmation.get("confirmed") is not True:
            raise ReconstructionValidationError(f"I4.3G2A.3 binding is not confirmed/ready: {scenario_id}")
        if raw.get("semantic_binding_inferred") is not False:
            raise ReconstructionValidationError("I4.3G2A.3 semantic binding must remain explicit, never inferred.")
        diagnostics = _require_mapping(raw.get("diagnostics"), label=f"{scenario_id}.diagnostics")
        if scenario_id != _COMPLETE_ROOM_ID and diagnostics.get("objective_proxy_qualified") is not True:
            raise ReconstructionValidationError(f"I4.3G2A.3 objective diagnostic is not qualified: {scenario_id}")
        identity = _require_mapping(raw.get("source_identity"), label=f"{scenario_id}.source_identity")
        review = _require_mapping(raw.get("visual_review"), label=f"{scenario_id}.visual_review")
        if verify_files:
            _verify_identity(identity)
            _verify_visual_review(review)
        by_id[str(scenario_id)] = dict(raw)
    if tuple(by_id) != _FRESH_IDS:
        raise ReconstructionValidationError(f"I4.3G2A.3 fresh scenario order drift: {tuple(by_id)}")

    if verify_files:
        complete = by_id[_COMPLETE_ROOM_ID]
        complete_identity = _require_mapping(complete["source_identity"], label="complete.source_identity")
        complete_root = Path(str(complete_identity["directory"])).expanduser().resolve()
        complete_frames = analyze_texture(complete_root, policy=policy)

        low = by_id[_LOW_TEXTURE_ID]
        low_identity = _require_mapping(low["source_identity"], label="low.source_identity")
        low_derivation = _require_mapping(low.get("derivation"), label="low.derivation")
        low_mode = low_derivation.get("mode")
        if low_mode == "controlled_natural_subset_from_complete_room":
            expected_names, expected_diag = _choose_low_texture_names(complete_frames, policy=policy)
            if _identity_names(low_identity) != expected_names:
                raise ReconstructionValidationError("I4.3G2A.3 low-texture controlled subset drifted from the frozen policy.")
        elif low_mode == "separate_explicit_real_capture":
            low_root = Path(str(low_identity["directory"])).expanduser().resolve()
            low_frames = analyze_texture(low_root, policy=policy)
            expected_diag = _explicit_low_texture_diagnostic(
                reference_frames=complete_frames,
                candidate_frames=low_frames,
                policy=policy,
            )
        else:
            raise ReconstructionValidationError("I4.3G2A.3 low-texture derivation mode is invalid.")
        if expected_diag.get("objective_proxy_qualified") is not True:
            raise ReconstructionValidationError("I4.3G2A.3 recomputed low-texture diagnostic is not qualified.")
        recorded_low_diag = _require_mapping(low.get("diagnostics"), label="low.diagnostics")
        if recorded_low_diag.get("selection") != expected_diag.get("selection"):
            raise ReconstructionValidationError("I4.3G2A.3 low-texture diagnostic provenance drift.")
        if recorded_low_diag.get("methodology") != LOW_TEXTURE_METHOD or recorded_low_diag.get("methodology_amendment_id") != LOW_TEXTURE_AMENDMENT_ID:
            raise ReconstructionValidationError("I4.3G2A.3 low-texture methodology provenance drift.")
        if recorded_low_diag.get("objective_proxy_qualified") is not expected_diag.get("objective_proxy_qualified"):
            raise ReconstructionValidationError("I4.3G2A.3 low-texture qualification verdict drift.")
        if abs(float(recorded_low_diag.get("qualification_margin_ratio", float("inf"))) - float(expected_diag["qualification_margin_ratio"])) > 1e-9:
            raise ReconstructionValidationError("I4.3G2A.3 low-texture qualification margin drift.")
        recorded_ratios = _require_mapping(recorded_low_diag.get("component_ratios_to_reference"), label="low.component_ratios_to_reference")
        expected_ratios = _require_mapping(expected_diag.get("component_ratios_to_reference"), label="expected_low.component_ratios_to_reference")
        for component in ("laplacian_variance", "contrast_stddev", "edge_density"):
            if abs(float(recorded_ratios.get(component, float("inf"))) - float(expected_ratios[component])) > 1e-9:
                raise ReconstructionValidationError(f"I4.3G2A.3 low-texture {component} ratio drift.")
        legacy = _require_mapping(recorded_low_diag.get("legacy_texture_proxy"), label="low.legacy_texture_proxy")
        if legacy.get("authoritative_for_qualification") is not False:
            raise ReconstructionValidationError("I4.3G2A.3 retired log-proxy gate regained qualification authority.")

        weak = by_id[_WEAK_OVERLAP_ID]
        weak_identity = _require_mapping(weak["source_identity"], label="weak.source_identity")
        weak_derivation = _require_mapping(weak.get("derivation"), label="weak.derivation")
        weak_mode = weak_derivation.get("mode")
        if weak_mode == "controlled_ordered_strided_subset_from_complete_room":
            low_derivation = _require_mapping(low.get("derivation"), label="low.derivation")
            weak_exclusions = _identity_names(low_identity) if low_derivation.get("mode") == "controlled_natural_subset_from_complete_room" else ()
            expected_names, expected_weak_diag = analyze_and_choose_weak_overlap(
                complete_root,
                policy=policy,
                excluded_names=weak_exclusions,
            )
            if _identity_names(weak_identity) != expected_names:
                raise ReconstructionValidationError("I4.3G2A.3 weak-overlap controlled subset drifted from the frozen policy.")
        elif weak_mode == "separate_explicit_real_capture":
            weak_root = Path(str(weak_identity["directory"])).expanduser().resolve()
            expected_weak_diag = _explicit_weak_overlap_diagnostic(
                reference_dir=complete_root,
                candidate_dir=weak_root,
                policy=policy,
            )
        else:
            raise ReconstructionValidationError("I4.3G2A.3 weak-overlap derivation mode is invalid.")
        if expected_weak_diag.get("objective_proxy_qualified") is not True:
            raise ReconstructionValidationError("I4.3G2A.3 recomputed weak-overlap diagnostic is not qualified.")
        recorded_weak_diag = _require_mapping(weak.get("diagnostics"), label="weak.diagnostics")
        if recorded_weak_diag.get("selection") != expected_weak_diag.get("selection"):
            raise ReconstructionValidationError("I4.3G2A.3 weak-overlap diagnostic provenance drift.")
        if abs(float(recorded_weak_diag.get("median_ratio_to_reference", float("inf"))) - float(expected_weak_diag["median_ratio_to_reference"])) > 1e-9:
            raise ReconstructionValidationError("I4.3G2A.3 weak-overlap diagnostic value drift.")
    return by_id


__all__ = [
    "I43G2A3_SCHEMA_VERSION",
    "I43G2A3_PHASE",
    "LOW_TEXTURE_AMENDMENT_ID",
    "LOW_TEXTURE_METHOD",
    "LEGACY_LOW_TEXTURE_METHOD",
    "FreshBindingPolicy",
    "DiagnosticFrame",
    "build_exact_source_identity",
    "analyze_texture",
    "analyze_and_choose_weak_overlap",
    "build_fresh_scenario_binding_snapshot",
    "validate_fresh_scenario_binding_snapshot",
]
