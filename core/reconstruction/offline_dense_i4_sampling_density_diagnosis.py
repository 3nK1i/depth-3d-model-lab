"""I4.1.4 decoupled global-coverage / local-density sampling diagnosis.

Diagnosis-only analysis introduced after I4.1.3 fixed full-clip temporal
coverage but the Horse orbit still produced a collapsed dense-consistency path.
The module measures pair qualification as a function of temporal spacing,
camera baseline, target-centered viewpoint separation, order gap, and capture
segment.  It can compare the current full-span run with a same-source prefix
run and an optional short-control segment.

No sampling, SfM, DA3, dense-consistency, filtering, persistence, metric,
measurement, or mesh policy is modified here.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, isfinite
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError

I4_1_4_SCHEMA_VERSION = 1
_EPS = 1e-12


@dataclass(frozen=True)
class PairObservation:
    order_a: int
    order_b: int
    status: str
    delta_t_seconds: float
    order_gap: int
    camera_baseline_reconstruction_units: float
    camera_baseline_over_median_standoff: float
    target_view_angle_degrees: float
    source_mid_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "orders": [int(self.order_a), int(self.order_b)],
            "status": str(self.status),
            "pass": self.status == "pass",
            "delta_t_seconds": float(self.delta_t_seconds),
            "order_gap": int(self.order_gap),
            "camera_baseline_reconstruction_units": float(self.camera_baseline_reconstruction_units),
            "camera_baseline_over_median_standoff": float(self.camera_baseline_over_median_standoff),
            "target_view_angle_degrees": float(self.target_view_angle_degrees),
            "source_mid_fraction": float(self.source_mid_fraction),
        }


def _finite_matrices(camera_matrices: object) -> np.ndarray:
    values = np.asarray(camera_matrices, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4) or len(values) < 2 or not np.isfinite(values).all():
        raise ReconstructionValidationError("I4.1.4 requires finite Nx4x4 camera matrices.")
    return values


def _finite_target(target: object) -> np.ndarray:
    value = np.asarray(target, dtype=np.float64).reshape(-1)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ReconstructionValidationError("I4.1.4 target must be a finite 3-vector.")
    return value


def build_pair_observations(
    *,
    edge_statuses: Mapping[tuple[int, int], str],
    order_to_timestamp_ns: Mapping[int, int],
    order_to_source_index: Mapping[int, int],
    order_to_camera_matrix: Mapping[int, object],
    target_reconstruction: object,
) -> tuple[PairObservation, ...]:
    """Construct exact pair-level temporal/geometric diagnostics."""

    if not edge_statuses:
        raise ReconstructionValidationError("I4.1.4 requires at least one dense-consistency edge.")
    target = _finite_target(target_reconstruction)
    orders = sorted({int(v) for edge in edge_statuses for v in edge})
    missing = [
        order for order in orders
        if order not in order_to_timestamp_ns or order not in order_to_source_index or order not in order_to_camera_matrix
    ]
    if missing:
        raise ReconstructionValidationError(f"I4.1.4 edge orders are missing SfM metadata: {missing!r}")
    matrices = {int(order): np.asarray(order_to_camera_matrix[order], dtype=np.float64) for order in orders}
    _finite_matrices(np.stack([matrices[o] for o in orders], axis=0))
    centers = np.stack([matrices[o][:3, 3] for o in orders], axis=0)
    standoff = np.linalg.norm(centers - target.reshape(1, 3), axis=1)
    if np.any(standoff <= _EPS):
        raise ReconstructionValidationError("I4.1.4 target coincides with a camera center.")
    median_standoff = float(np.median(standoff))
    source_values = np.asarray([int(order_to_source_index[o]) for o in orders], dtype=np.float64)
    source_min = float(np.min(source_values))
    source_span = max(float(np.max(source_values) - source_min), 1.0)

    result: list[PairObservation] = []
    for raw_key, raw_status in sorted(edge_statuses.items()):
        a, b = sorted((int(raw_key[0]), int(raw_key[1])))
        status = str(raw_status).strip().lower()
        if status not in {"pass", "degraded", "inconclusive"}:
            raise ReconstructionValidationError(f"I4.1.4 unsupported pair status: {raw_status!r}")
        ta = int(order_to_timestamp_ns[a]); tb = int(order_to_timestamp_ns[b])
        delta_t = abs(tb - ta) / 1_000_000_000.0
        if not isfinite(delta_t) or delta_t < 0.0:
            raise ReconstructionValidationError("I4.1.4 pair timestamp delta is invalid.")
        ca = matrices[a][:3, 3]; cb = matrices[b][:3, 3]
        baseline = float(np.linalg.norm(cb - ca))
        ua = (ca - target) / max(float(np.linalg.norm(ca - target)), _EPS)
        ub = (cb - target) / max(float(np.linalg.norm(cb - target)), _EPS)
        view_angle = float(degrees(acos(float(np.clip(np.dot(ua, ub), -1.0, 1.0)))))
        mid_source = 0.5 * (float(order_to_source_index[a]) + float(order_to_source_index[b]))
        result.append(PairObservation(
            order_a=a,
            order_b=b,
            status=status,
            delta_t_seconds=float(delta_t),
            order_gap=int(b - a),
            camera_baseline_reconstruction_units=baseline,
            camera_baseline_over_median_standoff=float(baseline / max(median_standoff, _EPS)),
            target_view_angle_degrees=view_angle,
            source_mid_fraction=float(np.clip((mid_source - source_min) / source_span, 0.0, 1.0)),
        ))
    return tuple(result)


def _metric_summary(values: Sequence[float]) -> dict[str, float | None]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return {"p25": None, "p50": None, "p75": None, "p90": None}
    return {
        "p25": float(np.quantile(data, 0.25)),
        "p50": float(np.quantile(data, 0.50)),
        "p75": float(np.quantile(data, 0.75)),
        "p90": float(np.quantile(data, 0.90)),
    }


def _bucket_summary(observations: Sequence[PairObservation], predicate: Any) -> dict[str, Any]:
    chosen = [item for item in observations if predicate(item)]
    decisive = [item for item in chosen if item.status in {"pass", "degraded"}]
    passed = sum(item.status == "pass" for item in decisive)
    return {
        "edge_count": len(chosen),
        "decisive_edge_count": len(decisive),
        "pass_count": int(passed),
        "degraded_count": int(sum(item.status == "degraded" for item in decisive)),
        "inconclusive_count": int(sum(item.status == "inconclusive" for item in chosen)),
        "pass_fraction": None if not decisive else float(passed / len(decisive)),
    }


def summarize_pair_observations(observations: Sequence[PairObservation]) -> dict[str, Any]:
    if not observations:
        raise ReconstructionValidationError("I4.1.4 cannot summarize an empty edge set.")
    obs = tuple(observations)
    decisive = [item for item in obs if item.status in {"pass", "degraded"}]
    passed = [item for item in decisive if item.status == "pass"]
    degraded = [item for item in decisive if item.status == "degraded"]
    return {
        "edge_count": len(obs),
        "decisive_edge_count": len(decisive),
        "pass_count": len(passed),
        "degraded_count": len(degraded),
        "inconclusive_count": int(sum(item.status == "inconclusive" for item in obs)),
        "pass_fraction": None if not decisive else float(len(passed) / len(decisive)),
        "delta_t_seconds": {
            "all": _metric_summary([x.delta_t_seconds for x in decisive]),
            "pass": _metric_summary([x.delta_t_seconds for x in passed]),
            "degraded": _metric_summary([x.delta_t_seconds for x in degraded]),
        },
        "target_view_angle_degrees": {
            "all": _metric_summary([x.target_view_angle_degrees for x in decisive]),
            "pass": _metric_summary([x.target_view_angle_degrees for x in passed]),
            "degraded": _metric_summary([x.target_view_angle_degrees for x in degraded]),
        },
        "baseline_over_median_standoff": {
            "all": _metric_summary([x.camera_baseline_over_median_standoff for x in decisive]),
            "pass": _metric_summary([x.camera_baseline_over_median_standoff for x in passed]),
            "degraded": _metric_summary([x.camera_baseline_over_median_standoff for x in degraded]),
        },
        "time_buckets": {
            "le_0_75s": _bucket_summary(obs, lambda x: x.delta_t_seconds <= 0.75 + 1e-9),
            "gt_0_75_le_1_5s": _bucket_summary(obs, lambda x: 0.75 < x.delta_t_seconds <= 1.50 + 1e-9),
            "gt_1_5_le_3_0s": _bucket_summary(obs, lambda x: 1.50 < x.delta_t_seconds <= 3.0 + 1e-9),
            "gt_3_0s": _bucket_summary(obs, lambda x: x.delta_t_seconds > 3.0),
        },
        "order_gap": {
            str(gap): _bucket_summary(obs, lambda x, g=gap: x.order_gap == g)
            for gap in sorted({x.order_gap for x in obs})
        },
        "source_thirds": {
            "early": _bucket_summary(obs, lambda x: x.source_mid_fraction < 1.0 / 3.0),
            "middle": _bucket_summary(obs, lambda x: 1.0 / 3.0 <= x.source_mid_fraction < 2.0 / 3.0),
            "late": _bucket_summary(obs, lambda x: x.source_mid_fraction >= 2.0 / 3.0),
        },
    }


def _pass_fraction(summary: Mapping[str, Any]) -> float | None:
    value = summary.get("pass_fraction")
    return None if value is None else float(value)


def classify_sampling_density_evidence(
    *,
    current: Mapping[str, Any],
    same_source_dense_prefix: Mapping[str, Any] | None,
    short_control: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Classify whether local density, capture regime, or both dominate.

    The thresholds are diagnostic effect-size rules only; they do not alter any
    reconstruction acceptance criterion.
    """

    current_pf = _pass_fraction(current)
    prefix_pf = None if same_source_dense_prefix is None else _pass_fraction(same_source_dense_prefix)
    control_pf = None if short_control is None else _pass_fraction(short_control)
    current_short = current.get("time_buckets", {}).get("le_0_75s", {}) if isinstance(current.get("time_buckets"), Mapping) else {}
    current_short_pf = _pass_fraction(current_short) if isinstance(current_short, Mapping) else None

    density_gain = None if current_pf is None or prefix_pf is None else float(prefix_pf - current_pf)
    regime_gain = None if prefix_pf is None or control_pf is None else float(control_pf - prefix_pf)
    density_signal = density_gain is not None and density_gain >= 0.10
    regime_signal = regime_gain is not None and regime_gain >= 0.15
    current_local_signal = current_short_pf is not None and current_pf is not None and current_short_pf - current_pf >= 0.10

    if density_signal and regime_signal:
        classification = "mixed_capture_regime_and_local_density"
    elif regime_signal:
        classification = "capture_regime_dominant_local_density_secondary" if (density_gain or 0.0) > 0.03 else "capture_regime_dominant"
    elif density_signal or current_local_signal:
        classification = "local_density_dominant"
    elif current_pf is not None and current_pf < 0.10 and prefix_pf is not None and prefix_pf < 0.20:
        classification = "same_source_dense_consistency_poor_even_at_higher_local_density"
    else:
        classification = "insufficient_or_mixed_evidence"

    decoupled_candidate = classification in {"local_density_dominant", "mixed_capture_regime_and_local_density"}
    cleaner_capture_candidate = classification in {
        "capture_regime_dominant",
        "capture_regime_dominant_local_density_secondary",
        "mixed_capture_regime_and_local_density",
        "same_source_dense_consistency_poor_even_at_higher_local_density",
    }
    return {
        "classification": classification,
        "current_pass_fraction": current_pf,
        "same_source_prefix_pass_fraction": prefix_pf,
        "short_control_pass_fraction": control_pf,
        "same_source_prefix_gain_over_current": density_gain,
        "short_control_gain_over_same_source_prefix": regime_gain,
        "current_sub_075s_pass_fraction": current_short_pf,
        "local_density_signal": bool(density_signal or current_local_signal),
        "capture_regime_signal": bool(regime_signal),
        "decoupled_sampling_architecture_candidate": bool(decoupled_candidate),
        "cleaner_static_capture_recommended": bool(cleaner_capture_candidate),
        "sampling_policy_change_authorized": False,
        "dense_consistency_threshold_relaxation_authorized": False,
    }


def build_i4_1_4_diagnosis(
    *,
    current_summary: Mapping[str, Any],
    same_source_prefix_summary: Mapping[str, Any] | None,
    short_control_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    diagnosis = classify_sampling_density_evidence(
        current=current_summary,
        same_source_dense_prefix=same_source_prefix_summary,
        short_control=short_control_summary,
    )
    return {
        "phase": "I4.1.4",
        "schema_version": I4_1_4_SCHEMA_VERSION,
        "status": "diagnosis_ready",
        "current_full_span": dict(current_summary),
        "same_source_dense_prefix": None if same_source_prefix_summary is None else dict(same_source_prefix_summary),
        "short_control": None if short_control_summary is None else dict(short_control_summary),
        "diagnosis": diagnosis,
        "authority": {
            "sfm_rerun": False,
            "da3_rerun": False,
            "dense_pair_rerun": False,
            "fusion_rerun": False,
            "sampling_policy_changed": False,
            "dense_consistency_policy_changed": False,
            "filter_policy_changed": False,
            "persistence_changed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "i4_2_detail_preservation_authorized": False,
            "mesh_authorization": False,
            "next_action": (
                "Review I4.1.4 classification. Promote a decoupled global/local sampling architecture only if local-density evidence is strong; "
                "otherwise use a cleaner static object-orbit capture and preserve Horse as an I5 robustness case. Do not relax dense-consistency thresholds."
            ),
        },
    }
