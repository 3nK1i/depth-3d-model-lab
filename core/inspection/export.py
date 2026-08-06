"""Deterministic Phase 7.5 inspection export and professional reporting."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import pickle
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.contracts.validation import ContractValidationError
from core.inspection.contracts import InspectionBundleSummary, InspectionObjectSummary

EXPORT_SCHEMA_VERSION = 1
UNCERTAINTY_NOTICE = "Estimated Uncertainty - Not Calibrated"


@dataclass(frozen=True, slots=True)
class InspectionExportOptions:
    output_dir: Path
    observation_ids: tuple[str, ...] = ()
    locked_ids: tuple[str, ...] = ()
    include_pdf: bool = True
    include_html: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser().resolve())
        object.__setattr__(self, "observation_ids", tuple(dict.fromkeys(str(v) for v in self.observation_ids if str(v))))
        object.__setattr__(self, "locked_ids", tuple(dict.fromkeys(str(v) for v in self.locked_ids if str(v))))
        if not self.include_pdf and not self.include_html:
            raise ContractValidationError("At least one report format must be enabled.")


@dataclass(frozen=True, slots=True)
class InspectionExportResult:
    export_dir: Path
    manifest_path: Path
    object_count: int
    file_count: int


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(v) for v in value]
    return value


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")
    return "_".join(part for part in cleaned.split("_") if part) or "object"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _save_rgb(path: Path, array: np.ndarray) -> None:
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(path)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.asarray(mask, dtype=bool).astype(np.uint8) * 255, mode="L").save(path)


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray | None) -> None:
    xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rgb = None
    if colors is not None:
        candidate = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        if len(candidate) == len(xyz):
            rgb = candidate
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(xyz)}\n")
        for name in ("x", "y", "z"):
            stream.write(f"property float {name}\n")
        if rgb is not None:
            for name in ("red", "green", "blue"):
                stream.write(f"property uchar {name}\n")
        stream.write("end_header\n")
        if rgb is None:
            for point in xyz:
                stream.write(f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g}\n")
        else:
            for point, color in zip(xyz, rgb, strict=True):
                stream.write(
                    f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )




def _measurement_label(kind: str) -> str:
    """Return a precise human-facing label without changing the export schema."""
    normalized = str(kind)
    if normalized == "volume":
        return "Bounding-Box Volume"
    return normalized.replace("_", " ").title()

def _measurement_display(measurement: Mapping[str, Any]) -> str:
    if measurement.get("status") != "available":
        reasons = ", ".join(str(v) for v in measurement.get("rejection_reasons", ()))
        return "Rejected" + (f" ({reasons})" if reasons else "")
    value = float(measurement.get("value_si", 0.0))
    uncertainty = float(measurement.get("expanded_uncertainty_si", 0.0))
    kind = str(measurement.get("kind", ""))
    if kind in {"width", "height", "depth"}:
        return f"{value * 1000:.1f} +/- {uncertainty * 1000:.1f} mm"
    if kind.startswith("orientation_"):
        return f"{math.degrees(value):.2f} +/- {math.degrees(uncertainty):.2f} deg"
    if kind == "volume":
        return f"{value * 1000:.2f} +/- {uncertainty * 1000:.2f} L"
    return f"{value:.3f} +/- {uncertainty:.3f} m"


def _automatic_masks(summary: InspectionBundleSummary) -> dict[str, np.ndarray]:
    """Read the trusted app-created state only to preserve original automatic masks."""
    if not summary.state_path.is_file():
        return {}
    try:
        with summary.state_path.open("rb") as stream:
            state = pickle.load(stream)  # noqa: S301 - trusted application artifact
        aligned = state.get("automatic_aligned")
        if aligned is None:
            return {}
        result: dict[str, np.ndarray] = {}
        for observation in aligned.aligned_observations:
            if observation.aligned_mask is not None:
                result[str(observation.observation_id)] = np.asarray(observation.aligned_mask, dtype=bool)
        return result
    except Exception:
        return {}


def _annotated_rgb(rgb: np.ndarray, objects: Sequence[InspectionObjectSummary], arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    palette = [(0, 190, 255, 78), (255, 190, 0, 78), (60, 220, 120, 78), (220, 90, 255, 78)]
    for index, obj in enumerate(objects):
        mask = np.asarray(arrays[obj.mask_key], dtype=bool)
        tint = Image.new("RGBA", image.size, palette[index % len(palette)])
        alpha = Image.fromarray(mask.astype(np.uint8) * palette[index % len(palette)][3], mode="L")
        overlay.alpha_composite(Image.composite(tint, Image.new("RGBA", image.size), alpha))
        if obj.detection_box_xyxy is not None:
            x1, y1, x2, y2 = obj.detection_box_xyxy
            color = palette[index % len(palette)][:3] + (255,)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            label = f"{obj.object_index + 1}. {obj.class_name}"
            draw.rectangle((x1, max(0, y1 - 19), x1 + max(90, 7 * len(label)), y1), fill=(8, 12, 20, 220))
            draw.text((x1 + 4, max(0, y1 - 17)), label, fill=(255, 255, 255, 255), font=ImageFont.load_default())
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def _object_payload(obj: InspectionObjectSummary, *, locked: bool) -> dict[str, Any]:
    return {
        "observation_id": obj.observation_id,
        "object_index": obj.object_index,
        "class_id": obj.class_id,
        "class_name": obj.class_name,
        "detection_confidence": obj.detection_confidence,
        "detection_box_xyxy": obj.detection_box_xyxy,
        "track_id": obj.track_id,
        "status": obj.status,
        "extraction_status": obj.extraction_status,
        "fitting_status": obj.fitting_status,
        "measurement_status": obj.measurement_status,
        "mask_source": obj.mask_source,
        "mask_revision": obj.mask_revision,
        "approval_status": obj.approval_status,
        "locked_in_session": bool(locked),
        "preferred_box_id": obj.preferred_box_id,
        "preferred_box_reason": obj.preferred_box_reason,
        "boxes": dict(obj.boxes),
        "measurements": list(obj.measurements),
        "diagnostics": dict(obj.diagnostics),
        "warnings": list(obj.warnings),
        "uncertainty_notice": UNCERTAINTY_NOTICE,
    }


def _write_measurements_csv(path: Path, objects: Sequence[InspectionObjectSummary], locked_ids: set[str]) -> None:
    fields = [
        "object_index", "observation_id", "class_name", "locked", "mask_source", "mask_revision",
        "measurement", "status", "value_si", "expanded_uncertainty_si", "unit_si", "reliability", "rejection_reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for obj in objects:
            for measurement in obj.measurements:
                writer.writerow({
                    "object_index": obj.object_index,
                    "observation_id": obj.observation_id,
                    "class_name": obj.class_name,
                    "locked": obj.observation_id in locked_ids,
                    "mask_source": obj.mask_source,
                    "mask_revision": obj.mask_revision,
                    "measurement": measurement.get("kind", ""),
                    "status": measurement.get("status", ""),
                    "value_si": measurement.get("value_si", ""),
                    "expanded_uncertainty_si": measurement.get("expanded_uncertainty_si", ""),
                    "unit_si": measurement.get("unit_si", ""),
                    "reliability": measurement.get("reliability", ""),
                    "rejection_reasons": ";".join(str(v) for v in measurement.get("rejection_reasons", ())),
                })


def _write_html(path: Path, summary: InspectionBundleSummary, objects: Sequence[InspectionObjectSummary], locked_ids: set[str]) -> None:
    rows = []
    for obj in objects:
        measurements = "".join(
            "<tr>"
            f"<td>{html.escape(_measurement_label(str(m.get('kind', ''))))}</td>"
            f"<td>{html.escape(_measurement_display(m))}</td>"
            f"<td>{html.escape(str(m.get('reliability', '')).title())}</td>"
            f"<td>{html.escape(str(m.get('status', '')).title())}</td>"
            "</tr>" for m in obj.measurements
        )
        folder = f"objects/object_{obj.object_index + 1:03d}_{_slug(obj.class_name)}"
        rows.append(f"""
<section class="object">
<h2>Object {obj.object_index + 1}: {html.escape(obj.class_name)}</h2>
<div class="meta">Observation: {html.escape(obj.observation_id)} | Confidence: {obj.detection_confidence if obj.detection_confidence is not None else 'N/A'} | Locked: {'Yes' if obj.observation_id in locked_ids else 'No'}</div>
<img src="{folder}/approved_mask.png" alt="Approved mask">
<table><thead><tr><th>Measurement</th><th>Value</th><th>Reliability</th><th>Status</th></tr></thead><tbody>{measurements}</tbody></table>
</section>""")
    path.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>EIA TECH Inspection Report</title>
<style>body{{font:14px Arial,sans-serif;max-width:1100px;margin:30px auto;color:#17202a}}h1{{margin-bottom:4px}}.notice{{padding:12px;background:#fff4d6;border-left:5px solid #e8a83c}}.hero{{width:100%;max-height:600px;object-fit:contain;background:#111}}.meta{{color:#59636e;margin:8px 0 14px}}.object{{page-break-inside:avoid;border-top:1px solid #ccd3da;margin-top:28px;padding-top:16px}}.object img{{width:220px;background:#111}}table{{border-collapse:collapse;width:100%;margin-top:12px}}th,td{{border:1px solid #d7dde3;padding:7px;text-align:left}}th{{background:#eef2f5}}</style></head><body>
<h1>EIA TECH - Measurement Inspection Report</h1><div class="meta">Generated {datetime.now(timezone.utc).isoformat()} | {len(objects)} object(s)</div>
<p class="notice"><strong>{UNCERTAINTY_NOTICE}</strong><br>This report is an engineering estimate and is not a certified metrology record.</p>
<img class="hero" src="annotated_rgb.png" alt="Annotated RGB">
<h2>Provenance</h2><table><tbody>
<tr><th>Depth model</th><td>{html.escape(summary.depth_model_name)}</td></tr><tr><th>Perception model</th><td>{html.escape(summary.perception_model_name)}</td></tr>
<tr><th>Depth scale</th><td>{html.escape(summary.depth_scale)}</td></tr><tr><th>Intrinsics</th><td>{html.escape(summary.intrinsics_source)}</td></tr>
<tr><th>Calibration</th><td>{html.escape(summary.calibration_status)}</td></tr><tr><th>Bundle status</th><td>{html.escape(summary.status.value)}</td></tr></tbody></table>
{''.join(rows)}</body></html>""", encoding="utf-8")


def _write_pdf(path: Path, export_dir: Path, summary: InspectionBundleSummary, objects: Sequence[InspectionObjectSummary], locked_ids: set[str]) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "EIA TECH", fontsize=22, weight="bold")
        fig.text(0.08, 0.905, "Measurement Inspection Report", fontsize=17, weight="bold")
        fig.text(0.08, 0.875, UNCERTAINTY_NOTICE, fontsize=10, weight="bold")
        fig.text(0.08, 0.852, "Engineering estimate - not a certified metrology record.", fontsize=9)
        annotated = np.asarray(Image.open(export_dir / "annotated_rgb.png").convert("RGB"))
        ax = fig.add_axes([0.08, 0.42, 0.84, 0.40]); ax.imshow(annotated); ax.axis("off")
        provenance = (
            f"Depth model: {summary.depth_model_name}\nPerception model: {summary.perception_model_name}\n"
            f"Scale: {summary.depth_scale} | Intrinsics: {summary.intrinsics_source} | Calibration: {summary.calibration_status}\n"
            f"Objects: {len(objects)} | Status: {summary.status.value}"
        )
        fig.text(0.08, 0.34, provenance, fontsize=10, linespacing=1.6)
        fig.text(0.08, 0.08, "Full machine-readable measurements, uncertainty components and geometry are included in JSON/CSV exports.", fontsize=8)
        pdf.savefig(fig); plt.close(fig)

        for obj in objects:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.07, 0.94, f"Object {obj.object_index + 1}: {obj.class_name}", fontsize=16, weight="bold")
            fig.text(0.07, 0.91, f"Observation: {obj.observation_id} | Locked: {'Yes' if obj.observation_id in locked_ids else 'No'}", fontsize=8)
            mask_path = export_dir / "objects" / f"object_{obj.object_index + 1:03d}_{_slug(obj.class_name)}" / "approved_mask.png"
            ax = fig.add_axes([0.07, 0.60, 0.35, 0.26]); ax.imshow(Image.open(mask_path), cmap="gray"); ax.set_title("Approved/current mask", fontsize=9); ax.axis("off")
            lines = [f"{_measurement_label(str(m.get('kind', '')))}: {_measurement_display(m)} [{str(m.get('reliability', '')).title()}]" for m in obj.measurements]
            fig.text(0.46, 0.86, "Measurements", fontsize=12, weight="bold")
            fig.text(0.46, 0.83, "\n".join(lines), fontsize=8.2, va="top", linespacing=1.45)
            fig.text(0.07, 0.53, f"Mask: {obj.mask_source}, revision {obj.mask_revision}, approval {obj.approval_status}", fontsize=9)
            fig.text(0.07, 0.50, f"Preferred box: {obj.preferred_box_reason or obj.preferred_box_id}", fontsize=9)
            fig.text(0.07, 0.08, UNCERTAINTY_NOTICE, fontsize=8, weight="bold")
            pdf.savefig(fig); plt.close(fig)


def export_inspection_bundle(
    summary: InspectionBundleSummary,
    arrays: Mapping[str, np.ndarray],
    options: InspectionExportOptions,
) -> InspectionExportResult:
    selected = set(options.observation_ids)
    objects = tuple(obj for obj in summary.objects if not selected or obj.observation_id in selected)
    if not objects:
        raise ContractValidationError("No inspection objects were selected for export.")
    required = {"aligned_rgb", "depth_preview"}
    for obj in objects:
        required.update({obj.mask_key, obj.raw_points_key, obj.filtered_points_key})
    missing = sorted(required.difference(arrays))
    if missing:
        raise ContractValidationError("Inspection export arrays are incomplete: " + ", ".join(missing))

    target = options.output_dir
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent)))
    locked_ids = set(options.locked_ids)
    try:
        (temporary / "objects").mkdir()
        (temporary / "provenance").mkdir()
        rgb = np.asarray(arrays["aligned_rgb"], dtype=np.uint8)
        _save_rgb(temporary / "rgb.png", rgb)
        _save_rgb(temporary / "depth_preview.png", np.asarray(arrays["depth_preview"], dtype=np.uint8))
        _save_rgb(temporary / "annotated_rgb.png", _annotated_rgb(rgb, objects, arrays))
        auto_masks = _automatic_masks(summary)
        payloads = []
        for obj in objects:
            folder = temporary / "objects" / f"object_{obj.object_index + 1:03d}_{_slug(obj.class_name)}"
            folder.mkdir()
            current_mask = np.asarray(arrays[obj.mask_key], dtype=bool)
            automatic_mask = auto_masks.get(obj.observation_id, current_mask)
            _save_mask(folder / "automatic_mask.png", automatic_mask)
            _save_mask(folder / "approved_mask.png", current_mask)
            _write_ply(folder / "raw_points.ply", arrays[obj.raw_points_key], arrays.get(obj.raw_colors_key))
            _write_ply(folder / "filtered_points.ply", arrays[obj.filtered_points_key], arrays.get(obj.filtered_colors_key))
            payload = _object_payload(obj, locked=obj.observation_id in locked_ids)
            _write_json(folder / "boxes.json", payload["boxes"])
            _write_json(folder / "measurements.json", payload)
            payloads.append(payload)

        root_payload = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "uncertainty_notice": UNCERTAINTY_NOTICE,
            "certified_metrology": False,
            "source_bundle_schema_version": summary.schema_version,
            "status": summary.status.value,
            "image_size": {"width": summary.image_width, "height": summary.image_height},
            "objects": payloads,
        }
        _write_json(temporary / "measurements.json", root_payload)
        _write_measurements_csv(temporary / "measurements.csv", objects, locked_ids)
        _write_json(temporary / "provenance" / "models.json", {
            "depth": {"model_id": summary.depth_model_id, "display_name": summary.depth_model_name},
            "perception": {"model_id": summary.perception_model_id, "display_name": summary.perception_model_name},
        })
        _write_json(temporary / "provenance" / "camera.json", {
            "intrinsics_source": summary.intrinsics_source,
            "calibration_status": summary.calibration_status,
            "depth_scale": summary.depth_scale,
            "confidence_kind": summary.confidence_kind,
        })
        _write_json(temporary / "provenance" / "session.json", {
            "elapsed_seconds": summary.elapsed_seconds,
            "box_method": summary.box_method.value,
            "warnings": list(summary.warnings),
            "exported_observation_ids": [obj.observation_id for obj in objects],
            "locked_observation_ids": sorted(locked_ids.intersection({obj.observation_id for obj in objects})),
        })
        if options.include_html:
            _write_html(temporary / "report.html", summary, objects, locked_ids)
        if options.include_pdf:
            _write_pdf(temporary / "report.pdf", temporary, summary, objects, locked_ids)

        files = sorted(path for path in temporary.rglob("*") if path.is_file() and path.name != "manifest.json")
        manifest = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "uncertainty_notice": UNCERTAINTY_NOTICE,
            "object_count": len(objects),
            "files": [
                {"path": str(path.relative_to(temporary)), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in files
            ],
        }
        _write_json(temporary / "manifest.json", manifest)
        if target.exists():
            raise FileExistsError(f"Export destination already exists: {target}")
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    final_files = tuple(path for path in target.rglob("*") if path.is_file())
    return InspectionExportResult(target, target / "manifest.json", len(objects), len(final_files))
