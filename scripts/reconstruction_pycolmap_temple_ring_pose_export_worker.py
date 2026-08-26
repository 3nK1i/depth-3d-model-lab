#!/usr/bin/env python3
"""Read-only PyCOLMAP pose exporter for I4.2C.

The worker intentionally has no access to Middlebury R/t validation evidence.
It only opens the frozen authoritative I4.2B sparse model and serializes its 47
camera-from-world transforms for the parent validation process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping

import pycolmap

SCHEMA_VERSION = 1
EXPECTED_NAMES = tuple(f"templeR{index:04d}.png" for index in range(1, 48))
EXPECTED_FILES = ("cameras.bin", "images.bin", "points3D.bin")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("I4.2C pose-export request must be a JSON object.")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix34(transform: Any) -> list[list[float]]:
    matrix = transform.matrix()
    return [[float(matrix[row, column]) for column in range(4)] for row in range(3)]


def run(request: Mapping[str, Any]) -> dict[str, Any]:
    if int(request.get("schema_version", 0)) != SCHEMA_VERSION or request.get("phase") != "I4.2C-pose-export":
        raise ValueError("Unsupported I4.2C pose-export request.")
    if request.get("read_only") is not True or request.get("ground_truth_available_to_worker") is not False:
        raise ValueError("I4.2C pose exporter must remain read-only and ground-truth blind.")
    if str(pycolmap.__version__) != "4.1.1":
        raise RuntimeError(f"I4.2C requires PyCOLMAP 4.1.1, found {pycolmap.__version__}")

    sparse = Path(str(request["authoritative_sparse"])).expanduser().resolve()
    expected_names = tuple(str(name) for name in request.get("expected_image_names", []))
    expected_hashes = request.get("expected_sparse_sha256")
    if expected_names != EXPECTED_NAMES or not isinstance(expected_hashes, Mapping):
        raise RuntimeError("I4.2C pose-export canonical image/hash contract changed.")
    if set(expected_hashes) != set(EXPECTED_FILES):
        raise RuntimeError("I4.2C pose-export sparse checksum set is incomplete.")

    before = {}
    for filename in EXPECTED_FILES:
        path = sparse / filename
        if not path.is_file():
            raise RuntimeError(f"I4.2C authoritative sparse file missing: {filename}")
        before[filename] = sha256_file(path)
        if before[filename] != str(expected_hashes[filename]):
            raise RuntimeError(f"I4.2C authoritative sparse checksum changed before read: {filename}")

    reconstruction = pycolmap.Reconstruction(sparse)
    if int(reconstruction.num_reg_images()) != 47:
        raise RuntimeError(f"I4.2C expected 47 registered images, found {reconstruction.num_reg_images()}")
    camera_ids = sorted({int(reconstruction.image(int(image_id)).camera_id) for image_id in reconstruction.reg_image_ids()})
    if len(camera_ids) != 1:
        raise RuntimeError(f"I4.2C expected one shared camera, found {len(camera_ids)}")

    registered = []
    for image_id in sorted(int(value) for value in reconstruction.reg_image_ids()):
        image = reconstruction.image(image_id)
        registered.append({
            "image_id": image_id,
            "name": str(image.name),
            "camera_id": int(image.camera_id),
            "camera_from_reconstruction": matrix34(image.cam_from_world()),
        })
    if tuple(sorted(item["name"] for item in registered)) != EXPECTED_NAMES:
        raise RuntimeError("I4.2C authoritative sparse model does not contain canonical TempleRing 1..47 names.")

    after = {filename: sha256_file(sparse / filename) for filename in EXPECTED_FILES}
    if after != before:
        raise RuntimeError("I4.2C pose export mutated authoritative sparse evidence.")

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "I4.2C-pose-export",
        "status": "ok",
        "python_executable": sys.executable,
        "pycolmap_version": str(pycolmap.__version__),
        "read_only": True,
        "authoritative_sparse_modified": False,
        "registered_images": registered,
        "sparse_sha256_before": before,
        "sparse_sha256_after": after,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="I4.2C read-only TempleRing pose exporter")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    result_path = Path(args.result).expanduser().resolve()
    try:
        payload = run(read_json(Path(args.request).expanduser().resolve()))
        write_json(result_path, payload)
        return 0
    except Exception as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "phase": "I4.2C-pose-export",
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        try:
            write_json(result_path, payload)
        except Exception:
            pass
        print(payload["traceback"], file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
