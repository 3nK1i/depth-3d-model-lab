#!/usr/bin/env python3
"""Deterministic Phase 8.7E1 external NeRF worker mock."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding='utf-8'))
    dataset = json.loads(Path(request['dataset_manifest_path']).read_text(encoding='utf-8'))
    output = Path(request['output_dir']).resolve() / 'development-nerf-mock'
    output.mkdir(parents=True, exist_ok=True)
    config = output / 'config.yml'
    checkpoint = output / 'contract-only.ckpt'
    config.write_text('method_name: nerfacto\nstate: contract_only\n', encoding='utf-8')
    checkpoint.write_bytes(b'DEPTH3D-NERF-CONTRACT-ONLY\n')
    payload = {
        'schema_version': 1,
        'request_id': request['request_id'],
        'backend_id': 'depth3d-development-nerf-mock',
        'backend_version': '1',
        'backend_revision': 'internal',
        'backend_provenance': request['backend_provenance'],
        'method_name': request['method_name'],
        'training_state': 'contract_only',
        'scene_role': 'appearance_only',
        'artifact_dir': str(output),
        'config_path': str(config),
        'checkpoint_path': str(checkpoint),
        'dataset_fingerprint_sha256': request['dataset_fingerprint_sha256'],
        'input_frame_count': int(dataset['frame_count']),
        'training_steps': 0,
        'training_duration_s': 0.0,
        'render_capable': False,
        'device': 'mock',
        'pose_convention': dataset['pose_convention'],
        'reconstruction_frame_id': dataset['reconstruction_frame_id'],
        'metric_geometry_status': dataset['metric_geometry_status'],
        'unit_to_metre': dataset.get('unit_to_metre'),
        'relative_scale_uncertainty': dataset.get('relative_scale_uncertainty'),
        'canonical_geometry_authoritative': False,
        'certified_metrology': False,
    }
    Path(args.result).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
