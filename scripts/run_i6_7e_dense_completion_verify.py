#!/usr/bin/env python3
"""Subprocess-only full dense admission revalidation; never called on the Qt thread."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from app.reconstruction_dense_terminal_verification import verify_dense_completion
PROTOCOL = 'depth3d.reconstruction.dense.verify.v1'

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--offline-output-dir', type=Path, required=True)
    p.add_argument('--session-id', required=True)
    p.add_argument('--dense-run-id', required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--geometry-sha256', required=True)
    args = p.parse_args(argv)
    try:
        result = verify_dense_completion(
            offline_output_dir=args.offline_output_dir,
            sparse_session_id=args.session_id, dense_run_id=args.dense_run_id,
            expected_manifest_sha256=args.manifest_sha256,
            expected_geometry_sha256=args.geometry_sha256,
        )
    except Exception as exc:
        print(json.dumps(dict(protocol=PROTOCOL, event='dense_verification_failed',
                              error_type=type(exc).__name__, detail=str(exc)[:1000])), flush=True)
        return 2
    print(json.dumps(dict(protocol=PROTOCOL, event='dense_verified',
                          session_id=args.session_id, dense_run_id=args.dense_run_id,
                          admission=result.as_dict()), separators=(',', ':'), allow_nan=False), flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
