from __future__ import annotations

from pathlib import Path
import sys
import textwrap

from app.reconstruction_controller import (
    OfflineReconstructionConfiguration,
    OfflineReconstructionSourceKind,
)


def write_h3_worker(path: Path) -> Path:
    path.write_text(
        textwrap.dedent(
            r'''
            import argparse, json, signal, time
            from pathlib import Path

            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument('--session-id', required=True)
            parser.add_argument('--output-dir', required=True)
            args, _ = parser.parse_known_args()
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            mode = out.name.lower()

            def result(passed=True):
                payload = {
                    'phase': '8.8H3',
                    'pass': bool(passed),
                    'status': 'completed' if passed else 'failed',
                    'session_id': args.session_id,
                    'camera_pose_available': bool(passed),
                    'canonical_geometry_authoritative': False,
                    'persistent_scene_ready': False,
                    'certified_metrology': False,
                    'reconstruction_ui_enabled': True,
                    'tests_fixture_used': True,
                }
                (out / 'offline_result.json').write_text(json.dumps(payload), encoding='utf-8')

            if 'fail' in mode:
                print('H3 synthetic worker failure', flush=True)
                raise SystemExit(7)

            if 'cancel' in mode or 'shutdown' in mode:
                def on_term(_sig, _frame):
                    raise SystemExit(0)
                signal.signal(signal.SIGTERM, on_term)
                print('H3 cancellable worker ready', flush=True)
                while True:
                    time.sleep(0.05)

            result(True)
            print(json.dumps({
                'protocol': 'eia.reconstruction.offline.v1',
                'event': 'completed',
                'stage': 'camera_pose',
                'message': 'H3 mock completed',
                'progress_fraction': 1.0,
            }), flush=True)
            '''
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return path.resolve()


def write_ready_sequence(root: Path) -> Path:
    source = (root / "h3-sequence").resolve()
    source.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        (source / f"frame-{index:02d}.png").write_bytes(b"phase8.8h3")
    return source


def offline_configuration(root: Path, name: str) -> OfflineReconstructionConfiguration:
    source = write_ready_sequence(root)
    return OfflineReconstructionConfiguration(
        source_kind=OfflineReconstructionSourceKind.IMAGE_DIRECTORY,
        inputs=(source,),
        output_dir=(root / name).resolve(),
        sfm_python=Path(sys.executable).resolve(),
        max_keyframes=3,
        use_gpu=False,
    )
