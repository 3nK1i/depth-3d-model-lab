#!/usr/bin/env python3
"""Run the single real I5.10C exhaustive-matching MP-SfM evaluation."""
from __future__ import annotations

import argparse, hashlib, json, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_mp_sfm_real_evaluation import (
    build_i5_10c_candidate_command, build_i5_10c_real_evaluation,
    classify_i5_10c_execution, validate_i5_10b_for_i5_10c, validate_i5_10c_receipt,
)

OUTPUT_NAME = "i5_10c_mp_sfm_real_evaluation.json"


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def _read(path: Path) -> dict[str,Any]:
    x=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(x,dict): raise ReconstructionValidationError(f"I5.10C JSON root must be object: {path}")
    return x


def _bind(path: Path, expected: str, label: str) -> dict[str,str]:
    p=path.expanduser().resolve()
    if not p.is_file(): raise ReconstructionValidationError(f"I5.10C {label} missing: {p}")
    actual=_sha(p)
    if actual != expected.strip().lower(): raise ReconstructionValidationError(f"I5.10C {label} SHA mismatch: expected={expected} actual={actual}")
    return {"path":str(p),"sha256":actual}


def _verify_selected(preflight: Mapping[str,Any]) -> bool:
    source=preflight['source_revalidation']; root=Path(source['dataset_directory']).expanduser().resolve()
    for raw in source['selected_files']:
        p=root/raw['name']
        if not p.is_file() or p.stat().st_size != int(raw['size_bytes']) or _sha(p) != raw['sha256']: return False
    return True


def _stage(preflight: Mapping[str,Any], dest: Path) -> dict[str,Any]:
    if dest.exists(): raise ReconstructionValidationError(f"I5.10C staged window already exists: {dest}")
    dest.mkdir(parents=True,exist_ok=False)
    root=Path(preflight['source_revalidation']['dataset_directory']).expanduser().resolve(); files=[]
    for raw in preflight['source_revalidation']['selected_files']:
        src=root/raw['name']; dst=dest/raw['name']
        if not src.is_file() or _sha(src)!=raw['sha256'] or src.stat().st_size!=int(raw['size_bytes']):
            raise ReconstructionValidationError(f"I5.10C source drift before staging: {raw['name']}")
        shutil.copy2(src,dst)
        if _sha(dst)!=raw['sha256']: raise ReconstructionValidationError(f"I5.10C staged byte drift: {raw['name']}")
        files.append(dict(raw))
    return {"directory":str(dest.resolve()),"image_count":len(files),"files":files}


def _verify_stage(stage: Mapping[str,Any]) -> bool:
    root=Path(stage['directory'])
    return all((root/r['name']).is_file() and _sha(root/r['name'])==r['sha256'] and (root/r['name']).stat().st_size==int(r['size_bytes']) for r in stage['files'])


def _verify_anchors(preflight: Mapping[str,Any]) -> tuple[bool,dict[str,Any]]:
    out={}; ok=True
    for rel,raw in preflight['repository_anchors'].items():
        p=(ROOT/rel).resolve(); exists=p.is_file(); actual=_sha(p) if exists else None; good=exists and actual==raw['sha256']; ok = ok and good
        out[rel]={"path":str(p),"expected_sha256":raw['sha256'],"actual_sha256":actual,"verified":good}
    return ok,out


def _utc() -> str: return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--i5-10b-preflight',type=Path,required=True); ap.add_argument('--i5-10b-sha256',required=True)
    ap.add_argument('--post-i5-10b-delta-checkpoint',type=Path,required=True); ap.add_argument('--post-i5-10b-delta-sha256',required=True)
    ap.add_argument('--evaluation-receipt',type=Path,required=True); ap.add_argument('--evaluation-receipt-sha256',required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    out=args.output_dir.expanduser().resolve()
    if out.exists(): raise ReconstructionValidationError(f"I5.10C output directory must be new/immutable: {out}")
    out.mkdir(parents=True,exist_ok=False)

    pb=_bind(args.i5_10b_preflight,args.i5_10b_sha256,'I5.10B preflight'); pre=validate_i5_10b_for_i5_10c(_read(Path(pb['path'])))
    db=_bind(args.post_i5_10b_delta_checkpoint,args.post_i5_10b_delta_sha256,'post-I5.10B delta checkpoint')
    rb=_bind(args.evaluation_receipt,args.evaluation_receipt_sha256,'I5.10C evaluation receipt'); receipt=validate_i5_10c_receipt(_read(Path(rb['path'])))

    source_before=_verify_selected(pre); worker_before,anchors_before=_verify_anchors(pre)
    if not source_before or not worker_before: raise ReconstructionValidationError('I5.10C pre-execution source/worker integrity failed.')
    stage=_stage(pre,out/'inputs'/'kitti_forward_motion')

    attempt=out/'01_exhaustive_attempt_001'; sfm_out=attempt/'sfm'; attempt.mkdir(parents=True,exist_ok=False)
    use_gpu=pre['runtime_preflight']['device_mode']=='cuda'; sfm_python=pre['runtime_preflight']['python_executable']
    cmd=build_i5_10c_candidate_command(current_python=sys.executable,sfm_python=sfm_python,staged_window=Path(stage['directory']),sfm_output=sfm_out,use_gpu=use_gpu)
    log=attempt/'console.log'; started=_utc()
    print('I5.10C command: '+' '.join(cmd),flush=True)
    with log.open('w',encoding='utf-8') as f:
        p=subprocess.Popen(cmd,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            print(line,end='',flush=True); f.write(line); f.flush()
        rc=int(p.wait())
    finished=_utc(); text=log.read_text(encoding='utf-8',errors='replace')
    report_path=sfm_out/'real_sfm_report.json'; report=_read(report_path) if report_path.is_file() else None
    try:
        analysis=classify_i5_10c_execution(return_code=rc,report=report,console_text=text,preflight=pre)
    except Exception as exc:
        analysis=classify_i5_10c_execution(return_code=98,report=None,console_text=f"I5.10C report analysis integrity failure: {type(exc).__name__}: {exc}",preflight=pre)

    source_after=_verify_selected(pre); staged_after=_verify_stage(stage); worker_after,anchors_after=_verify_anchors(pre)
    if analysis['terminal_status'] in {'pass','degraded','rejected'} and not all((source_after,staged_after,worker_after)):
        analysis=classify_i5_10c_execution(return_code=97,report=None,console_text='I5.10C post-execution source/staged/worker integrity failure',preflight=pre)

    record={
        'candidate_command':cmd,'attempt_started_utc':started,'attempt_finished_utc':finished,'return_code':rc,
        'console_log':str(log.resolve()),'console_log_sha256':_sha(log),'report':str(report_path.resolve()) if report_path.is_file() else None,
        'report_sha256':_sha(report_path) if report_path.is_file() else None,'source_integrity_after':source_after,
        'staged_integrity_after':staged_after,'worker_integrity_after':worker_after,'attempt_directory':str(attempt.resolve()),
        'repository_anchors_before':anchors_before,'repository_anchors_after':anchors_after,
    }
    result=build_i5_10c_real_evaluation(preflight=pre,preflight_binding=pb,post_i5_10b_delta_binding=db,evaluation_receipt=receipt,evaluation_receipt_binding=rb,
        execution_analysis=analysis,execution_record=record,source_integrity_before=source_before,source_integrity_after=source_after,staged_integrity_after=staged_after,
        worker_integrity_before=worker_before,worker_integrity_after=worker_after)
    result['staged_input_evidence']=stage
    evidence=out/OUTPUT_NAME; evidence.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print('I5.10C Real Controlled MP-SfM Evaluation: '+('EVIDENCE_COMPLETE' if result['pass'] else 'INCOMPLETE'))
    print(f"terminal_status={result['terminal_status']} quality_status={result['quality_status']}")
    print(f"candidate=exhaustive registered={analysis.get('registered_images')}/60 sparse_model={analysis.get('mapping_produced_sparse_model')}")
    print(f"historical_baseline=sequential_overlap10 registered=0/60 quality=rejected")
    print(f"source_integrity_after={source_after} staged_integrity_after={staged_after} worker_integrity_after={worker_after}")
    print('retry_until_pass=False threshold_relaxation=False production_default_modified=False baseline_rerun=False')
    print('pycolmap_authority=True metric_scale=False metrology=False auto=False hybrid=False')
    print('I5.10D evidence-only qualification and closure: '+('AUTHORIZED / NOT STARTED' if result['decision']['i5_10d_authorized'] else 'NOT AUTHORIZED'))
    print(f"evidence={evidence}")
    return 0 if result['pass'] else 3

if __name__=='__main__':
    try: raise SystemExit(main())
    except (ReconstructionValidationError,OSError,ValueError,TypeError,KeyError,json.JSONDecodeError) as exc:
        print(f"I5.10C ERROR: {type(exc).__name__}: {exc}",file=sys.stderr); raise SystemExit(2)
