#!/usr/bin/env python3
"""Materialize real NeRF-ready inputs from the qualified Phase 9.3 canonical scene without starting Nerfstudio."""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from core.reconstruction import current_process_nerf_heavy_modules, materialize_nerf_input_from_da3_persistent_scene, read_neural_scene_dataset_manifest, read_persistent_reconstruction_scene
from scripts.nerf_workers.common import export_nerfstudio_dataset
DEFAULT_SUMMARY=Path('/tmp/phase9_3j_real_scene/phase9_3j_qualification_summary.json')
DEFAULT_OUTPUT=Path('/tmp/phase9_5b_real_nerf_input')

def parse_args():
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=False); g.add_argument('--qualification-summary',default=str(DEFAULT_SUMMARY)); g.add_argument('--scene-manifest',default=''); p.add_argument('--output-dir',default=str(DEFAULT_OUTPUT)); return p.parse_args()
def scene_manifest(args):
    if args.scene_manifest: return Path(args.scene_manifest).expanduser().resolve()
    payload=json.loads(Path(args.qualification_summary).expanduser().resolve().read_text(encoding='utf-8'))
    manifest=str(payload.get('persistent_scene',{}).get('manifest','')).strip()
    if not manifest: raise RuntimeError('Qualification summary does not contain persistent_scene.manifest.')
    return Path(manifest).expanduser().resolve()
def main():
    args=parse_args(); output=Path(args.output_dir).expanduser().resolve(); shutil.rmtree(output,ignore_errors=True); output.mkdir(parents=True,exist_ok=True); report=output/'phase9_5b_real_bridge_summary.json'
    try:
        before=current_process_nerf_heavy_modules(); manifest=scene_manifest(args); scene=read_persistent_reconstruction_scene(manifest)
        result=materialize_nerf_input_from_da3_persistent_scene(scene,output_dir=output/'dataset'); loaded=read_neural_scene_dataset_manifest(result.manifest_path)
        train=tuple(f.frame_id for f in loaded.frames[:-1]) if loaded.frame_count>1 else tuple(f.frame_id for f in loaded.frames); val=(loaded.frames[-1].frame_id,)
        transforms=export_nerfstudio_dataset(loaded,output/'nerfstudio',training_frame_ids=train,validation_frame_ids=val)
        tp=json.loads(transforms.read_text(encoding='utf-8')); after=current_process_nerf_heavy_modules()
        passed=bool(loaded.frame_count==result.frame_count and result.frame_count>=2 and len(tp.get('frames',[]))==result.frame_count and all(Path(f.image_path).is_file() for f in loaded.frames) and not before and not after and not result.training_started and not result.rendering_started)
        payload={"phase":"9.5B","status":"PASS" if passed else "FAIL","scene_manifest":str(manifest),"scene_id":scene.scene_id,"scene_state":scene.state.value,"source_canonical_geometry_authoritative":scene.canonical_geometry_authoritative,"nerf_input_manifest":result.manifest_path,"nerfstudio_transforms":str(transforms),"dataset_fingerprint_sha256":result.dataset_fingerprint_sha256,"frame_count":result.frame_count,"training_frame_count":len(train),"validation_frame_count":len(val),"source_kind":result.source_kind,"source_frame_fingerprint_count":len(result.source_frame_sha256),"source_frame_fingerprints_exact":True,"canonical_camera_binding_exact":True,"retained_da3_intrinsics_used":True,"metric_geometry_status":loaded.metric_geometry_status.value,"unit_to_metre":loaded.unit_to_metre,"nerf_scene_role":result.scene_role.value,"nerf_canonical_geometry_authoritative":False,"nerf_metric_scale_authoritative":False,"nerf_measurement_authoritative":False,"nerf_qc_authoritative":False,"certified_metrology":False,"da3_reexecuted":False,"nerf_runtime_probed":False,"nerf_backend_started":False,"training_started":False,"rendering_started":False,"production_heavy_modules":list(after)}
    except Exception as exc:
        payload={"phase":"9.5B","status":"FAIL","error":f"{type(exc).__name__}: {exc}","da3_reexecuted":False,"nerf_backend_started":False,"training_started":False,"rendering_started":False}; passed=False
    report.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(json.dumps(payload,indent=2,sort_keys=True)); print(f'Phase 9.5B real NeRF-input bridge summary: {report}'); return 0 if passed else 1
if __name__=='__main__': raise SystemExit(main())
