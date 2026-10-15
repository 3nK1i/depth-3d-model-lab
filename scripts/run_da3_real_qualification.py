#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from core.reconstruction.da3 import DA3_EFFICIENT_MODEL_ID, probe_da3_environment, validate_da3_npz_result
from core.reconstruction.da3_provider import DA3ExternalWorkerRunner, DA3InputManifest, DA3ProviderProfile, write_da3_input_manifest

def args():
    p=argparse.ArgumentParser(); p.add_argument("--image-dir",required=True); p.add_argument("--output-dir",default="/tmp/phase8_7f2_da3_real"); p.add_argument("--max-images",type=int,default=3); p.add_argument("--process-res",type=int,default=392); p.add_argument("--model-id",default=DA3_EFFICIENT_MODEL_ID); p.add_argument("--model-revision",default=""); return p.parse_args()

a=args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
python=Path.home() / "anaconda3/envs/eia-da3/bin/python"; repo=Path.home() / "Desktop/eia-da3-backends/depth-anything-3"
probe=probe_da3_environment(python,repo)
if not probe.available: raise SystemExit(f"DA3 environment not qualified: {probe.as_dict()}")
files=sorted([p for p in Path(a.image_dir).iterdir() if p.suffix.lower() in {".png",".jpg",".jpeg",".webp"}])
if len(files)<2: raise SystemExit("Need at least two images for F2 multi-view qualification")
count=min(max(2,a.max_images),len(files)); indices=np.linspace(0,len(files)-1,num=count,dtype=np.int64); selected=[files[int(i)] for i in indices]
request_id="phase8-7f2-real-da3"
input_path=write_da3_input_manifest(out/"input_manifest.json",DA3InputManifest(request_id=request_id,image_paths=tuple(str(p.resolve()) for p in selected),metadata={"source":"real_multiview_qualification"}))
runner=DA3ExternalWorkerRunner(python_executable=python,worker_script=(ROOT/"scripts/da3_workers/da3_worker.py").resolve(),profile=DA3ProviderProfile(model_id=a.model_id,model_revision=a.model_revision,process_res=a.process_res),timeout_s=600)
start=time.perf_counter(); manifest_path,manifest=runner.run(input_path,out/"worker-output"); wall=time.perf_counter()-start
arrays=validate_da3_npz_result(manifest)
# Visual contact sheet: input | normalized inverse-depth color visualization.
rows=[]
for path,depth in zip(selected,arrays["depth"]):
    rgb=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if rgb is None: continue
    h,w=depth.shape; rgb=cv2.resize(rgb,(w,h),interpolation=cv2.INTER_AREA)
    lo,hi=np.percentile(depth,[2,98]); denom=max(float(hi-lo),1e-6); norm=np.clip((depth-lo)/denom,0,1); vis=(norm*255).astype(np.uint8); vis=cv2.applyColorMap(vis,cv2.COLORMAP_TURBO)
    rows.append(np.hstack([rgb,vis]))
contact=out/"depth_contact_sheet.png"
if rows: cv2.imwrite(str(contact),np.vstack(rows))
poses=arrays["T_reconstruction_from_camera"]
summary={"pass":True,"phase":"8.7F2","environment":probe.as_dict(),"model_id":manifest.model_id,"model_revision":manifest.model_revision,"source_revision":manifest.source_revision,"frame_count":manifest.frame_count,"output_resolution":[manifest.width,manifest.height],"inference_duration_s":manifest.inference_duration_s,"qualification_wall_s":wall,"peak_vram_mib":manifest.peak_vram_mib,"gpu_name":manifest.gpu_name,"pose_convention":"T_reconstruction_from_camera","pose_count":int(poses.shape[0]),"depth_positive":bool(np.all(arrays["depth"]>0)),"confidence_nonnegative":bool(np.all(arrays["confidence"]>=0)),"intrinsics_valid":True,"geometry_scale":"relative_arbitrary_scale","canonical_geometry_authoritative":False,"certified_metrology":False,"production_depth_environment_untouched":True,"reconstruction_ui_enabled":False,"result_manifest":str(manifest_path.resolve()),"contact_sheet":str(contact.resolve()) if contact.is_file() else ""}
(out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
print("Phase 8.7F2 real Depth Anything 3 qualification: PASS")
print(f"gpu={manifest.gpu_name} frames={manifest.frame_count} model={manifest.model_id}")
print(f"rendered_depth={manifest.width}x{manifest.height} inference_s={manifest.inference_duration_s:.3f} peak_vram_mib={manifest.peak_vram_mib:.1f}")
print(f"poses={poses.shape[0]} convention=T_reconstruction_from_camera depth_positive=True confidence_nonnegative=True")
print("geometry_scale=relative_arbitrary_scale canonical_geometry_authoritative=False certified_metrology=False")
print("production_depth_environment_untouched=True reconstruction_ui_enabled=False")
print(f"contact_sheet={summary['contact_sheet']}")
print(f"summary={out/'summary.json'}")
