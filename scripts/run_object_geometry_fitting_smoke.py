#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import cv2
from core.object_geometry import fit_object_geometry,make_fitting_request
from tests.object_geometry_fitting_helpers import extraction_batch,cuboid_points

def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',required=True); a=p.parse_args()
 points=cuboid_points(extents=(1.8,.8,.5),centre=(.3,-.2,2.4),angle=.55,noise=.002,outliers=30)
 result=fit_object_geometry(make_fitting_request(extraction_batch(points))).object_results[0]
 canvas=np.zeros((800,1200,3),np.uint8); canvas[:]=18
 def panel(x0,y0,w,h,title): cv2.rectangle(canvas,(x0,y0),(x0+w,y0+h),(60,60,60),1); cv2.putText(canvas,title,(x0+10,y0+28),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),1,cv2.LINE_AA)
 panel(20,20,560,350,'Camera X/Z points + fitted boxes'); panel(620,20,560,350,'Object-local principal frame'); panel(20,400,1160,360,'Fit diagnostics')
 def draw_pts(pts,x0,y0,w,h,axes=(0,2),color=(0,230,100)):
  q=pts[:,axes]; lo=q.min(0); hi=q.max(0); s=np.maximum(hi-lo,1e-9); uv=(q-lo)/s; uv[:,0]=x0+30+uv[:,0]*(w-60); uv[:,1]=y0+h-30-uv[:,1]*(h-60)
  for u,v in uv[::max(1,len(uv)//2500)]: cv2.circle(canvas,(int(u),int(v)),1,color,-1)
 draw_pts(points,20,20,560,350)
 R=result.principal_axes.orientation_matrix; local=(points-result.principal_axes.robust_centre_xyz)@R; draw_pts(local,620,20,560,350,axes=(0,1),color=(60,210,255))
 lines=[f'status={result.status.value}',f'preferred={result.preferred_box_id}',f'degeneracy={result.principal_axes.degeneracy.value}',f'orientation={result.object_frame.orientation_stability.value}',f'eigenvalues={np.array2string(result.principal_axes.eigenvalues,precision=5)}',f'AABB extents={np.array2string(result.axis_aligned_box.extents_xyz_m,precision=4)}',f'Camera robust extents={np.array2string(result.camera_aligned_box.extents_xyz_m,precision=4)}',f'OBB extents={np.array2string(result.oriented_box.extents_xyz_m,precision=4)}',f'metric_ready={result.metric_ready}']
 for i,t in enumerate(lines): cv2.putText(canvas,t,(45,445+i*34),cv2.FONT_HERSHEY_SIMPLEX,.68,(235,235,235),1,cv2.LINE_AA)
 cv2.imwrite(a.output,canvas)
 print(f'status={result.status.value} preferred_box={result.oriented_box.kind.value if result.oriented_box else "fallback"} orientation_stability={result.object_frame.orientation_stability.value} output={a.output}')
if __name__=='__main__': main()
