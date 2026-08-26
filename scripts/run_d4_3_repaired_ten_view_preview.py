#!/usr/bin/env python3
"""Interactive preview of the D4.3 selected repaired ten-view raw cloud."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
os.environ.pop("DEPTH3D_TEST_FAKE_VISPY", None)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget  # noqa: E402
from app.widgets.viewport.vispy_canvas import PointCloudCanvas  # noqa: E402
from core.geometry.sampling import enforce_point_budget  # noqa: E402


def _args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--d4-3-dir', type=Path, default=Path('/tmp/eia_d4_3_repaired_ten_view_chain'))
    p.add_argument('--color-mode', choices=('rgb','source'), default='source')
    p.add_argument('--point-budget', type=int, default=250000)
    p.add_argument('--point-size', type=float, default=2.0)
    p.add_argument('--overlay-sparse', action='store_true')
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--show', action='store_true')
    a=p.parse_args()
    if a.point_budget<=0 or a.point_size<=0: p.error('point budget/size must be positive')
    return a


def _source_colors(idx):
    palette=np.asarray([[255,75,75],[255,145,55],[255,215,65],[160,235,70],[65,220,105],[55,220,205],[70,155,255],[120,105,255],[195,90,245],[245,90,175]],dtype=np.uint8)
    v=np.asarray(idx,dtype=np.int64)
    if v.ndim!=1 or np.any(v<0) or np.any(v>=10): raise ValueError('source_view_index must contain 0..9')
    return palette[v]


def main():
    a=_args(); d=a.d4_3_dir.expanduser().resolve()
    summary=json.loads((d/'d4_3_summary.json').read_text(encoding='utf-8'))
    if not bool(summary.get('pass')) or not summary.get('selected_artifacts'):
        raise RuntimeError('D4.3 preview requires a passing selected repaired chain.')
    cloud=Path(summary['selected_artifacts']['combined_npz']).expanduser().resolve()
    with np.load(cloud,allow_pickle=False) as p:
        pts=np.asarray(p['points_reconstruction'],dtype=np.float64); rgb=np.asarray(p['colors_rgb'],dtype=np.uint8)
        src=np.asarray(p['source_view_index'],dtype=np.uint8)
        sparse=np.asarray(p.get('sparse_landmarks_reconstruction',np.empty((0,3))),dtype=np.float64)
        orders=np.asarray(p['selected_workspace_orders'],dtype=np.int32)
    colors=rgb if a.color_mode=='rgb' else _source_colors(src)
    pts,colors=enforce_point_budget(pts,colors,int(a.point_budget),seed=0)
    if a.overlay_sparse and sparse.size:
        pts=np.vstack([pts,sparse]); colors=np.vstack([colors,np.tile(np.asarray([[245,245,245]],dtype=np.uint8),(sparse.shape[0],1))])
    app=QApplication.instance() or QApplication(sys.argv[:1]); w=QMainWindow(); w.setWindowTitle('EIA TECH — D4.3 Repaired Ten-View Chain')
    c=QWidget(); l=QVBoxLayout(c)
    selected=summary['selection']['selected_candidate']
    l.addWidget(QLabel(f"D4.3 selected candidate {selected} | orders {'→'.join(map(str,orders.tolist()))} | {pts.shape[0]:,} rendered | color={a.color_mode} | NO fusion | arbitrary SfM gauge"))
    canvas=PointCloudCanvas(w); canvas.set_point_size(float(a.point_size)); canvas.set_grid_visible(True); canvas.update_points(pts,colors); canvas.fit_scene(); l.addWidget(canvas.native,1)
    w.setCentralWidget(c); w.resize(1366,768); w.show()
    for _ in range(14): app.processEvents(); time.sleep(.02)
    out=a.output.expanduser().resolve() if a.output is not None else d/f'd4_3_selected_{a.color_mode}.png'; out.parent.mkdir(parents=True,exist_ok=True)
    if not w.grab().save(str(out),'PNG'): canvas.close(); raise RuntimeError(f'Unable to save D4.3 preview: {out}')
    print(f'D4.3 existing-PointCloudCanvas preview: {out}')
    print(f"selected_candidate={selected} orders={'->'.join(map(str,orders.tolist()))} rendered_points={pts.shape[0]} color_mode={a.color_mode} overlay_sparse={bool(a.overlay_sparse)}")
    print('Selected repaired ten-view raw accumulation only: no fusion, filtering mutation, persistence, or authority promotion.')
    if a.show:
        code=app.exec(); canvas.close(); w.close(); return int(code)
    canvas.close(); w.close(); return 0

if __name__=='__main__': raise SystemExit(main())
