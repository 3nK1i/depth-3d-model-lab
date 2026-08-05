"""Dependency-light robust statistics used by Step 5."""
from __future__ import annotations
import numpy as np

def normalized_weights(weights, n):
    if weights is None: return np.full(n,1.0/n,dtype=np.float64)
    w=np.asarray(weights,dtype=np.float64)
    if w.shape!=(n,) or not np.isfinite(w).all() or np.any(w<0) or float(w.sum())<=0: return np.full(n,1.0/n,dtype=np.float64)
    return w/w.sum()

def weighted_mean(points, weights=None):
    p=np.asarray(points,dtype=np.float64); return np.sum(p*normalized_weights(weights,len(p))[:,None],axis=0)

def weighted_median_1d(values, weights=None):
    v=np.asarray(values,dtype=np.float64); w=normalized_weights(weights,len(v)); order=np.argsort(v,kind="mergesort"); v=v[order]; w=w[order]
    return float(v[np.searchsorted(np.cumsum(w),0.5,side="left")])

def robust_centre(points, weights=None):
    p=np.asarray(points,dtype=np.float64); return np.array([weighted_median_1d(p[:,i],weights) for i in range(3)],dtype=np.float64)

def trim_extreme_tails(points, multiplier=4.5, minimum_points=32):
    p=np.asarray(points,dtype=np.float64); med=np.median(p,axis=0); mad=np.median(np.abs(p-med),axis=0); scale=np.where(mad>1e-12,1.4826*mad,1e-12)
    keep=np.all(np.abs(p-med)<=multiplier*scale,axis=1)
    if int(keep.sum())<minimum_points: keep=np.ones(len(p),dtype=bool)
    return keep
