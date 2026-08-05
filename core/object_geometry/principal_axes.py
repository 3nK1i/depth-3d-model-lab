"""Robust deterministic principal-axis fitting."""
from __future__ import annotations
import numpy as np
from .fitting_contracts import GeometryDegeneracy, OrientationStability, PrincipalAxesEstimate
from .fitting_policy import RobustPcaPolicy
from .robust_statistics import normalized_weights, robust_centre, trim_extreme_tails, weighted_mean

def _canonicalize(R):
    R=np.asarray(R,dtype=np.float64).copy()
    for i in range(2):
        j=int(np.argmax(np.abs(R[:,i])))
        if R[j,i]<0: R[:,i]*=-1
    R[:,2]=np.cross(R[:,0],R[:,1]); R[:,2]/=np.linalg.norm(R[:,2])
    R[:,1]=np.cross(R[:,2],R[:,0]); R[:,1]/=np.linalg.norm(R[:,1])
    if np.linalg.det(R)<0: R[:,2]*=-1
    return R

def _classify(vals, policy):
    if vals[0] <= 1e-15: return GeometryDegeneracy.POINT_LIKE
    r2=vals[1]/vals[0]; r3=vals[2]/vals[0]
    if r2 < policy.linear_ratio_threshold: return GeometryDegeneracy.LINEAR
    if r3 < policy.planar_ratio_threshold: return GeometryDegeneracy.PLANAR
    return GeometryDegeneracy.NONE

def _stability(vals, policy):
    if vals[0]<=1e-15: return (OrientationStability.UNAVAILABLE,)*3
    g1=(vals[0]-vals[1])/max(vals[0],1e-15); g2=(vals[1]-vals[2])/max(vals[0],1e-15)
    a=OrientationStability.STABLE if g1>=policy.stable_axis_gap_threshold else OrientationStability.AMBIGUOUS
    c=OrientationStability.STABLE if g2>=policy.stable_axis_gap_threshold else OrientationStability.AMBIGUOUS
    b=OrientationStability.STABLE if min(g1,g2)>=policy.stable_axis_gap_threshold else OrientationStability.WEAK
    return (a,b,c)

def fit_principal_axes(cloud, policy=RobustPcaPolicy()):
    p=np.asarray(cloud.points_xyz_m,dtype=np.float64); w=cloud.point_weights if policy.use_point_weights else None
    if len(p)<policy.minimum_points: raise ValueError("too few points for principal-axis fitting")
    keep=trim_extreme_tails(p,policy.trim_mad_multiplier,policy.minimum_points) if policy.trim_extreme_tails else np.ones(len(p),bool)
    q=p[keep]; qw=None if w is None else np.asarray(w)[keep]
    c=robust_centre(q,qw); centroid=weighted_mean(q,qw); wn=normalized_weights(qw,len(q)); d=q-c
    cov=(d*wn[:,None]).T@d; vals,vecs=np.linalg.eigh(cov); order=np.argsort(vals)[::-1]; vals=np.maximum(vals[order],0); R=_canonicalize(vecs[:,order])
    total=float(vals.sum()); ratios=vals/total if total>0 else np.array([1.,0.,0.]); gaps=np.array([(vals[0]-vals[1])/max(vals[0],1e-15),(vals[1]-vals[2])/max(vals[0],1e-15)])
    return PrincipalAxesEstimate(cloud.observation_id,cloud.coordinate_frame_id,c,centroid,R,vals,ratios,gaps,_stability(vals,policy),_classify(vals,policy),len(q),warnings=("robust tail trimming applied",) if not np.all(keep) else ())
