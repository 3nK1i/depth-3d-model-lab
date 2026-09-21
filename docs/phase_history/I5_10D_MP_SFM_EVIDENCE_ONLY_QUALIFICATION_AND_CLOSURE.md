# I5.10D — MP-SfM Evidence-Only Qualification and Closure

## Purpose

I5.10D closes the controlled matching-strategy evaluation without executing any new SfM or MVS work. It binds the frozen I5.10A admission, I5.10B preflight, and the single I5.10C real exhaustive-matching evaluation.

## Frozen observed result

The historical I5.4/I5.10 baseline used sequential matching with overlap 10 on the exact frozen 60-image KITTI-style window. Incremental mapping produced no sparse model and registered 0/60 images; the result was rejected.

The single I5.10C exhaustive-matching candidate used the same exact source bytes/window, the same PyCOLMAP/SIFT/camera/quality policy, and changed only the admitted matching policy. It produced a sparse model with 3/60 registered images, 42 landmarks, 109 observations, about 1.013 px mean reprojection error, and mean track length about 2.595. The production quality policy still rejected it because registration was only 5% and landmark density was low.

## Qualified conclusion

I5.10 demonstrates **matching-policy sensitivity with partial sparse recovery, but no production-quality recovery** on this one admitted counterfactual. Exhaustive matching changed the failure mode from no sparse model to a small sparse model; it did not recover an acceptable reconstruction.

This evidence does not establish exhaustive matching as universally superior, does not authorize a production-default change, does not establish metric scale or metrology, and does not authorize a hybrid or Auto policy.

The forward-motion diagnostic is also bounded: only one adjacent registered motion pair survived, so it is insufficient for a sequence-level geometry claim.

## Roadmap decision

I5.10 is closed with bounded evidence. I5.11 OOM / disk / crash / recovery robustness is authorized next. I5.12–I5.14 remain deferred.
