# I5.5A — Dynamic-Scene Scenario Admission + Stress Design

I5.5A admits a paired static-control / controlled-dynamic-proxy experiment for I5.5B. It is admission-only: no stress images are materialized and no SfM or dense provider is executed here.

The source is an operator-confirmed ordered static multiview control. I5.5A freezes 19 views and a deterministic source-derived opaque foreground patch: donor view index 9, normalized donor crop `(x=0.38, y=0.36, w=0.24, h=0.22)`, translated independently across views at `y=0.58` from `x=0.03` to `x=0.73`. Control and stress are required to use the same decoded source pixels and lossless PNG materialization in I5.5B.

This is a controlled independent-foreground-motion proxy, **not natural dynamic-scene ground truth**. I5.5B must revalidate source identity, patch identity/trajectory, changed-pixel confinement, feature-bearing dynamic content, identical production SfM configuration, and control acceptance before any dynamic robustness conclusion. No masking, segmentation rescue, motion-outlier filter, threshold relaxation, production-default modification, MP-SfM, Auto, Hybrid, metric scale, metrology, or mesh authority is introduced.
