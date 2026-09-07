# I5.6A - MVS Robustness Scenario Admission + Envelope Design

I5.6A is admission-only. It reuses frozen authoritative PyCOLMAP SfM from the Castle-P19 I5.2/I5.3/I5.5 campaigns, verifies the exact registered staged images and selected sparse model required by the existing production MVS adapter, and freezes the I5.6B MVS-only replay matrix.

Admitted cases are control 19/19, low overlap 6/7, weak parallax 7/7, overexposure 19/19, symmetry 12/19, and controlled dynamic proxy 19/19. Blur and underexposure are upstream-blocked because no authoritative SfM model exists; KITTI forward motion is upstream-blocked by incremental mapping with no sparse model; clutter remains not formally admitted because its semantic stress proxy was not revalidated.

I5.6B must use the existing adaptive `COLMAP_MVS` production path, geometric consistency ON, current production cache/image-size defaults, sequential execution, no engineer override, and no permanent global view cap. TempleRing 47/47 remains a historical read-only positive anchor and is not rerun by I5.6A.

No SfM or MVS execution occurs in I5.6A. No DA3 comparison, threshold relaxation, result-driven retuning, provider promotion, Auto, Hybrid, MP-SfM, metric-scale claim, measurement authority, certified metrology, or mesh authority is introduced.
