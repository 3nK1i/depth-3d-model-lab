# I5.4A — Forward-Motion / KITTI Scenario Admission

I5.4A is an execution-free admission step. It starts only from a frozen I5.3C closure that explicitly authorizes I5.4. No sparse reconstruction, dense-provider execution, threshold relaxation, production-default change, MP-SfM, Hybrid, or Auto work occurs here.

The source is an operator-confirmed ordered KITTI-style forward-motion image sequence. That confirmation is provenance, not camera-motion ground truth. I5.4B must independently revalidate the motion geometry from the existing PyCOLMAP-authoritative reconstruction before any forward-motion robustness conclusion is allowed.

The frozen real-replay window is the first 60 native-order source images, contiguous at stride 1. I5.4A does not select a window after observing reconstruction quality and does not retry or tune the window until a preferred result appears. The source asset set and the admitted 60-image window are independently SHA-bound.

I5.4B must report registration coverage, landmarks, observations, reprojection error and track health. It must also derive consecutive camera-center motion in camera coordinates and test whether median forward-axis motion dominates median lateral motion. I2.4 parallax and baseline-to-range diagnostics are reported as scale-invariant triangulation-health evidence; they do not establish metric scale.

The ordered-sequence replay is frozen to sequential matching with overlap 10 and nominal image-sequence FPS 10.0 for the controlled experiment. These are experiment bounds only and do not modify production defaults.

If I5.4B does not revalidate forward-axis dominance, the result remains an inconclusive forward-motion proxy; the input window must not be retuned from the observed result. I5.4A authorizes only I5.4B. I5.5 remains unauthorized.
