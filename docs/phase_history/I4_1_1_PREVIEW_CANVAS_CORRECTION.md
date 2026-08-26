# I4.1.1 Preview Canvas Contract Correction

The first real diagnostic preview failed before rendering because it instantiated
`PointCloudCanvas` and then called APIs owned by `ReconstructionCanvas`, including
`sparse_landmarks` and canonical camera trajectory presentation.

The correction is presentation-only:

- use the real `ReconstructionCanvas` used by the Reconstruction workspace;
- load the Dense RGB cloud through `set_dense_rgb_points`;
- keep the robust Core target as a magenta sparse landmark;
- keep registered/aligned/selected cameras on the canonical camera trajectory layer;
- add a static integration contract that fails if the preview drifts back to the
  wrong canvas class or calls unavailable APIs.

No I4.1/I4.1.1 diagnostic evidence, thresholds, SfM, DA3, fusion, filtering,
persistence, metric/measurement authority, I4.2 authorization, or mesh authority
is changed by this correction.
