# I4.3G2C — Blinded matched-view entry

G2C begins only from the formally closed I4.3G2B evidence chain. This step reuses the already frozen TempleRing production-native MVS and DA3 outputs and the existing shared-camera renderer; it does not rerun SfM or either dense provider.

## Correction from the dormant G2B diagnostic

The earlier dormant diagnostic was technically usable as a renderer, but it was still a G2B artifact and wrote `blind_key.json` beside the public review outputs. G2C now:

- verifies the exact final G2B closure SHA before any review artifact is generated;
- verifies the frozen TempleRing verdict identity through that closure;
- preserves the PyCOLMAP-only global-frame and camera-pose authority contract;
- keeps the A/B provider mapping out of the public review package;
- writes only a cryptographic assignment commitment plus the blinded manual-review template;
- forbids reveal until the blinded scenario ratings are complete and explicitly confirmed;
- keeps provider selection, promotion, Auto, hybrid, metric-scale, metrology, and mesh authority false.

The old `run_i4_3g2b_temple_ring_matched_view.py` remains dormant historical scaffolding and is not the G2C entry point. Use `run_i4_3g2c_temple_ring_matched_view.py` for G2C.

## G2C.1 equal-density blinded rendering correction

Before the first real matched-view render, the reviewer-facing render contract was tightened so
provider identity cannot leak through point-count text or unequal display density. Both frozen
provider clouds are now deterministically sampled to one common draw count:
`min(requested_budget, mvs_source_points, da3_source_points)`. Reviewer labels contain only the
blind A/B label, view/framing, and the shared-frame/shared-camera/equal-draw-count statement.
Provider-native source counts remain preserved in provenance metadata, but are not embedded in
A/B images. No SfM or dense provider is rerun and no provider result is modified.


## G2C.1 native-window capture extent correction

Real TempleRing rendering exposed a compositor/window-manager edge case: repeated
QMainWindow/VisPy captures requested at 1366x768 could be saved with slightly
different raster heights. The A/B combiner correctly refused unequal images, but
that stopped the review after the detail views. G2C now freezes the window with
`setFixedSize(1366, 768)` and normalizes every successful capture to the same
1366x768 reviewer raster before pairing. The combiner independently requires that
exact extent for both A and B and reports their dimensions if the contract is ever
violated again. This changes review presentation only; provider evidence, shared
SfM, camera/framing, equal point draw count, and blind assignment are unchanged.
