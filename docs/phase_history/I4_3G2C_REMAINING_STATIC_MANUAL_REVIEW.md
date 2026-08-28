# I4.3G2C Remaining Static Manual Review Coverage

## Purpose

G2C must satisfy the qualification methodology frozen in I4.3G1 before the
G2 result is synthesized.  The G1 scenario matrix marks manual review as
required for Complete Room, TempleRing, House, Low Texture, and Weak Overlap.
TempleRing has already completed blinded matched-view review and controlled
reveal. House already contains a frozen historical manual review and remains
historical/read-only.

The remaining new G2C visual work is therefore limited to:

- `g2_complete_room_mvs_da3`
- `g2_low_texture_mvs_da3`
- `g2_weak_overlap_mvs_da3`

No provider or SfM rerun is authorized.

## Scope preservation

Complete Room and Low Texture are not promoted to full-scene comparisons.
Their G2B closures established authoritative-SfM coverage limitations while
preserving localized registered-view provider evidence. Their G2C matched
views are explicitly labeled and persisted as
`localized_authoritative_registered_views`.

Weak Overlap preserves its connected five-view authoritative registered scope.
Its review scope is `authoritative_registered_views`.

## Review controls

The generic G2C runner:

1. verifies the exact frozen G1 snapshot and manual-review requirement;
2. verifies the exact formal G2B closure and scenario verdict SHA256;
3. verifies both providers share one PyCOLMAP-authoritative SfM identity;
4. derives one common visual frame from authoritative sparse landmarks only;
5. applies identical fixed camera/framing/raster policy;
6. applies identical deterministic draw count to A and B;
7. writes only a blind-assignment commitment before review;
8. never reruns or promotes a provider.

Low Texture may be visually sparse because its frozen MVS artifact contains
very little localized geometry. Reviewers may use `not_assessable`; no post-hoc
render-density or provider threshold tuning is permitted.

## House historical evidence

House is not rerun. The recovered I4.1.9A-A3.1 closure already records a
confirmed manual matched-view assessment across oblique/front/side/top views
with the same authoritative SfM frame and shared camera parameters. G2 final
synthesis must cite it as historical evidence, not a new G2 execution.
