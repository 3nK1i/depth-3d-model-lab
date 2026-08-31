# I4.3G5B — Formal Cross-Provider Qualification Closure

## Scope

I4.3G5B is the evidence-only formal closure of I4.3G. It consumes the frozen I4.3G5A cross-provider synthesis only after independently re-verifying the full G1 → G2/G3/G4 → G5A evidence chain.

No provider inference, PyCOLMAP/SfM rerun, matched-view rendering, manual-review rerun, provider selection/promotion, global scoring/ranking, Auto policy, Hybrid authorization, metric-scale claim, metrology authority or surface-mesh generation is performed.

## Closure-chain verification

The production runner requires SHA256-bound artifacts for:

- I4.3G1 qualification framework;
- I4.3G2 formal closure;
- I4.3G3C formal closure;
- I4.3G4C formal closure;
- I4.3G5A synthesis.

The four predecessor artifacts are used to deterministically rebuild the G5A synthesis. The frozen G5A JSON must match that rebuild exactly, including its recorded predecessor paths/hashes, before G5B may close I4.3G. A valid-looking but altered G5A file therefore cannot close G.

## Preserved evidence

G5B preserves the three distinct qualification families, all 11 G1 scenarios and all 10 provider implementations. It retains the five frozen terminal quality events from G3/G4 as operating-envelope evidence, never as provider scores:

- RollingDepth degraded on temporal static sweep;
- RollingDepth rejected on temporal motion;
- Video Depth Anything degraded on temporal motion;
- Depth Anything V2 rejected on single-image low texture;
- Marigold rejected on single-image low texture.

No G2 scenario detail is invented where the historical G2 closure did not expose it.

## Authority locks

Formal G closure keeps:

- PyCOLMAP as sole authoritative offline camera/scene-frame owner;
- `permanent_global_view_cap=None`;
- no provider selected, promoted or ranked;
- no universal provider winner or global scalar score;
- Auto unimplemented;
- Hybrid unauthorized;
- no physical metric scale, measurement authority or certified metrology;
- no surface mesh.

## Closure result

A passing G5B artifact records `i4_3g_closed=true` and `i4_3h_authorized=true`. I4.3H is the next authorized boundary and remains responsible for the final provider-architecture closure/freeze; G5B does not perform H work.
