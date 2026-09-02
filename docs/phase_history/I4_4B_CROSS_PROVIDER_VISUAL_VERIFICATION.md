# I4.4B — Cross-Provider Visual Verification

## Purpose

I4.4B closes the **visual-verification** boundary for the frozen offline dense
foundation. It is deliberately evidence-only: it consumes the frozen I4.3
visual-family closures together with the formal I4.3H2 closure and the passing
I4.4A focused regression result, then verifies that visual-review coverage
across all executable provider families remains complete and still obeys the
same authority/provenance locks.

## Scope

I4.4B consumes only frozen evidence from:

- **I4.3G2** — MVS / DA3 visual-family closure
- **I4.3G3C** — temporal matched-view review closure
- **I4.3G4C** — single-image matched-view review closure
- **I4.3H2** — formal I4.3 closure
- **I4.4A** — focused offline dense foundation regression

It does **not**:

- rerun COLMAP MVS, DA3, RollingDepth, Video Depth Anything, or any single-image provider;
- rerun PyCOLMAP / SfM;
- regenerate matched views or repeat manual review;
- aggregate visual preferences into a scalar score;
- create any cross-family provider ranking or universal winner;
- select/promote a provider, implement Auto, or authorize Hybrid;
- claim metric scale, measurement authority, certified metrology, or mesh authority;
- start I4.4C or I4.4D.

## What is verified

I4.4B verifies:

1. all three executable visual-review families are accounted for;
2. all **11 frozen qualification scenarios** remain covered;
3. all **10 provider implementations** remain represented;
4. fresh visual-review coverage remains **10 scenarios** plus **1 historical read-only** scenario;
5. the family-specific review-dimension contracts remain intact:
   - G2 static MVS/DA3 review dimensions (8)
   - G3 temporal review dimensions (6)
   - G4 single-image review dimensions (7)
6. the frozen terminal quality-event set remains exactly aligned with H2/I4.4A;
7. production-native evidence remains primary and matched-view/manual review remains secondary diagnostic evidence;
8. no scoring/ranking/promotion/authority drift has been introduced.

## Output

The formal output is:

- `i4_4b_cross_provider_visual_verification.json`

A passing result records:

- `i4_4b_complete=true`
- `i4_4c_authorized=true`
- `i4_4c_started=false`
- `i4_4d_started=false`

and the next authorized boundary becomes:

**I4.4C — Provenance / Authority Audit**
