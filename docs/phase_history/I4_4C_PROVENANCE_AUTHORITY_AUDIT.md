# I4.4C — Provenance / Authority Audit

## Purpose

I4.4C is the final provenance and authority audit before I4.4D closes I4.
It is an additive, evidence-only boundary that verifies the formal I4.3
closure graph and the completed I4.4A/I4.4B chain remain content-addressed,
mutually consistent, and aligned with the current repository authority
contracts.

I4.4C does not redo provider qualification or visual review. It does not
execute a provider, rerun PyCOLMAP/SfM, regenerate matched views, repeat manual
review, create a provider ranking, implement Auto, authorize Hybrid, or perform
the final I4.4D closure.

## Provenance graph audited

The formal runner independently verifies:

1. the explicitly supplied I4.3H2 closure;
2. H2's bound I4.3H1 audit;
3. H1's bound I4.3G5B closure and H1 regression receipt;
4. all five G5B predecessor bindings;
5. the explicitly supplied I4.4A result and its bound H2/receipt;
6. the explicitly supplied I4.4B result and its bound H2, I4.4A,
   G2/G3C/G4C artifacts, and B receipt;
7. the I4.4C audit regression receipt.

The audit intentionally **does not** recursively rehash every historical raw
provider NPZ/image/native artifact. H1/H2 already froze the formal provenance
conclusions without making that stronger raw-artifact claim, and I4.4C
preserves the same boundary.

## Authority audit

I4.4C re-audits the live repository contract surfaces and requires them to
remain consistent with the frozen H1 architecture:

- PyCOLMAP/SfM remains the sole global camera/scene-frame authority;
- provider-native poses never become canonical camera authority;
- the default reconstruction scale remains arbitrary unless separate later
  calibration/metric evidence qualifies it;
- dense-provider execution itself creates no metric-scale, measurement,
  certified-metrology, or surface-mesh authority;
- Hybrid remains unauthorized;
- Auto remains unimplemented until the evidence-driven I5 boundary;
- no permanent global dense-view cap exists.

## Qualification guard

The five frozen scenario-specific terminal quality events remain preserved as
operating-envelope evidence. They are not converted into provider scores,
rankings, a universal winner, or an automatic-selection policy.

## Output and handoff

The formal artifact is:

- `i4_4c_provenance_authority_audit.json`

A passing audit records:

- `i4_4c_complete=true`
- `i4_4d_authorized=true`
- `i4_4d_started=false`

The next and only authorized boundary is:

**I4.4D — Final I4 Closure + Freeze**
