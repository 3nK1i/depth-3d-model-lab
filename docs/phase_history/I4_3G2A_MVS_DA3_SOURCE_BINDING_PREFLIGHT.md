# I4.3G2A — MVS/DA3 source and evidence binding preflight

G2A is the read-only preflight introduced after G1 froze logical scenario IDs but intentionally did not invent concrete filesystem dataset bindings.

It does **not** run SfM, MVS, DA3, Auto, hybrid fusion, or quality scoring. It verifies the G1 methodology snapshot, binds the already-known TempleRing authority chain when present, verifies the historical House A/B only as read-only context when its exact closure is available, and discovers candidate image directories for the fresh complete-room, low-texture, and weak-overlap scenarios.

A directory name is never treated as proof that a capture is a complete room, low texture, or weak overlap. Those three fresh scenario semantics require explicit source binding before G2B may start heavy execution.

Authority remains unchanged: PyCOLMAP owns the canonical frame/cameras; no permanent global view cap, provider promotion, Auto decision, hybrid authorization, metric claim, measurement authority, certified metrology, or mesh authority is introduced.

- Historical House verification follows the I4.2F content-addressed I4.2A chain, preferring the closure-local frozen copy and falling back to the byte-verified persistent source copy.

### House historical provenance resolution

House historical A/B evidence is read-only context.  G2A verifies it through the
persistent I4.2F chain, not by requiring the original `/tmp` closure to survive.
The resolver accepts either of the content-addressed I4.2F layouts already used
by the project:

- `frozen_summaries.i4_2a_summary` embedded in the I4.2F closure; or
- the SHA-pinned `i4_2f_evidence_manifest.json`, where the same record is under
  `summaries.i4_2a_summary` (or `frozen_summaries`).

For either layout, the selected I4.2A summary bytes must match their recorded
SHA256 and must preserve the frozen House closure SHA256 plus verdict
`mvs_better_but_imperfect`. A missing or tampered manifest fails closed.

### Recovered House closure contract

The original House I4.1.9A-A3.1 closure was written under `/tmp`; its original
byte SHA256 remains frozen as
`0fbc46280ec38e571a92c15f48eda9f6103735e6a8915c80798b58a7f2330b50`, but
the original bytes are no longer required to exist.  The persistent I4.2A
summary may instead point to a recovered semantic copy with a different byte
SHA256.  G2A accepts that recovery only when the complete content-addressed
chain is valid:

1. I4.2F closure is passing and authority-clean.
2. The I4.2F evidence manifest, when used, matches the SHA256 recorded by I4.2F.
3. The selected I4.2A summary matches its recorded SHA256.
4. The recovered House closure exists and matches the distinct SHA256 recorded
   by that I4.2A summary.
5. The recovered closure is a passing `I4.1.9A-A3.1` record whose manual verdict
   remains `mvs_better_but_imperfect` and whose comparison invariants remain
   intact.
6. `recovery_provenance.artifact_role` is exactly
   `recovered_semantic_copy_of_previously_passed_I4.1.9A_A3.1_closure`.
7. `recovery_provenance.original_closure_sha256_recorded_by_I4.2A` remains the
   original deleted-byte SHA256 above, and `original_bytes_available` is
   explicitly `false`.

The recovered artifact is never represented as byte-identical to the deleted
original.  Evidence records preserve both identities separately.  Any wrong
recovered SHA, wrong original SHA, wrong verdict, missing recovery provenance,
claim that original bytes remain available, or arbitrary replacement JSON
fails closed.

## G2A.3 fresh-scenario binding

The final three fresh semantic bindings are produced by
`scripts/run_i4_3g2a3_fresh_scenario_binding.py`. The generated snapshot keeps
complete-room semantics engineer-confirmed, derives or validates low-texture
and weak-overlap evidence under a frozen pre-provider diagnostic policy, and
content-addresses both source images and visual-review contact sheets.

The final G2A preflight accepts that artifact through
`--fresh-binding-snapshot`. It rehashes the bound files/review evidence and
recomputes the objective diagnostics before setting
`i4_3g2b_real_execution_authorized=true`. G2A.3 never authorizes G2B by itself.
