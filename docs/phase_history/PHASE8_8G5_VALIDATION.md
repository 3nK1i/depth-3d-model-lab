# Phase 8.8G5 — Progressive Disclosure / UI Polish

G5 reduces Reconstruction sidebar density without changing reconstruction behavior.

- Every existing settings section remains available in-place.
- Workflow, Session & Persistence, Inspection Target, Appearance Representations, and Authority & Safety remain expanded by default.
- Measurement/detail/QC/reference and backend-preview detail groups start collapsed and can be expanded independently.
- Existing context visibility for Offline Inspection and Semantic / Objects is preserved.
- Collapsing a group hides presentation rows only; it does not disable, clear, recompute, persist, resume, retry, or execute anything.
- G3 checkpoint/resume and G4 failure/recovery actions are unchanged.
- No `core/reconstruction` change, backend import, authority promotion, or domain-specific branch is introduced.

Validation: focused contract/UI tests, chained quick/protocol/backward suites, a dependency-light UI smoke, 1366×768 preview, and normal application startup/exit.
