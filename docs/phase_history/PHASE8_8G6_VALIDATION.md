# Phase 8.8G6 — Full Reconstruction UX Review

G6 is a **review-only** closure step for the completed Reconstruction UX stack. Source inspection found no production behavior change required after G1–G5. G6 therefore adds only deterministic review fixtures, contract/UI acceptance tests, a decisive mock-only smoke, and an interactive 1366×768 preview.

## Acceptance scope

- One coherent Reconstruction information hierarchy from workflow through authority/safety.
- Idle, ready-source, terminal-failure, persistent-scene, checkpoint/resume, and preserved-evidence states remain understandable and fail closed where required.
- G2 scene actions, G3 Prepare Resume, and G4 Retry From Source remain distinct.
- G5 disclosure changes visibility only; it does not alter action enablement or controller state.
- Canonical geometry remains authoritative, metrology remains explicitly uncertified, and appearance remains presentation-only.
- Settings remain vertically scrollable with no horizontal scrollbar at the 1366×768 target.
- No new backend execution, recovery engine, resume execution, persistence format, domain-specific branching, or reconstruction authority is introduced by G6.

## Review gates

Run `scripts/run_phase8_8g6_tests.sh` in `quick`, `protocol`, and `backward` modes, then run `scripts/run_phase8_8g6_full_ux_review_smoke.py`, inspect its JSON/screenshot, and finish with `scripts/run_phase8_8g6_full_ux_review_preview.py` plus a normal `python main.py` open/close sanity.
