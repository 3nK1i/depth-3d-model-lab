# Phase 8.9I1 — Final Release Preflight

## Baseline

- Branch: `phase8-reconstruction`
- HEAD / origin: `f959c369f8aa27347cab20cc90b469c6674d0840`
- Worktree: clean
- Local `v0.8.0` tag: absent
- Remote `v0.8.0` tag: absent
- Runtime/package identity before transition: `0.7.5` / `Phase 7.5 Stable` / `7.5`
- Audit report SHA-256: `04d4e9e1fda1852035e3c668fe19f544b36f34f592f9a20e2e1dea7d2ad92ac3`
- I2 source snapshot SHA-256: `f626c9966de3e075f9fcebad8ae75fb5c0add8080dc372b5df08c87592103e31`

## Findings

I1 confirmed that `core/version.py`, `pyproject.toml`, and `config/app.yaml` are the canonical coordinated release identity. `main.py` consumes the central version and release name rather than hardcoding them, and `core/runtime/release.py` performs generic consistency/reporting against the central metadata.

The audit also identified historical Phase 7/7.5/8/G/H tests that intentionally pinned the pre-release `0.7.5` identity. The I2 transition must update current-runtime assertions while preserving the historical G/H runner and evidence files themselves. A global search/replace is explicitly forbidden.

## I2 boundary

8.9I2 is allowed to change release metadata, release-facing documentation, and current regression assertions required by the `0.8.0` transition. It must not alter reconstruction implementation, application behavior, external backend boundaries, historical G/H runners, or create/push any Git tag.

The final annotated `v0.8.0` tag remains reserved for 8.9I3 after I2 is fully green, committed, pushed, and synchronized.
