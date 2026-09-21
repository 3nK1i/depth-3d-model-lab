# I6.2C — End-to-End Offline Result Handoff

## Purpose

I6.2C formalizes the terminal product boundary between the existing isolated
offline worker, reconstruction persistence, I6 workspace state, and the
existing sparse-result viewer/export presentation.

This step does not change reconstruction algorithms or add dense execution.
It does not implement progressive SfM visualization; that remains I6.6.

## Contract

A completed worker result is forwarded to product/UI consumers only after:

1. the result is a completed passing mapping,
2. the result session UUID matches the admitted worker session,
3. the output directory matches the admitted worker output directory,
4. certified-metrology claims remain absent,
5. persistent-scene readiness cannot exist without canonical-geometry authority,
6. the persistence controller accepts the result.

The accepted handoff classifies sparse SfM with camera poses but no persistent
canonical scene as `partial`.

Optional visualization/export artifacts are presentation evidence, not scene
authority. Missing, relative, or out-of-root optional artifacts are quarantined
from viewer/export handoff and recorded as warnings; they do not downgrade an
otherwise valid scientific reconstruction result.

## Preserved boundaries

- PyCOLMAP remains canonical camera/global-frame authority.
- Sparse SfM stays partial and arbitrary-scale unless separate evidence says otherwise.
- Dense execution is not authorized by I6.2C.
- Persistent-scene promotion is not authorized by I6.2C.
- Production Auto remains deferred/unqualified/unauthorized.
- Certified metrology is not claimed.
- No provider selection, scientific threshold, or reconstruction algorithm changes.
- No progressive point-cloud streaming is added; I6.6 owns that feature.

## Validation

The step includes dependency-light result-handoff tests, controller/Qt tests,
end-to-end UI handoff tests, predecessor regressions, and full-unit closure.
A real UI-launched reconstruction remains required before formal closure.
