from __future__ import annotations

from copy import deepcopy

from core.reconstruction.i4_3_final_qualification import (
    H1EvidenceBinding,
    build_i4_3h1_final_qualification_audit,
)
from tests.i4_3h1_helpers import passing_g5b, passing_regression_receipt


def passing_h1() -> dict:
    return build_i4_3h1_final_qualification_audit(
        i4_3g5b_closure=passing_g5b(),
        regression_receipt=passing_regression_receipt(focused_tests=40),
        evidence_bindings={
            "i4_3g5b_closure": H1EvidenceBinding(path="/evidence/g5b.json", sha256="a" * 64),
            "i4_3h1_regression_receipt": H1EvidenceBinding(
                path="/evidence/h1-regression.json", sha256="b" * 64
            ),
        },
        g5b_predecessor_hashes_reverified=True,
    )


def clone(value: dict) -> dict:
    return deepcopy(value)
