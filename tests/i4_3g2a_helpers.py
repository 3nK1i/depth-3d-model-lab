from __future__ import annotations

from tests.i4_3g1_helpers import passing_i4_3a_snapshot, passing_i4_3f3_snapshot
from core.reconstruction.cross_provider_qualification import build_i4_3g1_framework_snapshot


def passing_i4_3g1_snapshot() -> dict:
    return build_i4_3g1_framework_snapshot(
        i4_3a_snapshot=passing_i4_3a_snapshot(),
        i4_3f3_snapshot=passing_i4_3f3_snapshot(),
    )
