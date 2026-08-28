from copy import deepcopy
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temporal_provider_closure import build_i4_3d3_temporal_provider_closure
from tests.i4_3d3_helpers import d1_snapshot, d2_snapshot


@pytest.mark.parametrize("key", ["metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated"])
def test_i43d3_rejects_authority_promotion_in_either_temporal_binding(key: str) -> None:
    for source in ("d1", "d2"):
        d1 = deepcopy(d1_snapshot())
        d2 = deepcopy(d2_snapshot())
        (d1 if source == "d1" else d2)["authority"][key] = True
        with pytest.raises(ReconstructionValidationError):
            build_i4_3d3_temporal_provider_closure(i4_3d1_snapshot=d1, i4_3d2_snapshot=d2)


@pytest.mark.parametrize("source", ["d1", "d2"])
def test_i43d3_rejects_global_view_cap(source: str) -> None:
    d1 = deepcopy(d1_snapshot())
    d2 = deepcopy(d2_snapshot())
    (d1 if source == "d1" else d2)["architecture"]["permanent_global_view_cap"] = 40
    with pytest.raises(ReconstructionValidationError, match="permanent global view cap"):
        build_i4_3d3_temporal_provider_closure(i4_3d1_snapshot=d1, i4_3d2_snapshot=d2)
