from pathlib import Path

from core.reconstruction import materialize_appearance_dataset, PipelineIngestionEvidence, PipelineCameraTrajectoryEvidence
from tests.reconstruction_f3_helpers import make_f3_fixture


def test_f4_dataset_materialization_is_deterministic_when_rebuilt_in_place(tmp_path: Path) -> None:
    f=make_f3_fixture(request_id="f4-stress")
    ing=f.store.find_unique(PipelineIngestionEvidence)
    traj=f.store.find_unique(PipelineCameraTrajectoryEvidence)
    root=(tmp_path/"shared").resolve()
    a=materialize_appearance_dataset(ing,traj,output_dir=root)
    bytes_a=[Path(frame.image_path).read_bytes() for frame in a.dataset.frames]
    b=materialize_appearance_dataset(ing,traj,output_dir=root)
    bytes_b=[Path(frame.image_path).read_bytes() for frame in b.dataset.frames]
    assert a.dataset_fingerprint_sha256==b.dataset_fingerprint_sha256
    assert bytes_a==bytes_b
    assert [x.T_reconstruction_from_camera.tolist() for x in a.dataset.frames]==[x.T_reconstruction_from_camera.tolist() for x in b.dataset.frames]
