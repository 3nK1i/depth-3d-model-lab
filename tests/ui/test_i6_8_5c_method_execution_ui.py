"""Workstation Qt integration: real page seam, mocked *launch*, no heavy jobs."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('PyQt6')

from app.reconstruction_controller import ReconstructionWorkflow
from app.widgets.reconstruction_page import ReconstructionPage
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_method_selection_contract import ReconstructionMethod


class _Controller:
    is_shutdown = False
    is_active = False
    dense_ready_for_explicit_start = False
    persistent_scene = None

    def __init__(self, path: Path, workflow: ReconstructionWorkflow):
        self.source_selection = SimpleNamespace(workflow=workflow, ready=True, path=path)
        self.last_offline_one_start_preflight = None
        self.last_offline_result_handoff = None
        self.launched: list[str] = []

    def start_selected_da3_canonical(self):
        self.launched.append('da3_quick')
        return 'quick-session'

    def start_explicit_post_sparse_dense(self):
        self.launched.append('colmap_mvs')
        return 'dense-run'


@pytest.fixture
def prepared_page(qtbot, tmp_path):
    page = ReconstructionPage()
    qtbot.addWidget(page)
    video = tmp_path / 'test.mp4'
    video.write_bytes(b'fixture, not decoded')
    controller = _Controller(video, ReconstructionWorkflow.RECORDED_VIDEO)
    page._controller = controller
    return page, controller, video


def test_da3_quick_requires_external_runtime_evidence_and_rechecks_jobs(prepared_page):
    page, controller, _ = prepared_page
    with pytest.raises(ReconstructionValidationError, match='runtime_not_verified'):
        page.start_selected_reconstruction_method(ReconstructionMethod.DA3_QUICK)
    assert controller.launched == []
    assert page.start_selected_reconstruction_method(
        ReconstructionMethod.DA3_QUICK,
        runtime_verified_methods=(ReconstructionMethod.DA3_QUICK,),
    ) == 'quick-session'
    assert controller.launched == ['da3_quick']
    controller.is_active = True
    with pytest.raises(ReconstructionValidationError, match='another_job_active'):
        page.start_selected_reconstruction_method(
            ReconstructionMethod.DA3_QUICK,
            runtime_verified_methods=(ReconstructionMethod.DA3_QUICK,),
        )
    assert controller.launched == ['da3_quick']


def test_da3_full_scene_uses_actual_page_supervisor_after_exact_method_admission(
    prepared_page, monkeypatch,
):
    page, controller, video = prepared_page
    calls = []
    import app.widgets.reconstruction_page as page_module
    def fake_builder(**kwargs):
        calls.append(('builder', kwargs['video_path']))
        return 'existing-job-spec', 'existing-command'
    def fake_supervisor(spec, command):
        calls.append(('supervisor', spec, command))
        return 'full-scene-job'
    monkeypatch.setattr(page_module, 'build_full_scene_da3_job', fake_builder)
    monkeypatch.setattr(page, 'start_digital_twin_processing_job', fake_supervisor)
    assert page.start_selected_reconstruction_method(
        ReconstructionMethod.DA3_FULL_SCENE,
        runtime_verified_methods=(ReconstructionMethod.DA3_FULL_SCENE,),
    ) == 'full-scene-job'
    assert calls == [('builder', video), ('supervisor', 'existing-job-spec', 'existing-command')]
    assert controller.launched == []
    image_dir = video.parent / 'images'
    image_dir.mkdir()
    controller.source_selection.workflow = ReconstructionWorkflow.IMAGE_SEQUENCE
    controller.source_selection.path = image_dir
    with pytest.raises(ReconstructionValidationError, match='incompatible_source_workflow'):
        page.start_selected_reconstruction_method(
            ReconstructionMethod.DA3_FULL_SCENE,
            runtime_verified_methods=(ReconstructionMethod.DA3_FULL_SCENE,),
        )
    assert len(calls) == 2


def test_colmap_requires_current_sparse_admission_and_reuses_existing_launch(prepared_page, monkeypatch):
    page, controller, video = prepared_page
    released = []
    monkeypatch.setattr(page, '_release_appearance_resources_for_heavy_handoff', released.append)
    with pytest.raises(ReconstructionValidationError, match='admitted_sparse_required'):
        page.start_selected_reconstruction_method(
            ReconstructionMethod.COLMAP_MVS,
            runtime_verified_methods=(ReconstructionMethod.COLMAP_MVS,),
        )
    assert released == []
    controller.dense_ready_for_explicit_start = True
    controller.last_offline_one_start_preflight = SimpleNamespace(
        session_id='sparse-id', source_path=str(video), workflow='recorded_video',
    )
    controller.last_offline_result_handoff = SimpleNamespace(session_id='sparse-id')
    assert page.start_selected_reconstruction_method(
        ReconstructionMethod.COLMAP_MVS,
        runtime_verified_methods=(ReconstructionMethod.COLMAP_MVS,),
    ) == 'dense-run'
    assert released == ['Dense reconstruction']
    assert controller.launched == ['colmap_mvs']


def test_colmap_refuses_stale_sparse_from_different_selected_source(prepared_page):
    page, controller, video = prepared_page
    controller.dense_ready_for_explicit_start = True
    controller.last_offline_one_start_preflight = SimpleNamespace(
        session_id='old-sparse-id', source_path=str(video), workflow='recorded_video',
    )
    controller.last_offline_result_handoff = SimpleNamespace(session_id='old-sparse-id')
    controller.source_selection.path = video.parent / 'different.mp4'
    controller.source_selection.path.write_bytes(b'different source')
    with pytest.raises(ReconstructionValidationError, match='admitted_sparse_required'):
        page.start_selected_reconstruction_method(
            ReconstructionMethod.COLMAP_MVS,
            runtime_verified_methods=(ReconstructionMethod.COLMAP_MVS,),
        )
    assert controller.launched == []
