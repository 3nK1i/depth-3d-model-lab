"""I6.8A contract tests: telemetry is observational, bounded and run-bound."""
from __future__ import annotations

from dataclasses import replace

import pytest

from core.reconstruction.reconstruction_job_telemetry_contract import (
    JOB_TELEMETRY_MAX_LINE_BYTES,
    JOB_TELEMETRY_PREFIX,
    JobTelemetryObservation,
    JobTelemetryStreamGate,
    ResourceReading,
    StageWorkCounter,
    TelemetryValidationError,
    parse_job_telemetry_line,
    serialize_job_telemetry_line,
)

SID = "11111111-2222-4333-8444-555555555555"
RUN = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
STAMP = "2026-09-21T16:00:00+00:00"


def sample(**overrides):
    data = dict(
        source_session_id=SID,
        job_kind="post_sparse_dense",
        dense_run_id=RUN,
        producer="mvs_worker",
        sequence=1,
        observed_at_utc=STAMP,
        stage="dense_fusion",
        detail="Stereo fusion active",
        counter=StageWorkCounter("fusion_images_started", 1, 87),
        resource_readings=(
            ResourceReading("gpu_device_vram_used_bytes", 2**30, "observed", gpu_index=0),
            ResourceReading("process_rss_bytes", None, "unavailable"),
        ),
    )
    data.update(overrides)
    return JobTelemetryObservation(**data)


def test_serialization_round_trip_and_explicit_observation_only_fields():
    obj = sample()
    assert JobTelemetryObservation.from_dict(obj.as_dict()) == obj
    assert obj.as_dict()["schema"] == "depth3d.reconstruction.job_telemetry.v1"
    assert "final_dense_result_admitted" not in obj.as_dict()
    assert "metric_scale_claimed" not in obj.as_dict()
    assert "progress_fraction" not in obj.as_dict()
    assert "eta_seconds" not in obj.as_dict()
    assert obj.resource_readings[0].scope == "gpu_device"
    line = serialize_job_telemetry_line(obj)
    assert line.startswith(JOB_TELEMETRY_PREFIX)
    assert parse_job_telemetry_line(line) == obj
    assert parse_job_telemetry_line("ordinary log output") is None
    assert len(line.encode()) <= JOB_TELEMETRY_MAX_LINE_BYTES


@pytest.mark.parametrize("change", [
    {"source_session_id": "foreign"},
    {"dense_run_id": "foreign"},
    {"dense_run_id": None},
    {"producer": ""},
    {"producer": "any code"},
    {"sequence": True},
    {"sequence": 0},
    {"observed_at_utc": "2026-09-21T16:00:00"},
    {"observed_at_utc": "2026-09-21T17:00:00+01:00"},
    {"stage": "sfm_matching"},
    {"detail": "x" * 161},
])
def test_invalid_dense_observation_rejected(change):
    with pytest.raises(TelemetryValidationError):
        sample(**change)


def test_sparse_run_requires_absent_dense_id_and_matching_stage():
    event = sample(job_kind="offline_sfm", dense_run_id=None,
                   producer="sfm_worker", stage="sfm_mapping", counter=None)
    assert event.dense_run_id is None
    with pytest.raises(TelemetryValidationError):
        replace(event, dense_run_id=RUN)
    with pytest.raises(TelemetryValidationError):
        replace(event, stage="dense_fusion")


@pytest.mark.parametrize("counter", [
    StageWorkCounter("fusion_images_started", 0, 87),
    StageWorkCounter("fusion_images_started", 1, None),
])
def test_counters_do_not_imply_a_total_job_fraction(counter):
    obj = sample(counter=counter)
    assert "progress_fraction" not in obj.as_dict()
    assert "eta_seconds" not in obj.as_dict()


@pytest.mark.parametrize("args", [
    ("fusion_images_started", -1, 87),
    ("fusion_images_started", 88, 87),
    ("fusion_images_started", 1, 0),
    ("fusion_images_started", True, 87),
    ("unknown_units", 1, 87),
])
def test_bad_work_counter_rejected(args):
    with pytest.raises(TelemetryValidationError):
        StageWorkCounter(*args)


def test_counter_stage_mismatch_rejected():
    with pytest.raises(TelemetryValidationError):
        sample(stage="dense_patch_match", counter=StageWorkCounter("fusion_images_started", 1, 87))


@pytest.mark.parametrize("name,value,availability,gpu_index", [
    ("gpu_device_vram_used_bytes", 2**30, "observed", None),
    ("gpu_device_vram_used_bytes", 2**30, "observed", True),
    ("process_rss_bytes", 10, "observed", 0),
    ("process_rss_bytes", None, "observed", None),
    ("system_ram_available_bytes", 0, "unavailable", None),
    ("gpu_device_util_percent", 101, "observed", 0),
    ("process_cpu_percent", float("nan"), "observed", None),
    ("unknown_resource", 0, "observed", None),
])
def test_invalid_resource_reading_rejected(name,value,availability,gpu_index):
    with pytest.raises(TelemetryValidationError):
        ResourceReading(name, value, availability, gpu_index)


def test_gpu_device_metrics_never_claim_process_attribution():
    device = ResourceReading("gpu_device_vram_used_bytes", 2**30, "observed", 0)
    process = ResourceReading("gpu_process_vram_used_bytes", 2**29, "observed", 0)
    assert device.scope == "gpu_device"
    assert process.scope == "gpu_process"
    assert ResourceReading("process_peak_rss_bytes", 2048, "observed").scope == "process"
    assert ResourceReading("process_tree_rss_bytes", 2048, "observed").scope == "process_tree"


def test_rejects_control_and_authority_spoofing_fields():
    payload = sample().as_dict()
    for key, val in [
        ("final_dense_result_admitted", True),
        ("metric_scale_claimed", True),
        ("progress_fraction", 1.0),
        ("eta_seconds", 10),
    ]:
        with pytest.raises(TelemetryValidationError):
            JobTelemetryObservation.from_dict({**payload, key: val})


def test_rejects_oversized_malformed_and_duplicate_resource_fields():
    with pytest.raises(TelemetryValidationError):
        sample(resource_readings=tuple(ResourceReading("process_rss_bytes", 0, "observed") for _ in range(13)))
    with pytest.raises(TelemetryValidationError):
        sample(resource_readings=(ResourceReading("process_rss_bytes", 1, "observed"),
                                  ResourceReading("process_rss_bytes", 2, "observed")))
    with pytest.raises(TelemetryValidationError):
        parse_job_telemetry_line(JOB_TELEMETRY_PREFIX + "{" + " " * JOB_TELEMETRY_MAX_LINE_BYTES)
    with pytest.raises(TelemetryValidationError):
        parse_job_telemetry_line(JOB_TELEMETRY_PREFIX + "not-json")


def test_stream_gate_rejects_foreign_duplicate_and_out_of_order_observations():
    gate = JobTelemetryStreamGate("post_sparse_dense", SID, RUN)
    assert gate.offer(sample(sequence=1)) is True
    assert gate.offer(sample(sequence=1)) is False
    assert gate.offer(sample(sequence=3)) is True  # gaps are legal: latest-only transport
    assert gate.offer(sample(sequence=2)) is False  # out of order
    assert gate.offer(sample(sequence=4, source_session_id="22222222-3333-4444-8555-666666666666")) is False
    assert gate.offer(sample(sequence=4, dense_run_id="99999999-aaaa-4bbb-8ccc-dddddddddddd")) is False
    assert gate.offer(sample(sequence=4, producer="resource_sampler")) is True
    assert gate.offer(sample(sequence=4)) is True  # independent producer sequences
    gate.close()
    assert gate.offer(sample(sequence=5)) is False
    assert gate.rejected_count == 5
    assert gate.accepted_count == 4


def test_reset_changes_binding_and_clears_old_producer_sequence():
    gate = JobTelemetryStreamGate("post_sparse_dense", SID, RUN)
    assert gate.offer(sample(sequence=99))
    new_run = "99999999-aaaa-4bbb-8ccc-dddddddddddd"
    gate.reset("post_sparse_dense", SID, new_run)
    assert gate.offer(sample(sequence=100)) is False
    assert gate.offer(sample(sequence=1, dense_run_id=new_run)) is True


def test_resource_unavailable_does_not_serialize_as_zero():
    payload = sample(resource_readings=(ResourceReading("system_ram_available_bytes", None, "unavailable"),)).as_dict()
    item = payload["resource_readings"][0]
    assert item["value"] is None
    assert item["availability"] == "unavailable"
    assert item["scope"] == "system"


def test_malformed_nested_objects_fail_closed_as_telemetry_validation_errors():
    payload = sample().as_dict()
    for edit in (
        {"stage": []},
        {"producer": []},
        {"job_kind": []},
        {"counter": {"unit": [], "observed_count": 1, "expected_count": 87}},
        {"resource_readings": [{"name": [], "value": 0, "availability": "observed", "gpu_index": 0,
                                 "scope": "gpu_device", "unit": "bytes"}]},
    ):
        with pytest.raises(TelemetryValidationError):
            JobTelemetryObservation.from_dict({**payload, **edit})


def test_rejects_wrong_scope_and_noninteger_byte_value():
    with pytest.raises(TelemetryValidationError):
        ResourceReading("system_ram_available_bytes", 1.5, "observed")
    payload = sample().as_dict()
    payload["resource_readings"][0]["scope"] = "gpu_process"
    with pytest.raises(TelemetryValidationError):
        JobTelemetryObservation.from_dict(payload)


def test_producer_kind_mismatch_rejected():
    with pytest.raises(TelemetryValidationError):
        sample(producer="sfm_worker")
    with pytest.raises(TelemetryValidationError):
        sample(job_kind="offline_sfm", dense_run_id=None, producer="mvs_worker",
               stage="sfm_mapping", counter=None)


def test_huge_numeric_reading_is_validation_failure_not_a_python_overflow():
    with pytest.raises(TelemetryValidationError):
        ResourceReading("process_rss_bytes", 10**500, "observed")
