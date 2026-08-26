"""Validated offline frame ingestion and initial keyframe selection."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from time import monotonic_ns
from typing import Any

import numpy as np

from core.contracts import ImageSize
from core.contracts.validation import require_nonempty
from core.reconstruction.cache import (
    BoundedFrameCache,
    FrameCacheStats,
    IngestionCachePolicy,
)
from core.reconstruction.contracts import (
    FrameInvalidReason,
    FrameValidity,
    InputSourceMetadata,
    ReconstructionFrame,
    ReconstructionKeyframe,
    TimestampOrigin,
)
from core.reconstruction.control import ReconstructionOperationControl
from core.reconstruction.errors import (
    ReconstructionCancelledError,
    ReconstructionValidationError,
)
from core.reconstruction.events import (
    EventSeverity,
    ReconstructionEventRecorder,
    ReconstructionEventType,
    ReconstructionStage,
)
from core.reconstruction.keyframes import KeyframeSelectionPolicy, KeyframeSelector
from core.reconstruction.quality import (
    FrameQualityMetrics,
    FrameQualityPolicy,
    analyze_frame_quality,
)
from core.reconstruction.sampling import (
    FrameSampler,
    FrameSamplingPlan,
    FrameSamplingPolicy,
    build_frame_sampling_plan,
)
from core.reconstruction.session import ReconstructionSession
from core.reconstruction.sources import (
    DecodedFrameCandidate,
    ReconstructionFrameSource,
    SourceReadIssue,
)
from core.reconstruction.validation import require_uuid

INGESTION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FrameIngestionRecord:
    frame: ReconstructionFrame
    source_locator: str
    source_timestamp_ns: int
    timestamp_origin: TimestampOrigin
    quality: FrameQualityMetrics | None
    sampled: bool
    keyframe_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.frame, ReconstructionFrame):
            raise TypeError("frame must be a ReconstructionFrame.")
        if not isinstance(self.timestamp_origin, TimestampOrigin):
            raise TypeError("timestamp_origin must be a TimestampOrigin.")
        if isinstance(self.source_timestamp_ns, bool) or int(self.source_timestamp_ns) != self.source_timestamp_ns:
            raise ReconstructionValidationError("source_timestamp_ns must be an integer.")
        if not isinstance(self.sampled, bool):
            raise TypeError("sampled must be a bool.")
        if self.quality is not None and not isinstance(self.quality, FrameQualityMetrics):
            raise TypeError("quality must be FrameQualityMetrics or None.")
        object.__setattr__(
            self,
            "source_locator",
            require_nonempty(self.source_locator, field_name="source_locator"),
        )
        object.__setattr__(self, "source_timestamp_ns", int(self.source_timestamp_ns))
        object.__setattr__(self, "keyframe_id", str(self.keyframe_id).strip())
        if self.keyframe_id and not self.sampled:
            raise ReconstructionValidationError("Keyframes must originate from sampled frames.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame.as_dict(),
            "source_locator": self.source_locator,
            "source_timestamp_ns": self.source_timestamp_ns,
            "timestamp_origin": self.timestamp_origin.value,
            "quality": self.quality.as_dict() if self.quality else None,
            "sampled": self.sampled,
            "keyframe_id": self.keyframe_id,
        }


@dataclass(frozen=True, slots=True)
class CachedKeyframeImage:
    frame_id: str
    image: np.ndarray

    def __post_init__(self) -> None:
        if not str(self.frame_id).strip():
            raise ReconstructionValidationError("frame_id must not be empty.")
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a NumPy array.")
        candidate = np.ascontiguousarray(self.image).copy()
        candidate.setflags(write=False)
        object.__setattr__(self, "frame_id", str(self.frame_id).strip())
        object.__setattr__(self, "image", candidate)


@dataclass(frozen=True, slots=True)
class ReconstructionIngestionResult:
    source: InputSourceMetadata
    records: tuple[FrameIngestionRecord, ...]
    read_issues: tuple[SourceReadIssue, ...]
    keyframes: tuple[ReconstructionKeyframe, ...]
    keyframe_images: tuple[CachedKeyframeImage, ...]
    cache_policy: IngestionCachePolicy
    cache_stats: FrameCacheStats
    sampling_plan: FrameSamplingPlan
    elapsed_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.source, InputSourceMetadata):
            raise TypeError("source must be InputSourceMetadata.")
        records = tuple(self.records)
        issues = tuple(self.read_issues)
        keyframes = tuple(self.keyframes)
        images = tuple(self.keyframe_images)
        if any(not isinstance(item, FrameIngestionRecord) for item in records):
            raise TypeError("records must contain FrameIngestionRecord values.")
        if any(not isinstance(item, SourceReadIssue) for item in issues):
            raise TypeError("read_issues must contain SourceReadIssue values.")
        if any(not isinstance(item, ReconstructionKeyframe) for item in keyframes):
            raise TypeError("keyframes must contain ReconstructionKeyframe values.")
        if any(not isinstance(item, CachedKeyframeImage) for item in images):
            raise TypeError("keyframe_images must contain CachedKeyframeImage values.")
        if not isinstance(self.cache_policy, IngestionCachePolicy):
            raise TypeError("cache_policy must be an IngestionCachePolicy.")
        if not isinstance(self.cache_stats, FrameCacheStats):
            raise TypeError("cache_stats must be FrameCacheStats.")
        if not isinstance(self.sampling_plan, FrameSamplingPlan):
            raise TypeError("sampling_plan must be a FrameSamplingPlan.")
        if isinstance(self.elapsed_ns, bool) or int(self.elapsed_ns) < 0:
            raise ReconstructionValidationError("elapsed_ns must be non-negative.")
        frame_ids = [record.frame.frame_id for record in records]
        if len(frame_ids) != len(set(frame_ids)):
            raise ReconstructionValidationError("Ingestion result frame IDs must be unique.")
        keyframe_frame_ids = {item.frame_id for item in keyframes}
        if any(image.frame_id not in keyframe_frame_ids for image in images):
            raise ReconstructionValidationError(
                "Cached keyframe images must reference selected keyframes."
            )
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "read_issues", issues)
        object.__setattr__(self, "keyframes", keyframes)
        object.__setattr__(self, "keyframe_images", images)
        object.__setattr__(self, "elapsed_ns", int(self.elapsed_ns))

    @property
    def decoded_frames(self) -> int:
        return len(self.records)

    @property
    def valid_frames(self) -> int:
        return sum(record.frame.is_valid for record in self.records)

    @property
    def rejected_frames(self) -> int:
        return sum(not record.frame.is_valid for record in self.records) + len(self.read_issues)

    @property
    def sampled_frames(self) -> int:
        return sum(record.sampled for record in self.records)

    @property
    def timestamp_origin_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            key = record.timestamp_origin.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": INGESTION_SCHEMA_VERSION,
            "source": self.source.as_dict(),
            "summary": {
                "decoded_frames": self.decoded_frames,
                "valid_frames": self.valid_frames,
                "rejected_frames": self.rejected_frames,
                "sampled_frames": self.sampled_frames,
                "keyframes": len(self.keyframes),
                "timestamp_origin_counts": self.timestamp_origin_counts,
                "elapsed_ns": self.elapsed_ns,
            },
            "records": [record.as_dict() for record in self.records],
            "read_issues": [issue.as_dict() for issue in self.read_issues],
            "keyframes": [item.as_dict() for item in self.keyframes],
            "cache_policy": self.cache_policy.as_dict(),
            "cache_stats": self.cache_stats.as_dict(),
            "sampling_plan": self.sampling_plan.as_dict(),
        }

    def apply_to_session(self, session: ReconstructionSession) -> None:
        """Append ordered frame/keyframe contracts to a matching empty session."""

        if not isinstance(session, ReconstructionSession):
            raise TypeError("session must be a ReconstructionSession.")
        snapshot = session.snapshot()
        if snapshot.frames or snapshot.keyframes:
            raise ReconstructionValidationError(
                "Ingestion results can be applied only to an empty reconstruction session."
            )
        if self.records and session.session_id != self.records[0].frame.session_id:
            raise ReconstructionValidationError("Ingestion session ID does not match the session.")
        if snapshot.source.source_id != self.source.source_id:
            raise ReconstructionValidationError("Ingestion source ID does not match the session.")
        for record in self.records:
            session.add_frame(record.frame)
        for keyframe in self.keyframes:
            session.add_keyframe(keyframe)


class _FrameValidator:
    def __init__(
        self,
        *,
        session_id: str,
        source: InputSourceMetadata,
        quality_policy: FrameQualityPolicy,
        duplicate_history: int,
    ) -> None:
        self.session_id = require_uuid(session_id, field_name="session_id")
        self.source = source
        self.quality_policy = quality_policy
        if isinstance(duplicate_history, bool) or int(duplicate_history) <= 0:
            raise ReconstructionValidationError("duplicate_history must be positive.")
        self.duplicate_history = int(duplicate_history)
        self.expected_size = source.expected_image_size
        self.last_sequence_index = -1
        self.last_canonical_timestamp_ns: int | None = None
        rate = source.expected_frame_rate_hz or 30.0
        self.nominal_interval_ns = max(1, int(round(1_000_000_000 / rate)))
        self.recent_hashes: OrderedDict[str, None] = OrderedDict()

    def validate(
        self, candidate: DecodedFrameCandidate
    ) -> tuple[ReconstructionFrame, FrameQualityMetrics | None]:
        if candidate.sequence_index <= self.last_sequence_index:
            raise ReconstructionValidationError(
                "Frame source sequence indices must be strictly increasing."
            )
        self.last_sequence_index = candidate.sequence_index
        reasons: list[FrameInvalidReason] = []

        raw_timestamp = candidate.timestamp_ns
        previous = self.last_canonical_timestamp_ns
        if raw_timestamp < 0 or (previous is not None and raw_timestamp <= previous):
            reasons.append(FrameInvalidReason.TIMESTAMP_INVALID)
            canonical_timestamp = 0 if previous is None else previous + self.nominal_interval_ns
        else:
            canonical_timestamp = raw_timestamp
        self.last_canonical_timestamp_ns = canonical_timestamp

        image = candidate.image
        supported = (
            image.dtype == np.uint8
            and image.ndim in {2, 3}
            and (image.ndim == 2 or image.shape[2] == 3)
            and image.shape[0] > 0
            and image.shape[1] > 0
        )
        if supported:
            image_size = ImageSize(width=int(image.shape[1]), height=int(image.shape[0]))
        else:
            reasons.append(FrameInvalidReason.UNSUPPORTED_FORMAT)
            if image.ndim >= 2 and image.shape[0] > 0 and image.shape[1] > 0:
                image_size = ImageSize(width=int(image.shape[1]), height=int(image.shape[0]))
            elif self.expected_size is not None:
                image_size = self.expected_size
            else:
                image_size = ImageSize(1, 1)

        if self.expected_size is None and supported:
            self.expected_size = image_size
        elif self.expected_size is not None and image_size != self.expected_size:
            reasons.append(FrameInvalidReason.DIMENSION_MISMATCH)

        quality: FrameQualityMetrics | None = None
        content_hash = ""
        if supported:
            content_hash = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
            if content_hash in self.recent_hashes:
                reasons.append(FrameInvalidReason.DUPLICATE)
            self.recent_hashes[content_hash] = None
            self.recent_hashes.move_to_end(content_hash)
            while len(self.recent_hashes) > self.duplicate_history:
                self.recent_hashes.popitem(last=False)
            quality = analyze_frame_quality(image, self.quality_policy)
            if not quality.accepted:
                reasons.append(FrameInvalidReason.INSUFFICIENT_QUALITY)

        unique_reasons = tuple(dict.fromkeys(reasons))
        validity = FrameValidity.VALID if not unique_reasons else FrameValidity.INVALID
        frame = ReconstructionFrame(
            frame_id=f"{self.source.source_id}:frame:{candidate.sequence_index:08d}",
            session_id=self.session_id,
            source_id=self.source.source_id,
            sequence_index=candidate.sequence_index,
            timestamp_ns=canonical_timestamp,
            image_size=image_size,
            validity=validity,
            invalid_reasons=unique_reasons,
            content_sha256=content_hash,
        )
        return frame, quality


class ReconstructionIngestionPipeline:
    """One-pass ingestion pipeline isolated from the stable capture workers."""

    def __init__(
        self,
        *,
        session_id: str,
        source: ReconstructionFrameSource,
        sampling_policy: FrameSamplingPolicy,
        keyframe_policy: KeyframeSelectionPolicy,
        quality_policy: FrameQualityPolicy,
        cache_policy: IngestionCachePolicy,
        duplicate_history: int = 2048,
        progress_event_interval: int = 25,
    ) -> None:
        self.session_id = require_uuid(session_id, field_name="session_id")
        if not isinstance(source, ReconstructionFrameSource):
            raise TypeError("source must implement ReconstructionFrameSource.")
        self.source = source
        self.sampling_policy = sampling_policy
        self.keyframe_policy = keyframe_policy
        self.quality_policy = quality_policy
        self.cache_policy = cache_policy
        self.duplicate_history = duplicate_history
        if isinstance(progress_event_interval, bool) or int(progress_event_interval) <= 0:
            raise ReconstructionValidationError("progress_event_interval must be positive.")
        self.progress_event_interval = int(progress_event_interval)

    def run(
        self,
        *,
        control: ReconstructionOperationControl | None = None,
        events: ReconstructionEventRecorder | None = None,
    ) -> ReconstructionIngestionResult:
        active_control = control or ReconstructionOperationControl()
        if events is not None and not isinstance(events, ReconstructionEventRecorder):
            raise TypeError("events must be ReconstructionEventRecorder or None.")
        start_ns = monotonic_ns()
        validator = _FrameValidator(
            session_id=self.session_id,
            source=self.source.metadata,
            quality_policy=self.quality_policy,
            duplicate_history=self.duplicate_history,
        )
        sampling_plan = build_frame_sampling_plan(
            self.sampling_policy,
            estimated_total_frames=self.source.estimated_total_frames,
            expected_frame_rate_hz=self.source.metadata.expected_frame_rate_hz,
        )
        sampler = FrameSampler(self.sampling_policy, plan=sampling_plan)
        selector = KeyframeSelector(self.keyframe_policy)
        cache = BoundedFrameCache(self.cache_policy)
        records: list[FrameIngestionRecord] = []
        total = self.source.estimated_total_frames

        if events is not None:
            events.emit(
                event_type=ReconstructionEventType.STATUS,
                stage=ReconstructionStage.INGESTION,
                message="Reconstruction input ingestion started.",
                elapsed_ns=0,
                metadata={"source_id": self.source.metadata.source_id},
            )

        try:
            for candidate in self.source:
                active_control.checkpoint()
                frame, quality = validator.validate(candidate)
                sampled = sampler.should_select(frame)
                keyframe: ReconstructionKeyframe | None = None
                if sampled and quality is not None:
                    keyframe = selector.consider(frame, candidate.image, quality)
                    if keyframe is not None:
                        cache.put(frame.frame_id, candidate.image)
                records.append(
                    FrameIngestionRecord(
                        frame=frame,
                        source_locator=candidate.source_locator,
                        source_timestamp_ns=candidate.timestamp_ns,
                        timestamp_origin=candidate.timestamp_origin,
                        quality=quality,
                        sampled=sampled,
                        keyframe_id=keyframe.keyframe_id if keyframe else "",
                    )
                )
                if events is not None and total:
                    completed = min(len(records), total)
                    should_emit = (
                        completed == 1
                        or completed == total
                        or completed % self.progress_event_interval == 0
                    )
                    if should_emit:
                        events.emit(
                            event_type=ReconstructionEventType.PROGRESS,
                            stage=ReconstructionStage.INGESTION,
                            message=f"Validated frame {completed} of {total}.",
                            elapsed_ns=monotonic_ns() - start_ns,
                            progress_fraction=completed / total,
                            completed_work=completed,
                            total_work=total,
                        )
        except ReconstructionCancelledError:
            if events is not None:
                events.emit(
                    event_type=ReconstructionEventType.CANCELLATION,
                    stage=ReconstructionStage.INGESTION,
                    message="Reconstruction input ingestion cancelled.",
                    elapsed_ns=monotonic_ns() - start_ns,
                    severity=EventSeverity.WARNING,
                )
            raise

        issues = self.source.issues
        if events is not None and total:
            latest_progress = max(
                (
                    event.progress_fraction or 0.0
                    for event in events.events
                    if event.stage is ReconstructionStage.INGESTION
                    and event.event_type is ReconstructionEventType.PROGRESS
                ),
                default=0.0,
            )
            if latest_progress < 1.0:
                events.emit(
                    event_type=ReconstructionEventType.PROGRESS,
                    stage=ReconstructionStage.INGESTION,
                    message=f"Input scan completed for {total} source entries.",
                    elapsed_ns=monotonic_ns() - start_ns,
                    progress_fraction=1.0,
                    completed_work=total,
                    total_work=total,
                )
        if events is not None:
            events.emit(
                event_type=ReconstructionEventType.STATUS,
                stage=ReconstructionStage.KEYFRAME_SELECTION,
                message=(
                    f"Selected {len(selector.keyframes)} keyframes from "
                    f"{sampler.selected_count} sampled frames."
                ),
                elapsed_ns=monotonic_ns() - start_ns,
                metadata={
                    "decoded_frames": len(records),
                    "read_issues": len(issues),
                    "sampled_frames": sampler.selected_count,
                    "keyframes": len(selector.keyframes),
                    "sampling_plan_mode": sampling_plan.mode.value,
                    "full_span_cap_active": sampling_plan.full_span_cap_active,
                },
            )

        keyframe_images = tuple(
            CachedKeyframeImage(frame_id=frame_id, image=image)
            for frame_id, image in cache.items()
            if any(keyframe.frame_id == frame_id for keyframe in selector.keyframes)
        )
        return ReconstructionIngestionResult(
            source=self.source.metadata,
            records=tuple(records),
            read_issues=issues,
            keyframes=selector.keyframes,
            keyframe_images=keyframe_images,
            cache_policy=self.cache_policy,
            cache_stats=cache.stats,
            sampling_plan=sampling_plan,
            elapsed_ns=monotonic_ns() - start_ns,
        )
