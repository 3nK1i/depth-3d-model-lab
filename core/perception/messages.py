"""Typed, pickle-safe requests for future generic perception workers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from core.contracts.common import FrameReference
from core.contracts.spatial import PixelCoordinateSpace, PixelSpaceRole
from core.contracts.validation import ContractValidationError, readonly_array, require_nonempty
from core.perception.metadata import PerceptionTask


class SegmentationPromptKind(str, Enum):
    BOX = "box"
    POINTS = "points"
    INITIAL_MASK = "initial_mask"


@dataclass(frozen=True, slots=True)
class SegmentationPrompt:
    prompt_id: str
    kind: SegmentationPromptKind
    pixel_space: PixelCoordinateSpace
    box_xyxy: tuple[float, float, float, float] | None = None
    positive_points_xy: tuple[tuple[float, float], ...] = ()
    negative_points_xy: tuple[tuple[float, float], ...] = ()
    initial_mask_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_id", require_nonempty(self.prompt_id, field_name="prompt_id"))
        if not isinstance(self.pixel_space, PixelCoordinateSpace):
            raise TypeError("pixel_space must be a PixelCoordinateSpace.")
        if self.kind is SegmentationPromptKind.BOX:
            if self.box_xyxy is None or len(self.box_xyxy) != 4:
                raise ContractValidationError("A box prompt requires box_xyxy.")
            values = tuple(float(value) for value in self.box_xyxy)
            if not np.isfinite(np.asarray(values)).all():
                raise ContractValidationError("box_xyxy must contain finite coordinates.")
            x0, y0, x1, y1 = values
            if x1 <= x0 or y1 <= y0:
                raise ContractValidationError("box_xyxy must have positive area.")
            object.__setattr__(self, "box_xyxy", values)
        elif self.kind is SegmentationPromptKind.POINTS:
            if not self.positive_points_xy and not self.negative_points_xy:
                raise ContractValidationError("A point prompt requires at least one point.")
        elif self.kind is SegmentationPromptKind.INITIAL_MASK:
            object.__setattr__(
                self,
                "initial_mask_id",
                require_nonempty(self.initial_mask_id, field_name="initial_mask_id"),
            )
        for field_name in ("positive_points_xy", "negative_points_xy"):
            points = tuple(tuple(float(value) for value in point) for point in getattr(self, field_name))
            if any(len(point) != 2 for point in points):
                raise ContractValidationError(f"{field_name} must contain x/y pairs.")
            if points and not np.isfinite(np.asarray(points)).all():
                raise ContractValidationError(f"{field_name} must contain finite coordinates.")
            object.__setattr__(self, field_name, points)


@dataclass(frozen=True, slots=True)
class FramePerceptionRequest:
    frame_reference: FrameReference
    rgb: np.ndarray
    rgb_space: PixelCoordinateSpace
    requested_tasks: tuple[PerceptionTask, ...]

    def __post_init__(self) -> None:
        if self.rgb_space.frame_reference != self.frame_reference:
            raise ContractValidationError("rgb_space does not match frame_reference.")
        if self.rgb_space.role is not PixelSpaceRole.RGB:
            raise ContractValidationError("rgb_space must use PixelSpaceRole.RGB.")
        rgb = readonly_array(
            self.rgb,
            field_name="rgb",
            dtype=np.uint8,
            shape=(self.rgb_space.size.height, self.rgb_space.size.width, 3),
        )
        if not self.requested_tasks:
            raise ContractValidationError("requested_tasks must not be empty.")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "requested_tasks", tuple(self.requested_tasks))


@dataclass(frozen=True, slots=True)
class PromptedSegmentationRequest:
    frame_request: FramePerceptionRequest
    prompt: SegmentationPrompt
    parent_detection_id: str = ""
    parent_mask_id: str = ""

    def __post_init__(self) -> None:
        if self.prompt.pixel_space.frame_reference != self.frame_request.frame_reference:
            raise ContractValidationError("Prompt and frame request refer to different frames.")
        if not self.parent_detection_id and not self.parent_mask_id:
            raise ContractValidationError(
                "Prompted segmentation requires a parent detection or mask identifier."
            )
        object.__setattr__(self, "parent_detection_id", str(self.parent_detection_id).strip())
        object.__setattr__(self, "parent_mask_id", str(self.parent_mask_id).strip())


@dataclass(frozen=True, slots=True)
class SequenceResetRequest:
    session_id: int
    reason: str

    def __post_init__(self) -> None:
        if isinstance(self.session_id, bool) or int(self.session_id) < 0:
            raise ContractValidationError("session_id must be a non-negative integer.")
        object.__setattr__(self, "session_id", int(self.session_id))
        object.__setattr__(self, "reason", require_nonempty(self.reason, field_name="reason"))
