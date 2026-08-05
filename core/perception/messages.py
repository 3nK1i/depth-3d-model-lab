"""Typed, pickle-safe requests for future generic perception workers."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import numpy as np
from core.contracts.common import FrameReference
from core.contracts.spatial import PixelCoordinateSpace, PixelSpaceRole
from core.contracts.validation import ContractValidationError, JsonValue, normalize_metadata, readonly_array, require_nonempty
from core.perception.metadata import PerceptionTask

class SegmentationPromptKind(str, Enum):
    BOX = 'box'
    POINTS = 'points'
    INITIAL_MASK = 'initial_mask'

@dataclass(frozen=True, slots=True)
class SegmentationPrompt:
    prompt_id: str
    kind: SegmentationPromptKind
    pixel_space: PixelCoordinateSpace
    box_xyxy: tuple[float, float, float, float] | None = None
    positive_points_xy: tuple[tuple[float, float], ...] = ()
    negative_points_xy: tuple[tuple[float, float], ...] = ()
    initial_mask_id: str = ''

    def __post_init__(self) -> None:
        object.__setattr__(self, 'prompt_id', require_nonempty(self.prompt_id, field_name='prompt_id'))
        if not isinstance(self.pixel_space, PixelCoordinateSpace):
            raise TypeError('pixel_space must be a PixelCoordinateSpace.')
        if self.kind is SegmentationPromptKind.BOX:
            if self.box_xyxy is None or len(self.box_xyxy) != 4:
                raise ContractValidationError('A box prompt requires box_xyxy.')
            values = tuple((float(value) for value in self.box_xyxy))
            if not np.isfinite(np.asarray(values)).all():
                raise ContractValidationError('box_xyxy must contain finite coordinates.')
            x0, y0, x1, y1 = values
            if x1 <= x0 or y1 <= y0:
                raise ContractValidationError('box_xyxy must have positive area.')
            object.__setattr__(self, 'box_xyxy', values)
        elif self.kind is SegmentationPromptKind.POINTS:
            if not self.positive_points_xy and (not self.negative_points_xy):
                raise ContractValidationError('A point prompt requires at least one point.')
        elif self.kind is SegmentationPromptKind.INITIAL_MASK:
            object.__setattr__(self, 'initial_mask_id', require_nonempty(self.initial_mask_id, field_name='initial_mask_id'))
        for field_name in ('positive_points_xy', 'negative_points_xy'):
            points = tuple((tuple((float(value) for value in point)) for point in getattr(self, field_name)))
            if any((len(point) != 2 for point in points)):
                raise ContractValidationError(f'{field_name} must contain x/y pairs.')
            if points and (not np.isfinite(np.asarray(points)).all()):
                raise ContractValidationError(f'{field_name} must contain finite coordinates.')
            object.__setattr__(self, field_name, points)

@dataclass(frozen=True, slots=True)
class FramePerceptionRequest:
    frame_reference: FrameReference
    rgb: np.ndarray
    rgb_space: PixelCoordinateSpace
    requested_tasks: tuple[PerceptionTask, ...]

    def __post_init__(self) -> None:
        if self.rgb_space.frame_reference != self.frame_reference:
            raise ContractValidationError('rgb_space does not match frame_reference.')
        if self.rgb_space.role is not PixelSpaceRole.RGB:
            raise ContractValidationError('rgb_space must use PixelSpaceRole.RGB.')
        rgb = readonly_array(self.rgb, field_name='rgb', dtype=np.uint8, shape=(self.rgb_space.size.height, self.rgb_space.size.width, 3))
        if not self.requested_tasks:
            raise ContractValidationError('requested_tasks must not be empty.')
        tasks = tuple(self.requested_tasks)
        if any((not isinstance(task, PerceptionTask) for task in tasks)):
            raise TypeError('requested_tasks must contain PerceptionTask values.')
        if len(tasks) != len(set(tasks)):
            raise ContractValidationError('requested_tasks must be unique.')
        object.__setattr__(self, 'rgb', rgb)
        object.__setattr__(self, 'requested_tasks', tasks)

    def __reduce__(self):
        return (type(self), (self.frame_reference, self.rgb, self.rgb_space, self.requested_tasks))

@dataclass(frozen=True, slots=True)
class PromptedSegmentationRequest:
    frame_request: FramePerceptionRequest
    prompt: SegmentationPrompt
    parent_detection_id: str = ''
    parent_mask_id: str = ''

    def __post_init__(self) -> None:
        if self.prompt.pixel_space.frame_reference != self.frame_request.frame_reference:
            raise ContractValidationError('Prompt and frame request refer to different frames.')
        if not self.parent_detection_id and (not self.parent_mask_id):
            raise ContractValidationError('Prompted segmentation requires a parent detection or mask identifier.')
        object.__setattr__(self, 'parent_detection_id', str(self.parent_detection_id).strip())
        object.__setattr__(self, 'parent_mask_id', str(self.parent_mask_id).strip())

@dataclass(frozen=True, slots=True)
class SequenceResetRequest:
    session_id: int
    reason: str

    def __post_init__(self) -> None:
        if isinstance(self.session_id, bool) or int(self.session_id) < 0:
            raise ContractValidationError('session_id must be a non-negative integer.')
        object.__setattr__(self, 'session_id', int(self.session_id))
        object.__setattr__(self, 'reason', require_nonempty(self.reason, field_name='reason'))

@dataclass(frozen=True, slots=True)
class PerceptionWorkerConfiguration:
    primary_model_id: str
    primary_config: tuple[tuple[str, JsonValue], ...] = ()
    refiner_model_id: str = ''
    refiner_config: tuple[tuple[str, JsonValue], ...] = ()
    enable_prompted_refinement: bool = True
    allow_box_fallback: bool = True
    minimum_mask_pixels: int = 1
    allow_refiner_co_residency: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, 'primary_model_id', require_nonempty(self.primary_model_id, field_name='primary_model_id'))
        object.__setattr__(self, 'refiner_model_id', str(self.refiner_model_id).strip())
        object.__setattr__(self, 'primary_config', normalize_metadata(self.primary_config))
        object.__setattr__(self, 'refiner_config', normalize_metadata(self.refiner_config))
        if self.minimum_mask_pixels < 1:
            raise ContractValidationError('minimum_mask_pixels must be positive.')
        if not self.refiner_model_id and self.refiner_config:
            raise ContractValidationError('refiner_config requires refiner_model_id.')
        if self.refiner_model_id and (not self.enable_prompted_refinement):
            raise ContractValidationError('refiner_model_id requires prompted refinement to be enabled.')
        if self.allow_refiner_co_residency and (not self.refiner_model_id):
            raise ContractValidationError('Refiner co-residency requires refiner_model_id.')

@dataclass(frozen=True, slots=True)
class PerceptionPipelineRequest:
    request_id: int
    frame_request: FramePerceptionRequest
    manual_masks: tuple[tuple[str, tuple[object, ...]], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.request_id, bool) or int(self.request_id) < 0:
            raise ContractValidationError('request_id must be non-negative.')
        object.__setattr__(self, 'request_id', int(self.request_id))
        from core.perception.result import InstanceMask
        normalized: list[tuple[str, tuple[InstanceMask, ...]]] = []
        seen: set[str] = set()
        for raw_key, raw_masks in self.manual_masks:
            key = require_nonempty(raw_key, field_name='manual mask instance key')
            if key in seen:
                raise ContractValidationError(f'Duplicate manual mask instance key: {key!r}.')
            seen.add(key)
            masks = tuple(raw_masks)
            if any((not isinstance(mask, InstanceMask) for mask in masks)):
                raise TypeError('manual_masks must contain InstanceMask values.')
            normalized.append((key, masks))
        object.__setattr__(self, 'manual_masks', tuple(normalized))

@dataclass(frozen=True, slots=True)
class PerceptionShutdownRequest:
    reason: str = 'shutdown'

    def __post_init__(self) -> None:
        object.__setattr__(self, 'reason', require_nonempty(self.reason, field_name='reason'))

@dataclass(frozen=True, slots=True)
class PerceptionWorkerReady:
    primary_model_id: str
    refiner_model_id: str = ''

    def __post_init__(self) -> None:
        object.__setattr__(self, 'primary_model_id', require_nonempty(self.primary_model_id, field_name='primary_model_id'))
        object.__setattr__(self, 'refiner_model_id', str(self.refiner_model_id).strip())

@dataclass(frozen=True, slots=True)
class PerceptionWorkerResponse:
    request_id: int
    result: object | None = None
    error: str = ''
    fatal: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.request_id, bool) or int(self.request_id) < -1:
            raise ContractValidationError('request_id must be -1 or a non-negative integer.')
        object.__setattr__(self, 'request_id', int(self.request_id))
        object.__setattr__(self, 'error', str(self.error).strip())
        if self.fatal and (not self.error):
            raise ContractValidationError('Fatal worker responses require error details.')

@dataclass(frozen=True, slots=True)
class PerceptionWorkerStopped:
    reason: str = 'stopped'
