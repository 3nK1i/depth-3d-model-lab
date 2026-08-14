"""Phase 9.2 semantic-workspace contracts.

This module adds a lightweight review/label layer above the existing perception
contracts.  It deliberately does not replace :class:`ObjectObservation`, mask
authority, YOLO/SAM adapters, measurement authority, or canonical geometry.

Auto-segmentation proposals may begin unlabeled.  A user can assign an existing
YOLO-compatible class or create a project class, give an instance a human-friendly
name, and accept/reject it.  Accepted objects can then be normalized back to the
existing ``ObjectObservation`` structure so downstream consumers remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from enum import Enum
from typing import Iterable, Mapping

from core.contracts.common import FrameReference, ModelProvenance
from core.contracts.validation import ContractValidationError, require_nonempty
from core.perception.result import (
    DenseMaskPayload,
    InstanceMask,
    MaskSource,
    ObjectObservation,
    PerceptionFrameResult,
)


SEMANTIC_WORKSPACE_SCHEMA_VERSION = 1


class SemanticClassOrigin(str, Enum):
    """Where a class definition came from."""

    MODEL = "model"
    CUSTOM = "custom"
    IMPORTED = "imported"


class SemanticLabelSource(str, Enum):
    """Authority/provenance of the current semantic label."""

    MODEL_PREDICTION = "model_prediction"
    USER_ASSIGNED = "user_assigned"
    IMPORTED = "imported"
    UNASSIGNED = "unassigned"


class SemanticReviewStatus(str, Enum):
    """Human review state for an Auto-Segment workspace object."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class SemanticClassDefinition:
    """YOLO-compatible semantic class definition for a workspace/project."""

    class_id: int
    class_name: str
    origin: SemanticClassOrigin
    source_model_id: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.class_id, bool) or int(self.class_id) < 0:
            raise ContractValidationError("class_id must be a non-negative integer.")
        object.__setattr__(self, "class_id", int(self.class_id))
        object.__setattr__(
            self,
            "class_name",
            require_nonempty(self.class_name, field_name="class_name"),
        )
        if not isinstance(self.origin, SemanticClassOrigin):
            raise TypeError("origin must be SemanticClassOrigin.")
        object.__setattr__(self, "source_model_id", str(self.source_model_id).strip())
        object.__setattr__(self, "description", str(self.description).strip())
        if self.origin is SemanticClassOrigin.MODEL and not self.source_model_id:
            raise ContractValidationError("Model classes require source_model_id.")


@dataclass(frozen=True, slots=True)
class SemanticClassCatalog:
    """Immutable class map shared by model and custom Auto-Segment labels."""

    classes: tuple[SemanticClassDefinition, ...] = ()

    def __post_init__(self) -> None:
        ids = [item.class_id for item in self.classes]
        if len(ids) != len(set(ids)):
            raise ContractValidationError("Semantic class IDs must be unique.")
        names = [item.class_name.casefold() for item in self.classes]
        if len(names) != len(set(names)):
            raise ContractValidationError(
                "Semantic class names must be unique case-insensitively."
            )

    def by_id(self, class_id: int) -> SemanticClassDefinition | None:
        value = int(class_id)
        return next((item for item in self.classes if item.class_id == value), None)

    def by_name(self, class_name: str) -> SemanticClassDefinition | None:
        key = str(class_name).strip().casefold()
        return next((item for item in self.classes if item.class_name.casefold() == key), None)

    @property
    def next_class_id(self) -> int:
        return max((item.class_id for item in self.classes), default=-1) + 1

    def with_class(self, definition: SemanticClassDefinition) -> "SemanticClassCatalog":
        existing_id = self.by_id(definition.class_id)
        existing_name = self.by_name(definition.class_name)
        if existing_id is not None and existing_id != definition:
            raise ContractValidationError(
                f"class_id {definition.class_id} is already assigned to {existing_id.class_name!r}."
            )
        if existing_name is not None and existing_name != definition:
            raise ContractValidationError(
                f"class_name {definition.class_name!r} is already assigned to class_id "
                f"{existing_name.class_id}."
            )
        if existing_id is not None:
            return self
        return SemanticClassCatalog(self.classes + (definition,))

    def with_custom_class(
        self,
        class_name: str,
        *,
        description: str = "",
    ) -> tuple["SemanticClassCatalog", SemanticClassDefinition]:
        existing = self.by_name(class_name)
        if existing is not None:
            return self, existing
        definition = SemanticClassDefinition(
            class_id=self.next_class_id,
            class_name=class_name,
            origin=SemanticClassOrigin.CUSTOM,
            description=description,
        )
        return self.with_class(definition), definition

    @classmethod
    def from_model_names(
        cls,
        names: Mapping[int, str] | Iterable[tuple[int, str]],
        *,
        source_model_id: str,
    ) -> "SemanticClassCatalog":
        values = names.items() if isinstance(names, Mapping) else names
        return cls(
            tuple(
                SemanticClassDefinition(
                    class_id=int(class_id),
                    class_name=str(class_name),
                    origin=SemanticClassOrigin.MODEL,
                    source_model_id=source_model_id,
                )
                for class_id, class_name in sorted(values, key=lambda item: int(item[0]))
            )
        )


@dataclass(frozen=True, slots=True)
class SemanticWorkspaceObject:
    """One selectable Auto-Segment/perception object in the review workspace."""

    object_id: str
    observation: ObjectObservation | None = None
    proposal_mask: InstanceMask | None = None
    class_id: int | None = None
    class_name: str = ""
    instance_name: str = ""
    label_source: SemanticLabelSource = SemanticLabelSource.UNASSIGNED
    review_status: SemanticReviewStatus = SemanticReviewStatus.PENDING
    label_confidence: float | None = None
    notes: str = ""
    merged_from_object_ids: tuple[str, ...] = ()
    merged_from_mask_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "object_id", require_nonempty(self.object_id, field_name="object_id")
        )
        if self.observation is None and self.proposal_mask is None:
            raise ContractValidationError(
                "A workspace object requires an ObjectObservation or proposal_mask."
            )
        if self.observation is not None and not isinstance(self.observation, ObjectObservation):
            raise TypeError("observation must be ObjectObservation when provided.")
        if self.proposal_mask is not None and not isinstance(self.proposal_mask, InstanceMask):
            raise TypeError("proposal_mask must be InstanceMask when provided.")
        if self.observation is not None and self.proposal_mask is not None:
            raise ContractValidationError(
                "Use either observation or proposal_mask as the workspace source, not both."
            )
        if not isinstance(self.label_source, SemanticLabelSource):
            raise TypeError("label_source must be SemanticLabelSource.")
        if not isinstance(self.review_status, SemanticReviewStatus):
            raise TypeError("review_status must be SemanticReviewStatus.")
        if self.class_id is None:
            if str(self.class_name).strip():
                raise ContractValidationError(
                    "class_name must be empty while class_id is unassigned."
                )
            if self.label_source is not SemanticLabelSource.UNASSIGNED:
                raise ContractValidationError(
                    "Unassigned objects must use label_source=UNASSIGNED."
                )
        else:
            if isinstance(self.class_id, bool) or int(self.class_id) < 0:
                raise ContractValidationError("class_id must be non-negative when assigned.")
            object.__setattr__(self, "class_id", int(self.class_id))
            object.__setattr__(
                self,
                "class_name",
                require_nonempty(self.class_name, field_name="class_name"),
            )
            if self.label_source is SemanticLabelSource.UNASSIGNED:
                raise ContractValidationError(
                    "Assigned objects cannot use label_source=UNASSIGNED."
                )
        object.__setattr__(self, "instance_name", str(self.instance_name).strip())
        object.__setattr__(self, "notes", str(self.notes).strip())
        merged_objects = tuple(require_nonempty(value, field_name="merged_from_object_id") for value in self.merged_from_object_ids)
        merged_masks = tuple(require_nonempty(value, field_name="merged_from_mask_id") for value in self.merged_from_mask_ids)
        if bool(merged_objects) != bool(merged_masks):
            raise ContractValidationError("Merged-object and merged-mask provenance must be present together.")
        if merged_objects:
            if len(merged_objects) < 2 or len(merged_masks) < 2:
                raise ContractValidationError("A semantic merge requires at least two source objects/masks.")
            if len(merged_objects) != len(merged_masks):
                raise ContractValidationError("Merged-object and merged-mask provenance lengths must match.")
            if len(set(merged_objects)) != len(merged_objects) or len(set(merged_masks)) != len(merged_masks):
                raise ContractValidationError("Semantic merge provenance IDs must be unique.")
            if self.object_id in merged_objects:
                raise ContractValidationError("A merged object cannot name itself as a merge source.")
        object.__setattr__(self, "merged_from_object_ids", merged_objects)
        object.__setattr__(self, "merged_from_mask_ids", merged_masks)
        if self.label_confidence is not None:
            value = float(self.label_confidence)
            if not 0.0 <= value <= 1.0:
                raise ContractValidationError("label_confidence must be in [0, 1].")
            object.__setattr__(self, "label_confidence", value)
        if self.review_status is SemanticReviewStatus.ACCEPTED and self.class_id is None:
            raise ContractValidationError("Accepted objects require an assigned class.")

    @property
    def frame_reference(self) -> FrameReference:
        if self.observation is not None:
            return self.observation.frame_reference
        assert self.proposal_mask is not None
        return self.proposal_mask.frame_reference

    @property
    def selected_mask_id(self) -> str:
        if self.observation is not None:
            return self.observation.selected_mask_id
        assert self.proposal_mask is not None
        return self.proposal_mask.mask_id

    @property
    def is_geometry_authority(self) -> bool:
        """Semantic workspace labels never become canonical geometry authority."""

        return False

    @property
    def is_metric_authority(self) -> bool:
        """Semantic workspace labels never establish metric scale."""

        return False

    @property
    def is_measurement_authority(self) -> bool:
        """A label/mask is context for measurements, not measurement authority itself."""

        return False


@dataclass(frozen=True, slots=True)
class SemanticWorkspace:
    """Immutable per-frame semantic review state for Phase 9.2."""

    workspace_id: str
    frame_reference: FrameReference
    class_catalog: SemanticClassCatalog
    objects: tuple[SemanticWorkspaceObject, ...]
    revision: int = 0
    schema_version: int = SEMANTIC_WORKSPACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "workspace_id",
            require_nonempty(self.workspace_id, field_name="workspace_id"),
        )
        if not isinstance(self.class_catalog, SemanticClassCatalog):
            raise TypeError("class_catalog must be SemanticClassCatalog.")
        if isinstance(self.revision, bool) or int(self.revision) < 0:
            raise ContractValidationError("revision must be a non-negative integer.")
        object.__setattr__(self, "revision", int(self.revision))
        if int(self.schema_version) != SEMANTIC_WORKSPACE_SCHEMA_VERSION:
            raise ContractValidationError(
                f"Unsupported semantic workspace schema version: {self.schema_version}."
            )
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object_ids = [item.object_id for item in self.objects]
        if len(object_ids) != len(set(object_ids)):
            raise ContractValidationError("Semantic workspace object IDs must be unique.")
        instance_names = [
            item.instance_name.casefold() for item in self.objects if item.instance_name
        ]
        if len(instance_names) != len(set(instance_names)):
            raise ContractValidationError(
                "Non-empty instance names must be unique case-insensitively."
            )
        for item in self.objects:
            if item.frame_reference != self.frame_reference:
                raise ContractValidationError(
                    "Every workspace object must refer to the workspace frame."
                )
            if item.class_id is not None:
                definition = self.class_catalog.by_id(item.class_id)
                if definition is None or definition.class_name != item.class_name:
                    raise ContractValidationError(
                        "Assigned object class must match the workspace class catalog."
                    )

    def object_by_id(self, object_id: str) -> SemanticWorkspaceObject | None:
        key = str(object_id).strip()
        return next((item for item in self.objects if item.object_id == key), None)

    @property
    def accepted_objects(self) -> tuple[SemanticWorkspaceObject, ...]:
        return tuple(
            item for item in self.objects if item.review_status is SemanticReviewStatus.ACCEPTED
        )




def workspace_from_mask_proposals(
    masks: Iterable[InstanceMask],
    *,
    workspace_id: str,
    class_catalog: SemanticClassCatalog | None = None,
) -> SemanticWorkspace:
    """Create an unlabeled Auto-Segment workspace without inventing class semantics."""

    proposals = tuple(masks)
    if not proposals:
        raise ContractValidationError("Auto-Segment mask proposals must not be empty.")
    frame_reference = proposals[0].frame_reference
    if any(mask.frame_reference != frame_reference for mask in proposals):
        raise ContractValidationError("Auto-Segment mask proposals must belong to one frame.")
    mask_ids = [mask.mask_id for mask in proposals]
    if len(mask_ids) != len(set(mask_ids)):
        raise ContractValidationError("Auto-Segment proposal mask IDs must be unique.")
    return SemanticWorkspace(
        workspace_id=workspace_id,
        frame_reference=frame_reference,
        class_catalog=class_catalog or SemanticClassCatalog(),
        objects=tuple(
            SemanticWorkspaceObject(
                object_id=f"proposal:{mask.mask_id}",
                proposal_mask=mask,
            )
            for mask in proposals
        ),
    )

def append_workspace_mask_proposal(
    workspace: SemanticWorkspace,
    mask: InstanceMask,
    *,
    object_id: str,
) -> SemanticWorkspace:
    """Append one genuinely unlabeled interactive mask proposal.

    The caller owns the stable object ID.  No class or authority is invented.
    """

    if not isinstance(workspace, SemanticWorkspace):
        raise TypeError("workspace must be SemanticWorkspace.")
    if not isinstance(mask, InstanceMask):
        raise TypeError("mask must be InstanceMask.")
    key = require_nonempty(object_id, field_name="object_id")
    if workspace.object_by_id(key) is not None:
        raise ContractValidationError(f"Semantic workspace object already exists: {key!r}.")
    if mask.frame_reference != workspace.frame_reference:
        raise ContractValidationError("Interactive proposal mask must belong to the workspace frame.")
    return SemanticWorkspace(
        workspace_id=workspace.workspace_id,
        frame_reference=workspace.frame_reference,
        class_catalog=workspace.class_catalog,
        objects=workspace.objects + (SemanticWorkspaceObject(object_id=key, proposal_mask=mask),),
        revision=workspace.revision + 1,
    )


def replace_workspace_object_mask(
    workspace: SemanticWorkspace,
    object_id: str,
    mask: InstanceMask,
) -> SemanticWorkspace:
    """Replace one proposal mask while preserving semantic identity.

    A changed mask always returns to PENDING review.  Existing class/instance
    identity is preserved, but the human must re-accept the revised geometry.
    """

    if not isinstance(workspace, SemanticWorkspace):
        raise TypeError("workspace must be SemanticWorkspace.")
    if not isinstance(mask, InstanceMask):
        raise TypeError("mask must be InstanceMask.")
    key = require_nonempty(object_id, field_name="object_id")
    target = workspace.object_by_id(key)
    if target is None:
        raise ContractValidationError(f"Unknown semantic workspace object: {key!r}.")
    if target.proposal_mask is None:
        raise ContractValidationError(
            "Interactive mask replacement currently requires a proposal-backed workspace object."
        )
    if mask.frame_reference != workspace.frame_reference:
        raise ContractValidationError("Refined mask must belong to the workspace frame.")
    if mask.parent_mask_id != target.proposal_mask.mask_id:
        raise ContractValidationError("Refined mask must name the current proposal mask as its parent.")
    if mask.revision <= target.proposal_mask.revision:
        raise ContractValidationError("Refined mask revision must increase monotonically.")
    replacement = replace(
        target,
        proposal_mask=mask,
        review_status=SemanticReviewStatus.PENDING,
    )
    return SemanticWorkspace(
        workspace_id=workspace.workspace_id,
        frame_reference=workspace.frame_reference,
        class_catalog=workspace.class_catalog,
        objects=tuple(replacement if item.object_id == key else item for item in workspace.objects),
        revision=workspace.revision + 1,
    )



def merge_workspace_objects(
    workspace: SemanticWorkspace,
    object_ids: Iterable[str],
    *,
    primary_object_id: str = "",
) -> tuple[SemanticWorkspace, str]:
    """Union two or more proposal-backed masks into one reviewed semantic object.

    The operation is deterministic and purely semantic: no model is rerun and no
    geometry/metric/measurement authority is created.  The primary object's class
    and instance identity are preserved when assigned, while the merged result
    always returns to ``PENDING`` review.
    """

    if not isinstance(workspace, SemanticWorkspace):
        raise TypeError("workspace must be SemanticWorkspace.")
    requested = tuple(require_nonempty(value, field_name="object_id") for value in object_ids)
    if len(requested) < 2:
        raise ContractValidationError("Merge requires at least two semantic workspace objects.")
    if len(set(requested)) != len(requested):
        raise ContractValidationError("Merge object IDs must be unique.")

    requested_set = set(requested)
    ordered = tuple(item for item in workspace.objects if item.object_id in requested_set)
    if len(ordered) != len(requested):
        missing = sorted(requested_set - {item.object_id for item in ordered})
        raise ContractValidationError(f"Unknown semantic workspace merge object(s): {missing!r}.")
    if any(item.proposal_mask is None for item in ordered):
        raise ContractValidationError("Semantic merge currently requires proposal-backed objects.")

    primary_key = str(primary_object_id).strip() or ordered[0].object_id
    primary = next((item for item in ordered if item.object_id == primary_key), None)
    if primary is None:
        raise ContractValidationError("primary_object_id must be one of the merged objects.")

    masks = tuple(item.proposal_mask for item in ordered)
    assert all(mask is not None for mask in masks)
    typed_masks = tuple(mask for mask in masks if mask is not None)
    first = typed_masks[0]
    first_shape = tuple(first.payload.data.shape) if isinstance(first.payload, DenseMaskPayload) else None
    if first_shape is None:
        raise ContractValidationError("Semantic merge requires dense boolean masks.")
    for mask in typed_masks[1:]:
        if mask.frame_reference != workspace.frame_reference:
            raise ContractValidationError("All merged masks must belong to the workspace frame.")
        if mask.pixel_space.space_id != first.pixel_space.space_id:
            raise ContractValidationError("All merged masks must use the same pixel coordinate space.")
        if not isinstance(mask.payload, DenseMaskPayload) or tuple(mask.payload.data.shape) != first_shape:
            raise ContractValidationError("All merged masks must use equal-size dense boolean payloads.")

    union = typed_masks[0].payload.data.copy()
    for mask in typed_masks[1:]:
        union = union | mask.payload.data
    if not bool(union.any()):
        raise ContractValidationError("Merged semantic mask must contain foreground pixels.")

    source_object_ids: list[str] = []
    source_mask_ids: list[str] = []
    for item in ordered:
        assert item.proposal_mask is not None
        object_sources = item.merged_from_object_ids or (item.object_id,)
        mask_sources = item.merged_from_mask_ids or (item.proposal_mask.mask_id,)
        for value in object_sources:
            if value not in source_object_ids:
                source_object_ids.append(value)
        for value in mask_sources:
            if value not in source_mask_ids:
                source_mask_ids.append(value)

    digest_seed = "\x1f".join(source_mask_ids).encode("utf-8")
    digest = sha256(digest_seed).hexdigest()[:16]
    merged_object_id = f"merge:{digest}"
    if workspace.object_by_id(merged_object_id) is not None and merged_object_id not in requested_set:
        raise ContractValidationError(f"Deterministic merge object ID already exists: {merged_object_id!r}.")
    mask_id = f"merge-mask:{digest}:r{workspace.revision + 1}"
    provenance = ModelProvenance(
        model_id="semantic-mask-merge",
        display_name="Semantic Mask Merge",
        task="semantic_merge",
        metadata=(
            ("source_object_ids", ",".join(source_object_ids)),
            ("source_mask_ids", ",".join(source_mask_ids)),
        ),
    )
    merged_mask = InstanceMask(
        mask_id=mask_id,
        frame_reference=workspace.frame_reference,
        pixel_space=first.pixel_space,
        payload=DenseMaskPayload(union),
        source=MaskSource.SEMANTIC_MERGE,
        model_provenance=provenance,
        revision=max(mask.revision for mask in typed_masks) + 1,
        foreground_valid=True,
    )

    if primary.class_id is None:
        class_id = None
        class_name = ""
        instance_name = ""
        label_source = SemanticLabelSource.UNASSIGNED
    else:
        class_id = primary.class_id
        class_name = primary.class_name
        instance_name = primary.instance_name
        label_source = primary.label_source

    merged = SemanticWorkspaceObject(
        object_id=merged_object_id,
        proposal_mask=merged_mask,
        class_id=class_id,
        class_name=class_name,
        instance_name=instance_name,
        label_source=label_source,
        review_status=SemanticReviewStatus.PENDING,
        label_confidence=None,
        notes=f"Merged {len(ordered)} semantic masks.",
        merged_from_object_ids=tuple(source_object_ids),
        merged_from_mask_ids=tuple(source_mask_ids),
    )

    first_index = min(index for index, item in enumerate(workspace.objects) if item.object_id in requested_set)
    remaining = [item for item in workspace.objects if item.object_id not in requested_set]
    remaining.insert(first_index, merged)
    return (
        SemanticWorkspace(
            workspace_id=workspace.workspace_id,
            frame_reference=workspace.frame_reference,
            class_catalog=workspace.class_catalog,
            objects=tuple(remaining),
            revision=workspace.revision + 1,
        ),
        merged_object_id,
    )

def workspace_from_perception_result(
    result: PerceptionFrameResult,
    *,
    workspace_id: str,
) -> SemanticWorkspace:
    """Create a review workspace while preserving existing YOLO-style semantics."""

    classes: dict[int, str] = {}
    for observation in result.observations:
        previous = classes.get(observation.class_id)
        if previous is not None and previous != observation.class_name:
            raise ContractValidationError(
                f"Perception result maps class_id {observation.class_id} to multiple names."
            )
        classes[observation.class_id] = observation.class_name
    model_id = result.model_provenance[0].model_id if result.model_provenance else "perception"
    catalog = SemanticClassCatalog.from_model_names(classes, source_model_id=model_id)
    objects = tuple(
        SemanticWorkspaceObject(
            object_id=observation.observation_id,
            observation=observation,
            class_id=observation.class_id,
            class_name=observation.class_name,
            instance_name="",
            label_source=SemanticLabelSource.MODEL_PREDICTION,
            review_status=SemanticReviewStatus.PENDING,
            label_confidence=(
                observation.detection.confidence if observation.detection is not None else None
            ),
        )
        for observation in result.observations
    )
    return SemanticWorkspace(
        workspace_id=workspace_id,
        frame_reference=result.frame_reference,
        class_catalog=catalog,
        objects=objects,
    )



def assign_workspace_class(
    workspace: SemanticWorkspace,
    object_id: str,
    definition: SemanticClassDefinition,
    *,
    instance_name: str = "",
    source: SemanticLabelSource = SemanticLabelSource.USER_ASSIGNED,
) -> SemanticWorkspace:
    """Assign/reassign one object without mutating the original observation/mask."""

    if source is SemanticLabelSource.UNASSIGNED:
        raise ContractValidationError("Assigned classes require an explicit label source.")
    target = workspace.object_by_id(object_id)
    if target is None:
        raise ContractValidationError(f"Unknown semantic workspace object: {object_id!r}.")
    catalog = workspace.class_catalog.with_class(definition)
    replacement = replace(
        target,
        class_id=definition.class_id,
        class_name=definition.class_name,
        instance_name=str(instance_name).strip(),
        label_source=source,
        label_confidence=None if source is SemanticLabelSource.USER_ASSIGNED else target.label_confidence,
    )
    objects = tuple(replacement if item.object_id == target.object_id else item for item in workspace.objects)
    return SemanticWorkspace(
        workspace_id=workspace.workspace_id,
        frame_reference=workspace.frame_reference,
        class_catalog=catalog,
        objects=objects,
        revision=workspace.revision + 1,
    )



def set_workspace_review_status(
    workspace: SemanticWorkspace,
    object_id: str,
    status: SemanticReviewStatus,
) -> SemanticWorkspace:
    """Accept/reject/pending one object as an immutable workspace revision."""

    target = workspace.object_by_id(object_id)
    if target is None:
        raise ContractValidationError(f"Unknown semantic workspace object: {object_id!r}.")
    replacement = replace(target, review_status=status)
    objects = tuple(replacement if item.object_id == target.object_id else item for item in workspace.objects)
    return SemanticWorkspace(
        workspace_id=workspace.workspace_id,
        frame_reference=workspace.frame_reference,
        class_catalog=workspace.class_catalog,
        objects=objects,
        revision=workspace.revision + 1,
    )



def accepted_observations(workspace: SemanticWorkspace) -> tuple[ObjectObservation, ...]:
    """Return existing downstream contracts with reviewed semantic labels applied.

    Masks, detections, refinements, track IDs, border contacts, and warning provenance
    are preserved.  Only the observation-level class ID/name are changed when the
    user reclassifies an object.
    """

    output: list[ObjectObservation] = []
    for item in workspace.accepted_objects:
        if item.class_id is None:
            raise ContractValidationError("Accepted workspace object is unexpectedly unlabeled.")
        if item.observation is not None:
            output.append(
                replace(
                    item.observation,
                    class_id=item.class_id,
                    class_name=item.class_name,
                )
            )
        else:
            assert item.proposal_mask is not None
            output.append(
                ObjectObservation(
                    observation_id=item.object_id,
                    frame_reference=item.proposal_mask.frame_reference,
                    class_id=item.class_id,
                    class_name=item.class_name,
                    masks=(item.proposal_mask,),
                )
            )
    return tuple(output)
