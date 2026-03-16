"""Pipeline event types for lifecycle and progress reporting.

Events are emitted by the pipeline orchestrator to notify callers
about progress, stage transitions, and errors.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PipelineStage(StrEnum):
    """Stages in the processing pipeline."""

    CAPTURE = "capture"
    PREPROCESSING = "preprocessing"
    ASR = "asr"
    DIARIZATION = "diarization"
    TRANSLATION = "translation"
    OUTPUT = "output"


class EventType(StrEnum):
    """Types of pipeline events."""

    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_FAILED = "pipeline_failed"
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    PROGRESS = "progress"


@dataclass(frozen=True)
class PipelineEvent:
    """An event emitted during pipeline execution.

    Attributes:
        event_type: The type of event.
        stage: Which pipeline stage emitted the event, if applicable.
        message: Human-readable description.
        progress: Completion fraction (0.0–1.0) for progress events.
        data: Arbitrary extra data (e.g. segment count, error details).
    """

    event_type: EventType
    stage: PipelineStage | None = None
    message: str = ""
    progress: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)
