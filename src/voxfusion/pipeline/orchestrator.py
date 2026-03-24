"""Central PipelineOrchestrator that wires all components together.

Creates the appropriate ASR engine, diarizer, preprocessor, and output
formatter based on the active ``PipelineConfig``, then delegates to
``BatchPipeline`` (or, in the future, ``StreamingPipeline``).
"""

from pathlib import Path

from voxfusion.asr.factory import create_asr_engine
from voxfusion.config.models import PipelineConfig
from voxfusion.diarization.factory import create_diarizer
from voxfusion.logging import get_logger
from voxfusion.models.result import TranscriptionResult
from voxfusion.output import get_formatter
from voxfusion.pipeline.batch import BatchPipeline, EventCallback
from voxfusion.pipeline.events import PipelineEvent
from voxfusion.preprocessing.normalize import Normalizer
from voxfusion.preprocessing.pipeline import PreProcessingPipeline
from voxfusion.preprocessing.resample import Resampler

log = get_logger(__name__)


class PipelineOrchestrator:
    """Builds and runs the processing pipeline from configuration.

    Example::

        config = load_config()
        orchestrator = PipelineOrchestrator(config)
        result = await orchestrator.transcribe_file(Path("recording.wav"))
        print(orchestrator.format_result(result))
    """

    def __init__(
        self,
        config: PipelineConfig,
        on_event: EventCallback | None = None,
    ) -> None:
        self._config = config
        self._on_event = on_event

        # Build components
        self._asr, self._asr_backend = create_asr_engine(config.asr)
        self._diarizer_selection = create_diarizer(config.diarization, mode="file")
        self._preprocessor = self._build_preprocessor()
        log.info(
            "orchestrator.components_ready",
            asr_backend=self._asr_backend,
            asr_model=self._asr.model_name,
            diarization_requested=self._diarizer_selection.requested_strategy,
            diarization_resolved=self._diarizer_selection.resolved_strategy,
            startup_warnings=list(self._diarizer_selection.warnings),
        )

    def _build_preprocessor(self) -> PreProcessingPipeline:
        """Assemble the preprocessing chain based on config."""
        pipeline = PreProcessingPipeline()
        # Always resample to 16kHz (required by Whisper)
        pipeline.add(Resampler(target_sample_rate=16_000))
        # Always normalize
        pipeline.add(Normalizer())
        return pipeline

    async def transcribe_file(self, file_path: Path) -> TranscriptionResult:
        """Transcribe a single audio file end-to-end.

        Args:
            file_path: Path to the audio file.

        Returns:
            A ``TranscriptionResult`` with all segments.
        """
        log.info("orchestrator.transcribe_file", file=str(file_path))

        batch = BatchPipeline(
            asr_engine=self._asr,
            diarizer=self._diarizer_selection.engine,
            preprocessor=self._preprocessor,
            config=self._config,
            on_event=self._on_event,
            requested_diarization_strategy=self._diarizer_selection.requested_strategy,
            resolved_diarization_strategy=self._diarizer_selection.resolved_strategy,
            startup_warnings=self._diarizer_selection.warnings,
        )
        return await batch.process_file(file_path)

    def format_result(self, result: TranscriptionResult, fmt: str | None = None) -> str:
        """Format a transcription result using the configured output formatter.

        Args:
            result: The transcription result to format.
            fmt: Override output format (e.g. ``"json"``, ``"srt"``).
                 Defaults to the config value.

        Returns:
            The formatted string.
        """
        format_name = fmt or self._config.output.format
        formatter = get_formatter(format_name)
        return formatter.format(result)

    def write_result(
        self,
        result: TranscriptionResult,
        output_path: Path,
        fmt: str | None = None,
    ) -> None:
        """Format and write a transcription result to a file.

        Args:
            result: The transcription result.
            output_path: Destination file path.
            fmt: Override output format.
        """
        format_name = fmt or self._config.output.format
        formatter = get_formatter(format_name)
        formatter.write(result, output_path)
        log.info("orchestrator.result_written", path=str(output_path), format=format_name)

    def close(self) -> None:
        """Release orchestrator-owned resources."""
        self._asr.unload_model()
        self._asr.close()
