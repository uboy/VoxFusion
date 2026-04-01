"""Speaker diarization: channel-based, ML-based (pyannote), hybrid, and none."""

from voxfusion.diarization.alignment import SpeakerTurn, align_segments
from voxfusion.diarization.base import DiarizationEngine
from voxfusion.diarization.channel import ChannelDiarizer
from voxfusion.diarization.factory import DiarizerSelection, create_diarizer
from voxfusion.diarization.hybrid import HybridDiarizer
from voxfusion.diarization.none import NoneDiarizer
from voxfusion.diarization.types import DiarizationTurnResult

__all__ = [
    "ChannelDiarizer",
    "create_diarizer",
    "DiarizationTurnResult",
    "DiarizerSelection",
    "DiarizationEngine",
    "HybridDiarizer",
    "NoneDiarizer",
    "SpeakerTurn",
    "align_segments",
]
