"""Audio Preprocessing Pipeline.

Normaliza audio de qualquer fonte antes do VAD e da engine.
Pipeline: Ingestao -> [Resample] -> [DC Remove] -> [Gain Normalize] -> [Denoise] -> Output PCM 16-bit 16kHz mono.
"""

from __future__ import annotations

from theo.preprocessing.pipeline import AudioPreprocessingPipeline
from theo.preprocessing.stages import AudioStage

__all__ = ["AudioPreprocessingPipeline", "AudioStage"]
