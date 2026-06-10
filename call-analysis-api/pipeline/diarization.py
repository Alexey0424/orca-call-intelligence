"""Speaker diarization via pyannote, with the 4.0.x ``AudioDecoder`` workaround."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch

from pipeline.audio import load_wav_float32

logger = logging.getLogger(__name__)


def diarize(pipe, wav_path: str, num_speakers: int = 2) -> List[Dict[str, Any]]:
    """Return a sorted list of ``{start, end, speaker}`` turns.

    Workaround: pyannote 4.0.x has a bug where passing a file path triggers
    ``NameError: AudioDecoder`` (the import is missing from ``io.py``).
    Pre-loading the waveform with ``soundfile`` and passing
    ``{"waveform": tensor, "sample_rate": sr}`` skips the broken code path.
    Keep this until pyannote ships a fix.
    """
    y, sr = load_wav_float32(wav_path)
    waveform = torch.from_numpy(y).unsqueeze(0)  # [1, samples]
    output = pipe({"waveform": waveform, "sample_rate": sr},
                  num_speakers=num_speakers)
    annotation = getattr(output, "exclusive_speaker_diarization", None) or output
    turns = [{"start": float(t.start), "end": float(t.end), "speaker": str(s)}
             for t, s in annotation]
    turns.sort(key=lambda x: x["start"])
    logger.info("diarized turns=%d num_speakers=%d", len(turns), num_speakers)
    return turns
