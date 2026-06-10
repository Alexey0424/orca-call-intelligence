"""Audio I/O helpers: upload save, ffmpeg conversion, soundfile loading."""
from __future__ import annotations

import logging
import os
import subprocess
from typing import Tuple

import numpy as np
import soundfile as sf
from fastapi import UploadFile

logger = logging.getLogger(__name__)


def save_upload(file: UploadFile, tmpdir: str) -> str:
    path = os.path.join(tmpdir, file.filename or "audio.bin")
    with open(path, "wb") as f:
        f.write(file.file.read())
    return path


def to_wav_16k_mono(src: str, tmpdir: str) -> str:
    """Convert any audio file to a 16 kHz mono PCM-16 WAV.

    Raises ``subprocess.CalledProcessError`` if ffmpeg fails.
    """
    dst = os.path.join(tmpdir, "audio_16k_mono.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", src,
         "-ac", "1", "-ar", "16000", "-vn",
         "-acodec", "pcm_s16le", dst],
        check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return dst


def wav_duration(path: str) -> float:
    return float(sf.info(path).duration)


def load_wav_float32(path: str) -> Tuple[np.ndarray, int]:
    """Read a WAV as float32, mono, 1-D."""
    y, sr = sf.read(path, dtype="float32")
    y1d = y[:, 0] if y.ndim > 1 else y
    return y1d.astype(np.float32, copy=False), int(sr)
