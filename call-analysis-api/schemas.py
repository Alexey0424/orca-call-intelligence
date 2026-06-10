"""Pydantic response models for ``/scorecard``.

Intentionally permissive — sub-objects come straight from the LLM and we
don't want to reject a valid call over a stray field. Top-level shape is
enforced; nested blocks are ``Dict[str, Any]``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TimingsBlock(BaseModel):
    convert:      float
    asr:          float
    diarize:      float
    scorecard_ml: float
    total:        float


class MetadataBlock(BaseModel):
    call_id:                str
    file_name:              Optional[str] = None
    language:               Optional[str] = None
    num_speakers_requested: int
    closer_speaker:         str
    closer_name:            Optional[str] = None
    closer_confidence:      Optional[float] = None
    client_speaker:         Optional[str] = None
    client_name:            Optional[str] = None
    model_versions:         Dict[str, Any]


class ScorecardResponse(BaseModel):
    ok:           bool = True
    metadata:     MetadataBlock
    timings_sec:  TimingsBlock
    scorecard:    Dict[str, Any]
    utterances:   Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Per-utterance annotations. Only present when "
                    "include_utterances=true.",
    )
