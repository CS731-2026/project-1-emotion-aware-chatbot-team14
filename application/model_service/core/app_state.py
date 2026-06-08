"""Typed container for all ML components loaded at startup (app.py:lifespan).

Access pattern in request handlers:
    from typing import cast
    from core.app_state import HRIAppState
    hri = cast(HRIAppState, request.app.state.hri)
"""

from __future__ import annotations

from dataclasses import dataclass

from core.emotion.base import EmotionModel
from core.emotional_reasoning_agent import EmotionalReasoningAgent
from core.face_detector import FaceDetector
from core.llm.base import LLMProvider
from core.llm.reasoning_agent import LLMReasoningAgent
from core.stt.base import TranscriptionService


@dataclass
class HRIAppState:
    """All ML components loaded at startup; each field is None until loaded.

    Component load failures are independent, a missing model sets only that
    field to None; the rest of the service continues to run.
    """
    face_detector: FaceDetector | None = None
    emotion_model: EmotionModel | None = None
    stt: TranscriptionService | None = None
    llm: LLMProvider | None = None
    llm_agent: LLMReasoningAgent | None = None
    emotion_agent: EmotionalReasoningAgent | None = None
