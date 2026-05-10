import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any

import config
from core.app_state import HRIAppState
from fastapi import FastAPI, WebSocket

from routers import chat
from ws.handler import handle_websocket

logging.basicConfig(
    level=logging.INFO,
    format="[model_service] %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan handler — loads all ML components once at startup.

    Each component (face detector, STT, emotion model, LLM) is loaded
    independently so a missing dependency only disables that component;
    the service remains partially functional. All components are stored
    in a typed HRIAppState dataclass at app.state.hri.
    """
    hri = HRIAppState()
    app.state.hri = hri

    try:
        from core.face_detector import FaceDetector
        _fd = FaceDetector()
        hri.face_detector = _fd
        logger.info(
            "Face detector loaded on device=%s (%s, torch=%s, mps_built=%s, mps_available=%s)",
            _fd.device, _fd.device_reason, _fd.torch_version, _fd.mps_built, _fd.mps_available,
        )
    except Exception as e:
        logger.warning("Face detector not loaded: %s", e)

    try:
        from core.stt.factory import create_stt
        hri.stt = create_stt(config.STT_ENGINE, config.STT_MODEL)
        logger.info("STT loaded: %s / %s", config.STT_ENGINE, config.STT_MODEL)
    except Exception as e:
        logger.warning("STT not loaded: %s", e)

    # Emotion model — ONLY invoked in ws/handler.py:pick_emotion().
    # To swap in a real model: set EMOTION_VARIANT in .env and implement
    # the EmotionModel ABC in core/emotion/ following the factory pattern.
    # DEBUG: set TEST_EMOTIONS=true to bypass the model and emit random emotions.
    try:
        from core.emotion.factory import create_emotion_model
        hri.emotion_model = create_emotion_model(config.EMOTION_VARIANT)
        logger.info(
            "Emotion model loaded: variant=%s (TEST_EMOTIONS=%s)",
            config.EMOTION_VARIANT, config.TEST_EMOTIONS,
        )
    except Exception as e:
        logger.warning("Emotion model not loaded: %s", e)

    if config.LLM_PROVIDER and config.OPENAI_API_KEY or config.LLM_PROVIDER != "openai":
        try:
            from core.llm.factory import create_llm
            from core.llm.reasoning_agent import LLMReasoningAgent
            from core.emotional_reasoning_agent import EmotionalReasoningAgent

            hri.llm = create_llm(config.LLM_PROVIDER, config.LLM_MODEL, api_key=config.OPENAI_API_KEY)
            hri.llm_agent = LLMReasoningAgent(hri.llm)
            hri.emotion_agent = EmotionalReasoningAgent()
            logger.info("LLM loaded: %s / %s", config.LLM_PROVIDER, config.LLM_MODEL)
        except Exception as e:
            logger.warning("LLM not loaded: %s", e)

    fd = hri.face_detector
    logger.info(
        "Startup summary — face_detector=%s (device=%s) | emotion_model=%s "
        "(variant=%s, test_mode=%s) | stt=%s | llm=%s",
        fd is not None,
        fd.device if fd is not None else "none",
        hri.emotion_model is not None,
        config.EMOTION_VARIANT,
        config.TEST_EMOTIONS,
        hri.stt is not None,
        hri.llm is not None,
    )

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    hri: HRIAppState = app.state.hri
    fd = hri.face_detector
    return {
        "status": "ok",
        "service": "model_service",
        "face_detector_loaded": fd is not None,
        "face_detector_device": fd.device if fd is not None else None,
        "face_detector_device_reason": fd.device_reason if fd is not None else None,
        "torch_version": fd.torch_version if fd is not None else None,
        "mps_built": fd.mps_built if fd is not None else None,
        "mps_available": fd.mps_available if fd is not None else None,
        "emotion_model_loaded": hri.emotion_model is not None,
        "stt_loaded": hri.stt is not None,
        "llm_loaded": hri.llm is not None,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await handle_websocket(websocket)


app.include_router(chat.router, prefix="/api/v1")
