import logging
from contextlib import asynccontextmanager

import config
from fastapi import FastAPI, WebSocket

from routers import prediction
from routers import chat
from ws.handler import handle_websocket

logging.basicConfig(
    level=logging.INFO,
    format="[model_service] %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler — loads all ML components once at startup.

    Each component (face detector, STT, emotion model, LLM) is loaded
    independently so a missing dependency only disables that component;
    the service remains partially functional. Loaded instances are stored
    on app.state so every request handler can access them without globals.
    """
    # Load inference components based on config.
    # Non-stub stages only: models load when their env vars select real backends.

    try:
        from core.face_detector import FaceDetector
        app.state.face_detector = FaceDetector()
        logger.info(
            "Face detector loaded on device=%s (%s, torch=%s, mps_built=%s, mps_available=%s)",
            getattr(app.state.face_detector, "device", "unknown"),
            getattr(app.state.face_detector, "device_reason", "unknown"),
            getattr(app.state.face_detector, "torch_version", "unknown"),
            getattr(app.state.face_detector, "mps_built", None),
            getattr(app.state.face_detector, "mps_available", None),
        )
    except Exception as e:
        logger.warning("Face detector not loaded: %s", e)
        app.state.face_detector = None

    try:
        from core.stt.factory import create_stt
        app.state.stt = create_stt(config.STT_ENGINE, config.STT_MODEL)
        logger.info("STT loaded: %s / %s", config.STT_ENGINE, config.STT_MODEL)
    except Exception as e:
        logger.warning("STT not loaded: %s", e)
        app.state.stt = None

    try:
        from core.emotion.factory import create_emotion_model
        app.state.emotion_model = create_emotion_model(config.EMOTION_VARIANT)
        logger.info("Emotion model loaded: %s", config.EMOTION_VARIANT)
    except Exception as e:
        logger.warning("Emotion model not loaded: %s", e)
        app.state.emotion_model = None

    if config.LLM_PROVIDER and config.OPENAI_API_KEY or config.LLM_PROVIDER != "openai":
        try:
            from core.llm.factory import create_llm
            from core.llm.reasoning_agent import LLMReasoningAgent
            from core.emotional_reasoning_agent import EmotionalReasoningAgent

            llm = create_llm(config.LLM_PROVIDER, config.LLM_MODEL, api_key=config.OPENAI_API_KEY)
            app.state.llm         = llm
            app.state.llm_agent   = LLMReasoningAgent(llm)
            app.state.emotion_agent = EmotionalReasoningAgent()
            logger.info("LLM loaded: %s / %s", config.LLM_PROVIDER, config.LLM_MODEL)
        except Exception as e:
            logger.warning("LLM not loaded: %s", e)
            app.state.llm         = None
            app.state.llm_agent   = None
            app.state.emotion_agent = None
    else:
        app.state.llm         = None
        app.state.llm_agent   = None
        app.state.emotion_agent = None

    logger.info(
        "Startup summary: face_detector=%s face_device=%s emotion_model=%s stt=%s llm=%s test_emotions=%s",
        app.state.face_detector is not None,
        getattr(app.state.face_detector, "device", "none"),
        app.state.emotion_model is not None,
        app.state.stt is not None,
        app.state.llm is not None,
        config.TEST_EMOTIONS,
    )

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "model_service",
        "face_detector_loaded": app.state.face_detector is not None,
        "face_detector_device": getattr(app.state.face_detector, "device", None),
        "face_detector_device_reason": getattr(app.state.face_detector, "device_reason", None),
        "torch_version": getattr(app.state.face_detector, "torch_version", None),
        "mps_built": getattr(app.state.face_detector, "mps_built", None),
        "mps_available": getattr(app.state.face_detector, "mps_available", None),
        "emotion_model_loaded": app.state.emotion_model is not None,
        "stt_loaded": app.state.stt is not None,
        "llm_loaded": app.state.llm is not None,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await handle_websocket(websocket)


app.include_router(prediction.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
