import logging
from contextlib import asynccontextmanager

import config
from fastapi import FastAPI, WebSocket

from routers import prediction
from routers import chat
from ws.handler import handle_websocket

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load inference components based on config.
    # Non-stub stages only: models load when their env vars select real backends.

    try:
        from core.face_detector import FaceDetector
        app.state.face_detector = FaceDetector()
        logger.info("Face detector loaded")
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

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await handle_websocket(websocket)


app.include_router(prediction.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
