from fastapi import APIRouter, Request
from pydantic import BaseModel

from ws.handler import get_session

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    profile_id: str
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    response: str


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    session = get_session(body.profile_id)
    emotion_observations = session.emotion_buffer.history() if session else []
    transcript_segments  = session.transcript_buffer[-20:] if session else []

    # Use agents from app.state if loaded (Stage 8+), otherwise stub
    emotion_agent = getattr(request.app.state, "emotion_agent", None)
    llm_agent     = getattr(request.app.state, "llm_agent",     None)

    if emotion_agent and llm_agent:
        ctx      = emotion_agent.analyse(emotion_observations, transcript_segments)
        history  = [{"role": m.role, "content": m.content} for m in body.history]
        response = llm_agent.reason(body.message, ctx, history)
    else:
        # Stage 1–7 stub
        response = "stub response from harness"

    return ChatResponse(response=response)
