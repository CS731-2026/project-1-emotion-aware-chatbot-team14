import os
from dotenv import load_dotenv

load_dotenv()

PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")

STT_ENGINE      = os.getenv("STT_ENGINE",      "whisper-cpp")
STT_MODEL       = os.getenv("STT_MODEL",       "base.en")
WHISPER_CPP_DIR = os.getenv("WHISPER_CPP_DIR", "../../sandbox/student_taurajgreig/vendor/whisper.cpp")
STT_MIN_CONFIDENCE = float(os.getenv("STT_MIN_CONFIDENCE", "0.65"))
STT_MIN_TEXT_CHARS = int(os.getenv("STT_MIN_TEXT_CHARS", "5"))

EMOTION_VARIANT               = os.getenv("EMOTION_VARIANT",         "placeholder")
EMOTION_CHECKPOINT_PATH       = os.getenv("EMOTION_CHECKPOINT_PATH", "models/resnet18_emotion.pth")
EMOTION_DEVICE                = os.getenv("EMOTION_DEVICE") or None  # None = auto-detect
TEST_EMOTIONS                 = os.getenv("TEST_EMOTIONS",           "true").lower() == "true"
TEST_EMOTION_INTERVAL_SECONDS = int(os.getenv("TEST_EMOTION_INTERVAL_SECONDS", "30"))

LLM_PROVIDER    = os.getenv("LLM_PROVIDER",    "openai")
LLM_MODEL       = os.getenv("LLM_MODEL",       "gpt-4o-mini")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
