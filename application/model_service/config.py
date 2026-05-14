import os
from dotenv import load_dotenv

load_dotenv()

PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")

STT_ENGINE      = os.getenv("STT_ENGINE",      "whisper-cpp")
STT_MODEL       = os.getenv("STT_MODEL",       "base.en")
WHISPER_CPP_DIR = os.getenv("WHISPER_CPP_DIR", "../../sandbox/student_taurajgreig/vendor/whisper.cpp")

EMOTION_VARIANT = os.getenv("EMOTION_VARIANT", "placeholder")
TEST_EMOTIONS   = os.getenv("TEST_EMOTIONS",   "true").lower() == "true"

LLM_PROVIDER    = os.getenv("LLM_PROVIDER",    "gemini")
LLM_MODEL       = os.getenv("LLM_MODEL",       "gemini-2.5-flash")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
