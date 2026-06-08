import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# Repo root, resolved relative to this file (config.py lives in application/model_service/).
REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_CONFIG_PATH = Path(__file__).resolve().parent / "models.yaml"


def load_model_registry() -> dict:
    """Parse models.yaml. Returns the `models:` mapping (id → entry)."""
    if not MODELS_CONFIG_PATH.exists():
        return {}
    with MODELS_CONFIG_PATH.open() as f:
        data = yaml.safe_load(f) or {}
    return data.get("models", {})

PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")

STT_ENGINE      = os.getenv("STT_ENGINE",      "whisper-cpp")
STT_MODEL       = os.getenv("STT_MODEL",       "base.en")
WHISPER_CPP_DIR = os.getenv("WHISPER_CPP_DIR", "../../sandbox/student_taurajgreig/vendor/whisper.cpp")
STT_MIN_CONFIDENCE = float(os.getenv("STT_MIN_CONFIDENCE", "0.65"))
STT_MIN_TEXT_CHARS = int(os.getenv("STT_MIN_TEXT_CHARS", "5"))

EMOTION_VARIANT               = os.getenv("EMOTION_VARIANT",         "placeholder")
EMOTION_CHECKPOINT_PATH       = os.getenv("EMOTION_CHECKPOINT_PATH", "models/resnet18_emotion.pth")
# When set, the factory resolves variant + path from models.yaml and ignores
# EMOTION_VARIANT / EMOTION_CHECKPOINT_PATH. Preferred over the raw vars above.
EMOTION_MODEL_ID              = os.getenv("EMOTION_MODEL_ID") or None
EMOTION_DEVICE                = os.getenv("EMOTION_DEVICE") or None  # None = auto-detect

# ── Emotion debug flags ──────────────────────────────────────────────────────
# These are env-var defaults. The runtime values live in core/debug_flags.py
# and can be mutated at runtime from any code path (e.g. routers/chat.py).
# DEBUG: cycle through EMOTIONS on a fixed timer; bypasses the model entirely.
EMOTION_CYCLE_TEST_LABELS     = os.getenv("EMOTION_CYCLE_TEST_LABELS", "false").lower() == "true"
EMOTION_CYCLE_INTERVAL_SECONDS = int(os.getenv("EMOTION_CYCLE_INTERVAL_SECONDS", "30"))
# DEBUG: pin a specific label (must be in EMOTIONS), overrides cycle + model.
EMOTION_FORCE_LABEL           = os.getenv("EMOTION_FORCE_LABEL") or None
# DEBUG: log every model prediction at INFO level.
EMOTION_LOG_PREDICTIONS       = os.getenv("EMOTION_LOG_PREDICTIONS", "false").lower() == "true"

LLM_PROVIDER    = os.getenv("LLM_PROVIDER",    "openai")
LLM_MODEL       = os.getenv("LLM_MODEL",       "gpt-4o-mini")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
