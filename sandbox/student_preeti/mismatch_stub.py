"""
Mismatch detection.

Compares what the user SAID (stated sentiment) against what their FACE showed
during the answer (facial emotion window).

This stub uses a simple keyword check instead of VADER so the UI runs with
zero extra dependencies. Swap for the VADER version once added to requirements.
"""
from collections import Counter

NEGATIVE_EMOTIONS = {"angry", "sad", "disgust", "fear"}

_POSITIVE_WORDS = {
    "good", "fine", "okay", "ok", "great", "happy", "alright",
    "no problem", "all good", "thanks", "thank you", "yes",
}
_NEGATIVE_WORDS = {
    "bad", "worried", "scared", "confused", "no", "not", "unhappy",
    "uncomfortable", "frustrated", "angry",
}


def _classify_stated(text: str) -> str:
    """Crude positive/neutral/negative classifier. Replace with VADER."""
    t = text.lower()
    pos = sum(w in t for w in _POSITIVE_WORDS)
    neg = sum(w in t for w in _NEGATIVE_WORDS)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def detect_mismatch(stated_text: str, emotion_window: list[str]) -> dict:
    """
    Args:
        stated_text: the answer the user typed/spoke
        emotion_window: list of emotion labels captured during the answer

    Returns:
        {mismatch, stated_sentiment, facial_negativity, dominant_facial_emotion}
    """
    stated = _classify_stated(stated_text)

    if not emotion_window:
        return {
            "mismatch": False,
            "stated_sentiment": stated,
            "facial_negativity": None,
            "dominant_facial_emotion": None,
        }

    neg_count = sum(e in NEGATIVE_EMOTIONS for e in emotion_window)
    neg_ratio = neg_count / len(emotion_window)
    dominant = Counter(emotion_window).most_common(1)[0][0]

    # Flag only if: said something positive AND face was mostly negative AND
    # we have enough frames to be confident
    mismatch = (
        stated == "positive"
        and neg_ratio > 0.6
        and len(emotion_window) >= 10
    )

    return {
        "mismatch": mismatch,
        "stated_sentiment": stated,
        "facial_negativity": round(neg_ratio, 2),
        "dominant_facial_emotion": dominant,
    }
