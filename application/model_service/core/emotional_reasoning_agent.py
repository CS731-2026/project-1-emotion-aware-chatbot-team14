"""Emotional reasoning agent.

Analyses a window of emotion observations to produce a short emotional-context
string. This string is one of two inputs injected into LLMReasoningAgent per
the architecture spec (the other being the timestamped transcript).
"""

from __future__ import annotations

import statistics

from core.emotion.base import EMOTION_PROSE
from core.emotion.buffer import EmotionObservation
from ws.session import TranscriptSegment

POSITIVE_PHRASES = frozenset({
    "fine", "i'm fine", "feel fine", "i'm good", "i'm okay", "i'm ok",
    "good", "okay", "ok", "alright", "great", "no problem",
    "feeling okay", "feeling fine", "i feel good", "feel well",
})
CONCERNING_EMOTIONS = frozenset({"fear_anxiety", "distrust", "sadness", "confusion"})


class EmotionalReasoningAgent:
    """Derives a concise emotional-context string from recent face-based observations.

    Placeholder implementation: uses statistics.mode over raw emotion labels and
    computes duration from the earliest-to-latest timestamp in the window.
    Transcript segments are accepted but not yet used — incorporating verbal cues
    into the emotional context is a future prompt engineering decision.
    """

    def analyse(
        self,
        emotion_observations: list[EmotionObservation],
        transcript_segments: list[TranscriptSegment],
        session_dominant: str | None = None,
    ) -> str:
        """Summarise the user's emotional state for the LLM.

        Args:
            emotion_observations: Recent EmotionObservation records from the
                                  emotion buffer (face-based signal).
            transcript_segments:  Recent TranscriptSegment records from the STT
                                  pipeline. Accepted for future use; currently
                                  not factored into the output string.

        Returns:
            A short emotional context instruction, e.g.:
            "The user appears to be feeling anxious or fearful (~12s). Calibrate
            tone accordingly without referencing this directly."

            Returns a neutral fallback if no observations are present.
        """
        if not emotion_observations:
            return (
                "No clear emotional signal was detected. "
                "Respond with a warm, calm, and supportive tone — "
                "the patient may be tired or unfamiliar with the technology."
            )

        # Dominant emotion via statistical mode; fall back to most recent on tie.
        try:
            dominant = statistics.mode(obs.emotion for obs in emotion_observations)
        except statistics.StatisticsError:
            dominant = emotion_observations[-1].emotion

        timestamps = [obs.timestamp for obs in emotion_observations]
        duration_seconds = int(max(timestamps) - min(timestamps))

        prose = EMOTION_PROSE.get(dominant, dominant)
        lines = [
            f"From recent face-based cues observed through the webcam, "
            f"the user appears to be feeling {prose} (~{duration_seconds}s).",
        ]
        if session_dominant and session_dominant != dominant:
            dominant_prose = EMOTION_PROSE.get(session_dominant, session_dominant)
            lines.append(
                f"Across the full session, their most prevalent emotional state has been: {dominant_prose}."
            )
        lines.append(
            "Respond with genuine empathy calibrated to these cues. "
            "If the patient appears anxious, confused, or guarded, "
            "validate their feeling first — one warm sentence — before "
            "any explanation or reassurance. "
            "Never mention the webcam, emotion detection, or any internal signal."
        )

        recent_text = " ".join(s.text.lower() for s in transcript_segments[-5:])
        mismatch = dominant in CONCERNING_EMOTIONS and any(p in recent_text for p in POSITIVE_PHRASES)
        if mismatch:
            lines.append(
                "MISMATCH DETECTED: The patient said they feel fine, but their face suggests otherwise. "
                "Gently acknowledge this — e.g. 'It sounds like you're doing okay, though I want to make sure.' "
                "Offer 2–3 simple options to help them feel at ease (taking their time, skipping a question, "
                "or having something explained). Be warm and humble — never alarming or intrusive."
            )

        return " ".join(lines)
