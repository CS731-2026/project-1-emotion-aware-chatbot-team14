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
            return "No emotional signal detected. Proceed with neutral tone."

        # Dominant emotion via statistical mode; fall back to most recent on tie.
        try:
            dominant = statistics.mode(obs.emotion for obs in emotion_observations)
        except statistics.StatisticsError:
            dominant = emotion_observations[-1].emotion

        timestamps = [obs.timestamp for obs in emotion_observations]
        duration_seconds = int(max(timestamps) - min(timestamps))

        prose = EMOTION_PROSE.get(dominant, dominant)
        return (
            f"The user appears to be feeling {prose} (~{duration_seconds}s). "
            "Calibrate tone accordingly without referencing this directly."
        )
