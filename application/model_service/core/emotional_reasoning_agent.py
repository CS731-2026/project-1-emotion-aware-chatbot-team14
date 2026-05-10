"""Emotional reasoning agent.

Analyses a window of emotion observations and transcript segments to produce
a short emotional-context string that the LLMReasoningAgent injects as a
system-level instruction before each student turn.
"""

from __future__ import annotations

import statistics

from core.emotion.buffer import EmotionObservation
from ws.session import TranscriptSegment


class EmotionalReasoningAgent:
    """Derives a concise emotional-context description from recent observations.

    This is a placeholder implementation: it uses statistics.mode over the
    raw emotion labels and computes duration from the earliest-to-latest
    timestamp in the observation window.  No ML model is involved.
    """

    def analyse(
        self,
        emotion_observations: list[EmotionObservation],
        transcript_segments: list[TranscriptSegment],
    ) -> str:
        """Summarise the student's emotional state for the LLM.

        Args:
            emotion_observations: Recent EmotionObservation records from the
                                  emotion buffer.
            transcript_segments:  Recent TranscriptSegment records (unused in
                                  this placeholder but kept for future use).

        Returns:
            A single-sentence emotional context instruction, e.g.:
            "The student appears to be feeling happy (~12s). Calibrate tone
            accordingly without referencing this directly."

            Returns a neutral fallback if no observations are present.
        """
        if not emotion_observations:
            return (
                "No emotional signal detected. "
                "Proceed with neutral tone."
            )

        # Determine the dominant emotion via statistical mode.
        try:
            dominant = statistics.mode(
                obs.emotion for obs in emotion_observations
            )
        except statistics.StatisticsError:
            # No unique mode — fall back to the last observed emotion.
            dominant = emotion_observations[-1].emotion

        # Compute duration from earliest to latest timestamp in the window.
        timestamps = [obs.timestamp for obs in emotion_observations]
        duration_seconds = int(max(timestamps) - min(timestamps))

        return (
            f"The student appears to be feeling {dominant} "
            f"(~{duration_seconds}s). "
            "Calibrate tone accordingly without referencing this directly."
        )
