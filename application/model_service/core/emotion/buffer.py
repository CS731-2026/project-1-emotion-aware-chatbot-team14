import time
import statistics
from collections import deque
from dataclasses import dataclass


@dataclass
class EmotionObservation:
    emotion: str
    confidence: float
    timestamp: float  # unix seconds — from frontend capture time


class EmotionBuffer:
    """Rolling window of emotion observations; smooths predictions via mode."""

    def __init__(self, window: int = 10) -> None:
        self._buffer: deque[EmotionObservation] = deque(maxlen=window)

    def update(self, emotion: str, confidence: float, timestamp: float) -> None:
        """Append a new observation, evicting the oldest when the window is full."""
        self._buffer.append(EmotionObservation(emotion, confidence, timestamp))

    def current(self) -> str:
        """Return the most frequent emotion in the window (statistical mode)."""
        return statistics.mode(o.emotion for o in self._buffer) if self._buffer else "neutral"

    def history(self) -> list[EmotionObservation]:
        """Return a snapshot of all observations currently in the window."""
        return list(self._buffer)

    def seconds_in_current(self) -> float | None:
        """Return how many seconds the dominant emotion has been sustained.

        Scans backwards through the buffer to find the first observation that
        differs from the current dominant emotion; returns elapsed time since
        that point. Returns None if the buffer is empty.
        """
        if not self._buffer:
            return None
        dominant = self.current()
        for obs in reversed(self._buffer):
            if obs.emotion != dominant:
                return time.time() - obs.timestamp
        return time.time() - self._buffer[0].timestamp
