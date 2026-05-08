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
    def __init__(self, window: int = 10) -> None:
        self._buffer: deque[EmotionObservation] = deque(maxlen=window)

    def update(self, emotion: str, confidence: float, timestamp: float) -> None:
        self._buffer.append(EmotionObservation(emotion, confidence, timestamp))

    def current(self) -> str:
        return statistics.mode(o.emotion for o in self._buffer) if self._buffer else "neutral"

    def history(self) -> list[EmotionObservation]:
        return list(self._buffer)

    def seconds_in_current(self) -> float | None:
        if not self._buffer:
            return None
        dominant = self.current()
        for obs in reversed(self._buffer):
            if obs.emotion != dominant:
                return time.time() - obs.timestamp
        return time.time() - self._buffer[0].timestamp
