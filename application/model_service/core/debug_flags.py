"""Runtime debug flags.

Centralised, mutable runtime state seeded from .env at startup.

Two-layer design:
  - .env / config.py provides the *defaults* — read once at process start.
  - This module exposes those defaults as mutable singletons that any code
    path (routers, websocket handlers, ad-hoc shell hooks) can override
    at runtime without restarting the service.

Usage:
    from core import debug_flags

    # Read
    if debug_flags.emotion.cycle_test_labels:
        ...

    # Override at runtime (e.g. from routers/chat.py)
    debug_flags.emotion.force_label = "sadness"
"""

from __future__ import annotations

from dataclasses import dataclass

import config


@dataclass
class EmotionDebug:
    """Runtime overrides for the emotion pipeline.

    Resolution priority in ws/handler.py::pick_emotion is:
      1. ``force_label``      — pin a specific emotion (bypasses everything)
      2. ``cycle_test_labels`` — step through EMOTIONS on a timer (bypasses model)
      3. Real model output
      4. Neutral fallback

    ``log_predictions`` is independent — when true, every real model
    prediction is logged at INFO level.
    """

    cycle_test_labels: bool
    cycle_interval_seconds: int
    force_label: str | None
    log_predictions: bool


emotion = EmotionDebug(
    cycle_test_labels=config.EMOTION_CYCLE_TEST_LABELS,
    cycle_interval_seconds=config.EMOTION_CYCLE_INTERVAL_SECONDS,
    force_label=config.EMOTION_FORCE_LABEL,
    log_predictions=config.EMOTION_LOG_PREDICTIONS,
)
