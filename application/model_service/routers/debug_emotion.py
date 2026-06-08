"""Debug router, runtime overrides for the emotion pipeline.

Powers the toggles on the frontend /emotion-test/ page (force a label,
cycle test labels on a timer, log every prediction). Lets a developer
iterate on the emotion path without restarting the service or editing
.env between tests.

All endpoints mutate core/debug_flags.py singletons in-place, same
flags the WS handler's pick_emotion() consults on every frame.
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from core import debug_flags
from core.emotion.base import EMOTIONS

logger = logging.getLogger(__name__)
router = APIRouter()


_AllowedLabel = Literal[
    "neutral", "trust_relief", "sadness", "fear_anxiety", "confusion", "distrust",
]


class DebugFlagPatch(BaseModel):
    """All fields optional, only present fields are applied. Sending
    `force_label: null` clears the pin; omitting it leaves the pin
    untouched."""
    force_label: _AllowedLabel | None = None
    cycle_test_labels: bool | None = None
    log_predictions: bool | None = None
    cycle_interval_seconds: int | None = None


@router.get("/api/v1/debug/emotion")
async def get_debug_flags() -> dict:
    """Current state of the emotion debug flags."""
    return {
        "force_label":            debug_flags.emotion.force_label,
        "cycle_test_labels":      debug_flags.emotion.cycle_test_labels,
        "cycle_interval_seconds": debug_flags.emotion.cycle_interval_seconds,
        "log_predictions":        debug_flags.emotion.log_predictions,
        "allowed_labels":         list(EMOTIONS),
    }


@router.post("/api/v1/debug/emotion")
async def patch_debug_flags(patch: DebugFlagPatch) -> dict:
    """Apply a partial patch. Only fields present in the request body
    are updated. Returns the full post-patch flag state."""
    body = patch.model_dump(exclude_unset=True)

    if "force_label" in body:
        label = body["force_label"]
        if label is not None and label not in EMOTIONS:
            return {"error": f"force_label must be one of {EMOTIONS}"}
        debug_flags.emotion.force_label = label
        logger.info("debug: force_label = %r", label)

    if "cycle_test_labels" in body:
        debug_flags.emotion.cycle_test_labels = bool(body["cycle_test_labels"])
        logger.info("debug: cycle_test_labels = %s", debug_flags.emotion.cycle_test_labels)

    if "log_predictions" in body:
        debug_flags.emotion.log_predictions = bool(body["log_predictions"])
        logger.info("debug: log_predictions = %s", debug_flags.emotion.log_predictions)

    if "cycle_interval_seconds" in body:
        debug_flags.emotion.cycle_interval_seconds = int(body["cycle_interval_seconds"])
        logger.info("debug: cycle_interval_seconds = %d",
                    debug_flags.emotion.cycle_interval_seconds)

    return await get_debug_flags()
