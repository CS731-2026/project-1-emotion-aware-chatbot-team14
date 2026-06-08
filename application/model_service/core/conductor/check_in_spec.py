"""Pydantic mirrors of the frontend's CheckInSpec types.

These are the data shipped in ChatResponse.view.spec when the conductor
decides the frontend should render a check-in form (surface = "checkin").
The shapes must stay aligned with:
  application/frontend/src/lib/conversation/sampleCheckIns.ts

Only the PageSpec / QuestionSpec / Choice triplet is used by the
conductor at present, overlay-style specs are still a debug-only
construct on the frontend.
"""

from typing import Literal, Optional

from pydantic import BaseModel


class Choice(BaseModel):
    label: str
    value: str
    tone: Optional[Literal["neutral", "positive", "concerning"]] = None


class QuestionSpec(BaseModel):
    id: str
    prompt: str
    choices: list[Choice]
    allowFreeText: Optional[bool] = None


class PageSpec(BaseModel):
    elevation: Literal["page"] = "page"
    title: str
    subtitle: Optional[str] = None
    emotionAware: Optional[bool] = None
    questions: list[QuestionSpec]
    reveal: Literal["all-at-once", "sequential"] = "sequential"


# Future: union with OverlaySpec once / if the conductor needs overlay
# check-ins. For now the conductor only ships PageSpec.
CheckInSpec = PageSpec
