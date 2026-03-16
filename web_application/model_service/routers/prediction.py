from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class PredictRequest(BaseModel):
    input: object


class PredictResponse(BaseModel):
    message: str


@router.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest) -> PredictResponse:
    # TODO: load and run the model against body.input
    return PredictResponse(message="predict stub from model service")
