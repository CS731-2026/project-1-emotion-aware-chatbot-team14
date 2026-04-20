from fastapi import FastAPI
from routers import prediction

app = FastAPI()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


app.include_router(prediction.router, prefix="/api/v1")
