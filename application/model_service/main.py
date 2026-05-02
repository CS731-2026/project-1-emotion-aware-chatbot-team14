import uvicorn
import config
from app import app  # noqa: F401 — imported so uvicorn can reference it

if __name__ == "__main__":
    uvicorn.run("app:app", host=config.HOST, port=config.PORT, reload=True)
