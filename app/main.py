from pathlib import Path

from fastapi import FastAPI

from app.config import get_settings


app = FastAPI(title="Fedstock AI API")


@app.get("/ai/health")
def health_check():
    settings = get_settings()
    model_dir = Path(settings.model_local_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    return {
        "status": "ok",
        "service": "fedstock-ai",
        "env": settings.env,
        "aws_region": settings.aws_region,
        "artifact_bucket_configured": bool(settings.artifact_bucket),
        "model_local_dir": str(model_dir),
    }
