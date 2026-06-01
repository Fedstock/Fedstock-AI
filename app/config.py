import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    env: str
    aws_region: str
    artifact_bucket: str | None
    model_table: str
    round_table: str
    participant_update_table: str
    model_local_dir: str


def get_settings() -> Settings:
    return Settings(
        env=os.getenv("ENV", "local"),
        aws_region=os.getenv("AWS_REGION", "ap-northeast-2"),
        artifact_bucket=os.getenv("ARTIFACT_BUCKET"),
        model_table=os.getenv("MODEL_TABLE", "fl-mlops-prod-model-version-table"),
        round_table=os.getenv("ROUND_TABLE", "fl-mlops-prod-round-table"),
        participant_update_table=os.getenv(
            "PARTICIPANT_UPDATE_TABLE",
            "fl-mlops-prod-participant-update-table",
        ),
        model_local_dir=os.getenv("MODEL_LOCAL_DIR", "/tmp/models"),
    )
