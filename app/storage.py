from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from app.config import get_settings


class StorageError(Exception):
    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    parsed = urlparse(str(s3_uri).strip())
    if parsed.scheme != "s3" or not parsed.netloc:
        raise StorageError("S3 URI must use s3://bucket/key format.", status_code=400)
    key = parsed.path.lstrip("/")
    if not key:
        raise StorageError("S3 URI must include an object key.", status_code=400)
    return parsed.netloc, key


def build_s3_uri(bucket: str, key: str) -> str:
    clean_bucket = str(bucket or get_settings().artifact_bucket or "").strip()
    clean_key = str(key or "").strip().lstrip("/")
    if not clean_bucket:
        raise StorageError("S3 bucket is required or ARTIFACT_BUCKET must be configured.", status_code=500)
    if not clean_key:
        raise StorageError("S3 key is required.", status_code=400)
    return f"s3://{clean_bucket}/{clean_key}"


def _s3_client():
    region = get_settings().aws_region
    if not region:
        raise StorageError("AWS_REGION is not configured.", status_code=500)
    try:
        import boto3
    except ModuleNotFoundError as exc:
        raise StorageError("boto3 is required for S3 storage access.", status_code=500) from exc
    return boto3.client("s3", region_name=region)


def download_s3_file(s3_uri: str, local_path: Path, *, s3_client=None) -> Path:
    bucket, key = parse_s3_uri(s3_uri)
    client = s3_client or _s3_client()
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(bucket, key, str(local_path))
    except Exception as exc:
        raise StorageError(f"Failed to download S3 artifact: {s3_uri}", status_code=503) from exc
    if not local_path.exists():
        raise StorageError(f"S3 download did not create local file: {s3_uri}", status_code=500)
    return local_path


def upload_s3_file(local_path: Path, s3_uri: str, *, s3_client=None) -> str:
    bucket, key = parse_s3_uri(s3_uri)
    local_path = Path(local_path)
    if not local_path.exists() or not local_path.is_file():
        raise StorageError(f"Local artifact does not exist: {local_path}", status_code=500)
    client = s3_client or _s3_client()
    try:
        client.upload_file(str(local_path), bucket, key)
    except Exception as exc:
        raise StorageError(f"Failed to upload S3 artifact: {s3_uri}", status_code=503) from exc
    return build_s3_uri(bucket, key)
