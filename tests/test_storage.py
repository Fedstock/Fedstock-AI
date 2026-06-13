from __future__ import annotations

import shutil
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import storage


class FakeS3Client:
    def __init__(self) -> None:
        self.downloads: list[tuple[str, str, str]] = []
        self.uploads: list[tuple[str, str, str]] = []

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.downloads.append((bucket, key, filename))
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(b"model-bytes")

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.uploads.append((filename, bucket, key))


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {name}")
    if not ok and detail:
        print(f"        {detail}")
    return ok


def test_parse_s3_uri() -> None:
    bucket, key = storage.parse_s3_uri("s3://fedstock-artifacts/models/global/v1/global.pt")
    assert bucket == "fedstock-artifacts"
    assert key == "models/global/v1/global.pt"


def test_parse_s3_uri_rejects_invalid_scheme() -> None:
    try:
        storage.parse_s3_uri("https://fedstock-artifacts/models/global.pt")
    except storage.StorageError as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("invalid scheme should fail")


def test_build_s3_uri() -> None:
    assert (
        storage.build_s3_uri("fedstock-artifacts", "/models/global.pt")
        == "s3://fedstock-artifacts/models/global.pt"
    )


def test_build_s3_uri_uses_env_bucket() -> None:
    old_value = os.environ.get("ARTIFACT_BUCKET")
    os.environ["ARTIFACT_BUCKET"] = "env-artifacts"
    try:
        assert storage.build_s3_uri("", "models/global.pt") == "s3://env-artifacts/models/global.pt"
    finally:
        if old_value is None:
            os.environ.pop("ARTIFACT_BUCKET", None)
        else:
            os.environ["ARTIFACT_BUCKET"] = old_value


def test_download_s3_file_uses_client(tmp_path: Path) -> None:
    client = FakeS3Client()
    local_path = tmp_path / "models" / "global.pt"
    result = storage.download_s3_file(
        "s3://fedstock-artifacts/models/global.pt",
        local_path,
        s3_client=client,
    )
    assert result == local_path
    assert local_path.read_bytes() == b"model-bytes"
    assert client.downloads == [
        ("fedstock-artifacts", "models/global.pt", str(local_path))
    ]


def test_upload_s3_file_uses_client(tmp_path: Path) -> None:
    client = FakeS3Client()
    local_path = tmp_path / "cluster.pt"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(b"cluster")
    uri = storage.upload_s3_file(
        local_path,
        "s3://fedstock-artifacts/models/clusters/round-1/cluster-0.pt",
        s3_client=client,
    )
    assert uri == "s3://fedstock-artifacts/models/clusters/round-1/cluster-0.pt"
    assert client.uploads == [
        (str(local_path), "fedstock-artifacts", "models/clusters/round-1/cluster-0.pt")
    ]


def run() -> int:
    tests = [
        ("parse s3 uri", test_parse_s3_uri),
        ("parse rejects invalid scheme", test_parse_s3_uri_rejects_invalid_scheme),
        ("build s3 uri", test_build_s3_uri),
        ("build s3 uri uses env bucket", test_build_s3_uri_uses_env_bucket),
        ("download uses client", test_download_s3_file_uses_client),
        ("upload uses client", test_upload_s3_file_uses_client),
    ]
    failures = 0
    tmp_root = Path("outputs") / "_test_storage"
    shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    try:
        for idx, (name, test) in enumerate(tests):
            try:
                if "tmp_path" in test.__code__.co_varnames:
                    test(tmp_root / str(idx))
                else:
                    test()
                check(name, True)
            except Exception as exc:
                failures += 1
                check(name, False, repr(exc))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"\n==== {len(tests) - failures} passed, {failures} failed ====")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())
