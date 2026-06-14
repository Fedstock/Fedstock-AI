from __future__ import annotations

import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import main


TEST_ROOT = Path("outputs") / "_test_s3_aggregation"


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {name}")
    if not ok and detail:
        print(f"        {detail}")
    return ok


def _state(value: float) -> dict[str, torch.Tensor]:
    return {"weight": torch.tensor([value], dtype=torch.float32)}


@contextmanager
def patched_runtime():
    old_env = os.environ.get("MODEL_LOCAL_DIR")
    old_download = getattr(main, "download_s3_file", None)
    old_upload = getattr(main, "upload_s3_file", None)
    old_model_dir = getattr(main, "MODEL_LOCAL_DIR", None)
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    source_dir = TEST_ROOT / "source"
    source_dir.mkdir()
    uploaded_dir = TEST_ROOT / "uploaded"
    uploaded_dir.mkdir()
    torch.save(_state(1.0), source_dir / "client-a.pt")
    torch.save(_state(3.0), source_dir / "client-b.pt")
    uri_to_path = {
        "s3://bucket/updates/round-1/clients/A.pt": source_dir / "client-a.pt",
        "s3://bucket/updates/round-1/clients/B.pt": source_dir / "client-b.pt",
    }
    uploaded: dict[str, Path] = {}

    def fake_download(uri: str, local_path: Path) -> Path:
        source = uri_to_path[uri]
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, local_path)
        return local_path

    def fake_upload(local_path: Path, uri: str) -> str:
        copied = uploaded_dir / Path(uri).name
        shutil.copyfile(local_path, copied)
        uploaded[uri] = copied
        return uri

    os.environ["MODEL_LOCAL_DIR"] = str(TEST_ROOT / "models")
    main.MODEL_LOCAL_DIR = Path(os.environ["MODEL_LOCAL_DIR"])
    main.download_s3_file = fake_download
    main.upload_s3_file = fake_upload
    try:
        yield uploaded
    finally:
        if old_env is None:
            os.environ.pop("MODEL_LOCAL_DIR", None)
        else:
            os.environ["MODEL_LOCAL_DIR"] = old_env
        if old_model_dir is not None:
            main.MODEL_LOCAL_DIR = old_model_dir
        if old_download is not None:
            main.download_s3_file = old_download
        if old_upload is not None:
            main.upload_s3_file = old_upload
        shutil.rmtree(TEST_ROOT, ignore_errors=True)


def test_s3_aggregation_returns_s3_uri_and_weighted_model() -> None:
    with patched_runtime() as uploaded:
        client = TestClient(main.app)
        response = client.post(
            "/clients/fl-model/aggregate",
            json={
                "scope": "all_clients",
                "roundId": "round-1",
                "clusterId": "0",
                "modelVersion": "v1",
                "outputPrefixUri": "s3://bucket/models/clusters/round-1",
                "models": [
                    {
                        "clientId": "A",
                        "sampleCount": 1,
                        "modelArtifactUri": "s3://bucket/updates/round-1/clients/A.pt",
                    },
                    {
                        "clientId": "B",
                        "sampleCount": 3,
                        "modelArtifactUri": "s3://bucket/updates/round-1/clients/B.pt",
                    },
                ],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["aggregatedModelUri"] == "s3://bucket/models/clusters/round-1/cluster-0.pt"
        assert body["roundId"] == "round-1"
        assert body["clusterId"] == "0"
        assert body["modelVersion"] == "v1"
        assert body["receivedClientCount"] == 2
        assert "path" not in body
        saved = torch.load(uploaded[body["aggregatedModelUri"]], map_location="cpu", weights_only=True)
        assert float(saved["weight"][0]) == 2.5
        assert not (main.MODEL_LOCAL_DIR / "round-1").exists()


def test_s3_aggregation_rejects_count_mismatch() -> None:
    with patched_runtime():
        client = TestClient(main.app)
        response = client.post(
            "/clients/fl-model/aggregate",
            json={
                "scope": "all_clients",
                "roundId": "round-1",
                "expectedClientCount": 3,
                "outputPrefixUri": "s3://bucket/models/global/round-1",
                "models": [
                    {
                        "clientId": "A",
                        "sampleCount": 1,
                        "modelArtifactUri": "s3://bucket/updates/round-1/clients/A.pt",
                    }
                ],
            },
        )
        assert response.status_code == 409


def test_s3_aggregation_rejects_artifact_bucket_mismatch() -> None:
    with patched_runtime():
        old_bucket = os.environ.get("ARTIFACT_BUCKET")
        os.environ["ARTIFACT_BUCKET"] = "bucket"
        try:
            client = TestClient(main.app)
            response = client.post(
                "/clients/fl-model/aggregate",
                json={
                    "scope": "all_clients",
                    "roundId": "round-1",
                    "outputPrefixUri": "s3://other-bucket/models/global/round-1",
                    "models": [
                        {
                            "clientId": "A",
                            "sampleCount": 1,
                            "modelArtifactUri": "s3://bucket/updates/round-1/clients/A.pt",
                        }
                    ],
                },
            )
            assert response.status_code == 400
        finally:
            if old_bucket is None:
                os.environ.pop("ARTIFACT_BUCKET", None)
            else:
                os.environ["ARTIFACT_BUCKET"] = old_bucket


def run() -> int:
    tests = [
        ("s3 aggregation returns s3 uri and weighted model", test_s3_aggregation_returns_s3_uri_and_weighted_model),
        ("s3 aggregation rejects count mismatch", test_s3_aggregation_rejects_count_mismatch),
        ("s3 aggregation rejects artifact bucket mismatch", test_s3_aggregation_rejects_artifact_bucket_mismatch),
    ]
    failures = 0
    for name, test in tests:
        try:
            test()
            check(name, True)
        except Exception as exc:
            failures += 1
            check(name, False, repr(exc))
    print(f"\n==== {len(tests) - failures} passed, {failures} failed ====")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())
