from __future__ import annotations

import json
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


TEST_ROOT = Path("outputs") / "_test_fl_model_batch"


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {name}")
    if not ok and detail:
        print(f"        {detail}")
    return ok


def _model_bytes(value: float) -> bytes:
    path = TEST_ROOT / f"model-{value}.pt"
    torch.save({"weight": torch.tensor([value], dtype=torch.float32)}, path)
    return path.read_bytes()


@contextmanager
def patched_model_dir():
    old_env = os.environ.get("MODEL_LOCAL_DIR")
    old_model_dir = getattr(main, "MODEL_LOCAL_DIR", None)
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["MODEL_LOCAL_DIR"] = str(TEST_ROOT / "models")
    main.MODEL_LOCAL_DIR = Path(os.environ["MODEL_LOCAL_DIR"])
    try:
        yield
    finally:
        if old_env is None:
            os.environ.pop("MODEL_LOCAL_DIR", None)
        else:
            os.environ["MODEL_LOCAL_DIR"] = old_env
        if old_model_dir is not None:
            main.MODEL_LOCAL_DIR = old_model_dir
        shutil.rmtree(TEST_ROOT, ignore_errors=True)


def _metadata(expected_count: int = 2) -> dict[str, object]:
    return {
        "scope": "all_clients",
        "round_id": "fl-sync-1",
        "expected_client_count": expected_count,
        "models": [
            {"client_id": "A", "sample_weight": 10, "filename": "client_A.pt"},
            {"client_id": "B", "sample_weight": 20, "filename": "client_B.pt"},
        ],
    }


def test_batch_accepts_metadata_and_model_files() -> None:
    with patched_model_dir():
        client = TestClient(main.app)
        response = client.post(
            "/clients/fl-model/batch",
            files=[
                ("metadata", (None, json.dumps(_metadata()), "application/json")),
                ("model_files", ("client_A.pt", _model_bytes(1.0), "application/octet-stream")),
                ("model_files", ("client_B.pt", _model_bytes(2.0), "application/octet-stream")),
            ],
        )
        assert response.status_code == 200, response.text
        assert response.json() == {
            "ok": True,
            "round_id": "fl-sync-1",
            "received_client_count": 2,
        }
        assert (main.MODEL_LOCAL_DIR / "fl-sync-1" / "clients" / "client_A.pt").exists()


def test_batch_rejects_missing_metadata() -> None:
    with patched_model_dir():
        client = TestClient(main.app)
        response = client.post(
            "/clients/fl-model/batch",
            files=[
                ("model_files", ("client_A.pt", _model_bytes(1.0), "application/octet-stream")),
            ],
        )
        assert response.status_code == 400


def test_batch_rejects_expected_count_mismatch() -> None:
    with patched_model_dir():
        client = TestClient(main.app)
        response = client.post(
            "/clients/fl-model/batch",
            files=[
                ("metadata", (None, json.dumps(_metadata(expected_count=3)), "application/json")),
                ("model_files", ("client_A.pt", _model_bytes(1.0), "application/octet-stream")),
                ("model_files", ("client_B.pt", _model_bytes(2.0), "application/octet-stream")),
            ],
        )
        assert response.status_code == 409


def test_batch_rejects_invalid_model_file() -> None:
    with patched_model_dir():
        client = TestClient(main.app)
        metadata = {
            "scope": "all_clients",
            "round_id": "fl-sync-1",
            "expected_client_count": 1,
            "models": [
                {"client_id": "A", "sample_weight": 10, "filename": "client_A.pt"},
            ],
        }
        response = client.post(
            "/clients/fl-model/batch",
            files=[
                ("metadata", (None, json.dumps(metadata), "application/json")),
                ("model_files", ("client_A.pt", b"not-a-torch-file", "application/octet-stream")),
            ],
        )
        assert response.status_code == 422


def run() -> int:
    tests = [
        ("batch accepts metadata and model files", test_batch_accepts_metadata_and_model_files),
        ("batch rejects missing metadata", test_batch_rejects_missing_metadata),
        ("batch rejects expected count mismatch", test_batch_rejects_expected_count_mismatch),
        ("batch rejects invalid model file", test_batch_rejects_invalid_model_file),
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
