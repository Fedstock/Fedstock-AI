import io
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import main  # noqa: E402


TOKEN = "test-token"
TEST_ROOT = PROJECT_ROOT / "outputs" / "_test_fl_model_api"
_PASSED = 0
_FAILED = 0


def _remove_test_root():
    if TEST_ROOT.is_symlink() or TEST_ROOT.is_file():
        TEST_ROOT.unlink()
    elif TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)


def check(name, condition, detail=""):
    global _PASSED, _FAILED
    if condition:
        _PASSED += 1
        print(f"  PASS  {name}")
    else:
        _FAILED += 1
        safe_detail = str(detail).encode("cp949", errors="backslashreplace").decode("cp949")
        print(f"  FAIL  {name}  {safe_detail}")


def _headers(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


def _model_bytes(value):
    buffer = io.BytesIO()
    torch.save({"weight": torch.tensor([float(value)], dtype=torch.float32)}, buffer)
    return buffer.getvalue()


def _load_response_model(response):
    return torch.load(io.BytesIO(response.content), map_location="cpu", weights_only=True)


def _is_full_lstm_state_dict(model):
    expected = {
        "lstm.weight_ih_l0": (128, 12),
        "lstm.weight_hh_l0": (128, 32),
        "lstm.bias_ih_l0": (128,),
        "lstm.bias_hh_l0": (128,),
        "fc.weight": (1, 32),
        "fc.bias": (1,),
    }
    return all(key in model and tuple(model[key].shape) == shape for key, shape in expected.items())


def _post_model(client, path, *, client_id="A", scope="single_client", round_id="round-1", sample_weight=1.0, value=1.0, filename=None, headers=None):
    filename = filename or f"client_{client_id}.pt"
    return client.post(
        path,
        headers=_headers() if headers is None else headers,
        data={
            "client_id": client_id,
            "scope": scope,
            "round_id": round_id,
            "sample_weight": str(sample_weight),
        },
        files={
            "model_file": (filename, _model_bytes(value), "application/octet-stream"),
        },
    )


@contextmanager
def _token_env(value=TOKEN):
    old = os.environ.get("API_BEARER_TOKEN")
    if value is None:
        os.environ.pop("API_BEARER_TOKEN", None)
    else:
        os.environ["API_BEARER_TOKEN"] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("API_BEARER_TOKEN", None)
        else:
            os.environ["API_BEARER_TOKEN"] = old


@contextmanager
def _patched_run_dir(cluster_members=None):
    cluster_members = cluster_members or ["A", "B"]
    _remove_test_root()
    (TEST_ROOT / "models" / "bubbles").mkdir(parents=True, exist_ok=True)
    (TEST_ROOT / "models" / "clients").mkdir(parents=True, exist_ok=True)
    bubble_path = TEST_ROOT / "models" / "bubbles" / "bubble_0.pt"
    bubble_path.write_bytes(_model_bytes(42.0))

    old_run_dir = main.RUN_DIR
    old_client_model_dir = main.CLIENT_MODEL_DIR
    old_load_cluster_state = main._load_cluster_state
    old_assignments = dict(getattr(main, "CLIENT_CLUSTER_ASSIGNMENTS", {}))

    main.RUN_DIR = TEST_ROOT
    main.CLIENT_MODEL_DIR = TEST_ROOT / "models" / "clients"
    main._load_cluster_state = lambda: ([cluster_members], [], {cid: 0 for cid in cluster_members})
    if hasattr(main, "CLIENT_CLUSTER_ASSIGNMENTS"):
        main.CLIENT_CLUSTER_ASSIGNMENTS.clear()
    if hasattr(main, "FL_MODEL_SYNC_QUEUES"):
        main.FL_MODEL_SYNC_QUEUES.clear()
    try:
        yield bubble_path
    finally:
        main.RUN_DIR = old_run_dir
        main.CLIENT_MODEL_DIR = old_client_model_dir
        main._load_cluster_state = old_load_cluster_state
        if hasattr(main, "CLIENT_CLUSTER_ASSIGNMENTS"):
            main.CLIENT_CLUSTER_ASSIGNMENTS.clear()
            main.CLIENT_CLUSTER_ASSIGNMENTS.update(old_assignments)
        if hasattr(main, "FL_MODEL_SYNC_QUEUES"):
            main.FL_MODEL_SYNC_QUEUES.clear()
        _remove_test_root()


def test_fl_model_requires_bearer_token():
    client = TestClient(main.app)
    with _token_env(TOKEN), _patched_run_dir():
        response = _post_model(client, "/ai/clients/A/fl-model", headers={})
    check("fl model missing bearer token returns 401", response.status_code == 401, response.text)


def test_fl_model_rejects_path_body_client_mismatch():
    client = TestClient(main.app)
    with _token_env(TOKEN), _patched_run_dir():
        response = _post_model(client, "/ai/clients/A/fl-model", client_id="B")
    check("path/body client mismatch returns 400", response.status_code == 400, response.text)


def test_fl_model_rejects_non_pt_upload():
    client = TestClient(main.app)
    with _token_env(TOKEN), _patched_run_dir():
        response = _post_model(client, "/ai/clients/A/fl-model", filename="client_A.txt")
    check("non-pt upload returns 400", response.status_code == 400, response.text)


def test_single_client_downloads_assigned_bubble_model():
    client = TestClient(main.app)
    with _token_env(TOKEN), _patched_run_dir():
        response = _post_model(client, "/ai/clients/A/fl-model", value=7.0)

    check("single client fl model request succeeds", response.status_code == 200, response.text)
    check("single client response is binary", response.headers.get("content-type") == "application/octet-stream", response.headers)
    check("single client model scope header is cluster", response.headers.get("x-fedstock-model-scope") == "cluster", response.headers)
    check("single client cluster id header is 0", response.headers.get("x-fedstock-cluster-id") == "0", response.headers)
    check("single client model format header is state_dict", response.headers.get("x-fedstock-model-format") == "pytorch_state_dict", response.headers)
    if response.status_code == 200:
        model = _load_response_model(response)
        check("single client body is full lstm state_dict", _is_full_lstm_state_dict(model), model.keys())
    else:
        check("single client body is full lstm state_dict", False, response.text)


def test_assigned_model_download_endpoint():
    client = TestClient(main.app)
    with _token_env(TOKEN), _patched_run_dir():
        response = client.get("/ai/clients/A/fl-model", headers=_headers())

    check("assigned model download succeeds", response.status_code == 200, response.text)
    check("assigned model download format header", response.headers.get("x-fedstock-model-format") == "pytorch_state_dict", response.headers)
    if response.status_code == 200:
        model = _load_response_model(response)
        check("assigned model download body is full lstm state_dict", _is_full_lstm_state_dict(model), model.keys())
    else:
        check("assigned model download body is full lstm state_dict", False, response.text)


def test_all_clients_waits_then_returns_weighted_average_model():
    client = TestClient(main.app)
    with _token_env(TOKEN), _patched_run_dir(cluster_members=["A", "B"]):
        first = _post_model(client, "/ai/clients/A/fl-model", scope="all_clients", round_id="sync-1", sample_weight=1, value=1)
        second = _post_model(client, "/ai/clients/B/fl-model", client_id="B", scope="all_clients", round_id="sync-1", sample_weight=3, value=3)

    check("first all-client upload waits with 409", first.status_code == 409, first.text)
    check("second all-client upload succeeds", second.status_code == 200, second.text)
    check("all-client response model scope is cluster", second.headers.get("x-fedstock-model-scope") == "cluster", second.headers)
    if second.status_code == 200:
        model = _load_response_model(second)
        check("all-client weighted average is returned", torch.allclose(model["weight"], torch.tensor([2.5])), model)
    else:
        check("all-client weighted average is returned", False, second.text)


def test_fl_model_compatibility_alias():
    client = TestClient(main.app)
    with _token_env(TOKEN), _patched_run_dir():
        response = _post_model(client, "/clients/A/fl-model", value=5.0)
    check("fl model compatibility alias succeeds", response.status_code == 200, response.text)


def main_test():
    test_fl_model_requires_bearer_token()
    test_fl_model_rejects_path_body_client_mismatch()
    test_fl_model_rejects_non_pt_upload()
    test_single_client_downloads_assigned_bubble_model()
    test_assigned_model_download_endpoint()
    test_all_clients_waits_then_returns_weighted_average_model()
    test_fl_model_compatibility_alias()
    print(f"\n==== {_PASSED} passed, {_FAILED} failed ====")
    sys.exit(1 if _FAILED else 0)


if __name__ == "__main__":
    main_test()
