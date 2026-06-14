import os
import sys
from contextlib import contextmanager

import numpy as np
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app import main  # noqa: E402


TOKEN = "test-token"
_PASSED = 0
_FAILED = 0


def check(name, condition, detail=""):
    global _PASSED, _FAILED
    if condition:
        _PASSED += 1
        print(f"  PASS  {name}")
    else:
        _FAILED += 1
        print(f"  FAIL  {name}  {detail}")


def _importance_vector(strong_indices, dim=None, floor=0.05, peak=1.0):
    dim = dim or len(main.SELECTED_FEATURES)
    vec = np.full(dim, floor, dtype=np.float32)
    for index in strong_indices:
        if index < dim:
            vec[index] = peak
    return vec


def _payload(client_id="N1", scope="single_client", round_id="round-1", expected_count=None, vector=None):
    values = vector if vector is not None else _importance_vector([0, 1])
    return {
        "scope": scope,
        "roundId": round_id,
        "clientId": client_id,
        "sampleCount": 128,
        "featureNames": list(main.SELECTED_FEATURES),
        "featureImportance": [float(value) for value in values],
        "expectedClientCount": expected_count,
    }


def _headers(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


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
def _patched_cluster_state():
    existing = {
        "A": _importance_vector([0, 1]),
        "B": _importance_vector([0, 1, 2]),
        "C": _importance_vector([9, 10]),
        "D": _importance_vector([9, 10, 11]),
    }
    old_load_importances = main._load_feature_importances
    old_load_cluster_state = main._load_cluster_state
    main._load_feature_importances = lambda: existing
    main._load_cluster_state = lambda: ([["A", "B"], ["C", "D"]], [], {"A": 0, "B": 0, "C": 1, "D": 1})
    try:
        yield
    finally:
        main._load_feature_importances = old_load_importances
        main._load_cluster_state = old_load_cluster_state


def _reset_round_queues():
    for name in ("CLUSTER_ASSIGNMENT_QUEUES", "CLUSTER_ASSIGNMENT_COMPLETED"):
        queue = getattr(main, name, None)
        if queue is not None:
            queue.clear()


def test_requires_bearer_token():
    client = TestClient(main.app)
    with _token_env(TOKEN):
        response = client.post("/ai/clients/cluster-assignment", json=_payload())
        check("missing bearer token returns 401", response.status_code == 401, response.text)

        response = client.post("/ai/clients/cluster-assignment", headers=_headers("wrong"), json=_payload())
        check("wrong bearer token returns 401", response.status_code == 401, response.text)


def test_missing_server_token_disables_authentication():
    client = TestClient(main.app)
    with _token_env(None):
        response = client.post("/ai/clients/cluster-assignment", json=_payload())
        check("missing API_BEARER_TOKEN allows request", response.status_code == 200, response.text)


def test_rejects_feature_length_mismatch():
    client = TestClient(main.app)
    body = _payload()
    body["featureImportance"] = body["featureImportance"][:-1]
    with _token_env(TOKEN):
        response = client.post("/ai/clients/cluster-assignment", headers=_headers(), json=body)
    check("feature length mismatch returns 400", response.status_code == 400, response.text)


def test_single_client_assignment():
    client = TestClient(main.app)
    with _token_env(TOKEN), _patched_cluster_state():
        response = client.post("/ai/clients/cluster-assignment", headers=_headers(), json=_payload())

    body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    check("single client request succeeds", response.status_code == 200, response.text)
    check("single client status assigned", body.get("status") == "assigned", body)
    check("single client assigned to bubble", body.get("assignedTo") == "bubble", body)
    check("single client cluster id is 0", body.get("clusterId") == 0, body)
    check("single client cluster members include new client", "N1" in body.get("clusterMembers", []), body)
    check("single client queue completed", body.get("queueStatus") == "completed", body)
    check("single client response includes model download url", body.get("modelDownloadUrl") == "/ai/clients/N1/fl-model", body)
    check("single client response model format", body.get("modelFormat") == "pytorch_state_dict", body)


def test_all_clients_queue_and_complete():
    client = TestClient(main.app)
    _reset_round_queues()
    with _token_env(TOKEN):
        first = client.post(
            "/ai/clients/cluster-assignment",
            headers=_headers(),
            json=_payload(client_id="Q1", scope="all_clients", round_id="round-q", expected_count=2),
        )
        second = client.post(
            "/ai/clients/cluster-assignment",
            headers=_headers(),
            json=_payload(client_id="Q2", scope="all_clients", round_id="round-q", expected_count=2),
        )

    first_body = first.json() if first.headers.get("content-type", "").startswith("application/json") else {}
    second_body = second.json() if second.headers.get("content-type", "").startswith("application/json") else {}
    check("first all-clients request succeeds", first.status_code == 200, first.text)
    check("first all-clients request is queued", first_body.get("status") == "queued", first_body)
    check("first all-clients queue waiting", first_body.get("queueStatus") == "waiting", first_body)
    check("second all-clients request succeeds", second.status_code == 200, second.text)
    check("second all-clients request is assigned", second_body.get("status") == "assigned", second_body)
    check("second all-clients queue completed", second_body.get("queueStatus") == "completed", second_body)
    check("second all-clients members include both clients", set(second_body.get("clusterMembers", [])) == {"Q1", "Q2"}, second_body)


def test_compatibility_endpoint_alias():
    client = TestClient(main.app)
    with _token_env(TOKEN), _patched_cluster_state():
        response = client.post("/clients/cluster-assignment", headers=_headers(), json=_payload(client_id="N2"))
    check("compatibility endpoint alias succeeds", response.status_code == 200, response.text)


def main_test():
    test_requires_bearer_token()
    test_missing_server_token_disables_authentication()
    test_rejects_feature_length_mismatch()
    test_single_client_assignment()
    test_all_clients_queue_and_complete()
    test_compatibility_endpoint_alias()
    print(f"\n==== {_PASSED} passed, {_FAILED} failed ====")
    sys.exit(1 if _FAILED else 0)


if __name__ == "__main__":
    main_test()
