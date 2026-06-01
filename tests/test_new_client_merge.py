"""
Tests for incremental new-client clustering merge.

Covers:
  - assign_new_client() in src/fl/server_clustering.py  (pure algorithm)
  - BubbleServer.add_client() in src/fl/server.py        (assignment + warm start)

These tests use synthetic noisy feature-importance vectors, so they need no
client data, no model training, and no GPU. torch / flwr are stubbed because
BubbleServer.add_client never touches them (no training, no model saving here).

Run from the repo root:
    python3 tests/test_new_client_merge.py
"""

import os
import sys
import types

import numpy as np

# Make "src..." importable when run from the repo root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Stub heavy deps that server.py imports at module load but add_client never uses.
for _name in ("torch", "flwr"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)

from src.fl.server_clustering import assign_new_client  # noqa: E402
from src.fl.server import BubbleServer  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def importance_vector(strong_indices, dim=12, floor=0.05, peak=1.0):
    """Build a feature-importance vector that emphasises a few dimensions."""
    vec = np.full(dim, floor, dtype=np.float32)
    for i in strong_indices:
        vec[i] = peak
    return vec


class FakeClient:
    """
    Minimal stand-in for FedStockClient. Only the methods add_client touches
    are implemented, and weights are plain numpy arrays we can compare directly.
    """

    def __init__(self, importance, weights):
        self._importance = np.asarray(importance, dtype=np.float32)
        self._weights = [np.asarray(w, dtype=np.float32).copy() for w in weights]

    def extract_noisy_importance(self):
        return self._importance

    def get_parameters(self, config=None):
        return [w.copy() for w in self._weights]

    def set_parameters(self, parameters):
        self._weights = [np.asarray(p, dtype=np.float32).copy() for p in parameters]

    # In full-parameter mode shared == full for our purposes.
    def get_shared_parameters(self):
        return self.get_parameters()

    def set_shared_parameters(self, parameters):
        self.set_parameters(parameters)


def signature_weights(value, shape=(2, 2)):
    """A uniquely-valued weight tensor so we can tell clients apart."""
    return [np.full(shape, float(value), dtype=np.float32)]


# --------------------------------------------------------------------------- #
# Tiny assertion harness (no pytest dependency required)
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# 1. Pure assignment algorithm
# --------------------------------------------------------------------------- #

def test_assign_new_client():
    print("\n[1] assign_new_client (pure algorithm)")

    existing = {
        "A": importance_vector([0, 1]),
        "B": importance_vector([0, 1, 2]),     # bubble 0: front features
        "C": importance_vector([9, 10]),
        "D": importance_vector([9, 10, 11]),   # bubble 1: back features
        "E": importance_vector([5]),           # isolated: distinctive middle
    }
    bubbles = [["A", "B"], ["C", "D"]]
    isolated = ["E"]

    r = assign_new_client(importance_vector([0, 1]), existing, bubbles, isolated, new_client_id="N1")
    check("front-pattern client joins bubble 0",
          r["assigned_to"] == "bubble" and r["bubble_index"] == 0, r)
    check("joined bubble now contains the new client", "N1" in r["bubbles"][0], r["bubbles"])

    r = assign_new_client(importance_vector([9, 10, 11]), existing, bubbles, isolated, new_client_id="N2")
    check("back-pattern client joins bubble 1",
          r["assigned_to"] == "bubble" and r["bubble_index"] == 1, r)

    r = assign_new_client(importance_vector([5]), existing, bubbles, isolated, new_client_id="N3")
    check("client similar to lone E forms a new 2-client bubble",
          r["assigned_to"] == "new_bubble" and r["partner"] == "E", r)
    check("E removed from isolated after pairing", "E" not in r["isolated"], r["isolated"])
    check("new bubble holds both E and the new client",
          set(r["bubbles"][r["bubble_index"]]) == {"E", "N3"}, r["bubbles"])

    # A tight cluster -> small threshold -> a true outlier must stay isolated.
    tight = {
        "A": importance_vector([0, 1], peak=1.0),
        "B": importance_vector([0, 1], peak=1.02),
        "C": importance_vector([0, 1], peak=0.98),
    }
    r = assign_new_client(importance_vector([6, 7, 8]), tight, [["A", "B", "C"]], [], new_client_id="OUT")
    check("true outlier stays isolated",
          r["assigned_to"] == "isolated" and "OUT" in r["isolated"],
          f"dist={r['distance']}, thr={r['threshold']}")

    # Cold start: no existing clients -> isolated.
    r = assign_new_client(importance_vector([0, 1]), {}, [], [], new_client_id="FIRST")
    check("first-ever client is isolated (cold start)",
          r["assigned_to"] == "isolated" and r["isolated"] == ["FIRST"], r)


# --------------------------------------------------------------------------- #
# 2. BubbleServer.add_client (assignment + warm start)
# --------------------------------------------------------------------------- #

def build_server(tmp_output):
    clients = {
        "A": FakeClient(importance_vector([0, 1]), signature_weights(1)),
        "B": FakeClient(importance_vector([0, 1, 2]), signature_weights(2)),
        "C": FakeClient(importance_vector([9, 10]), signature_weights(3)),
        "D": FakeClient(importance_vector([9, 10, 11]), signature_weights(4)),
        "E": FakeClient(importance_vector([5]), signature_weights(5)),
    }
    server = BubbleServer(clients, output_dir=tmp_output)
    server.bubbles = [["A", "B"], ["C", "D"]]
    server.isolated = ["E"]
    server.noisy_importances = {cid: c.extract_noisy_importance() for cid, c in clients.items()}
    server.shared_lstm_weights = None  # full-parameter mode
    server.shared_global_weights = None
    return server, clients


def test_add_client(tmp_output):
    print("\n[2] BubbleServer.add_client (assignment + warm start)")

    # Join an existing bubble and warm-start from its representative (member A).
    server, clients = build_server(tmp_output)
    new = FakeClient(importance_vector([0, 1]), signature_weights(99))
    result = server.add_client("N1", new, save_results=False)
    check("new client joins bubble 0", result["assigned_to"] == "bubble" and result["bubble_index"] == 0, result)
    check("server state updated with new client", "N1" in server.bubbles[0], server.bubbles)
    rep = clients["A"].get_parameters()
    check("warm-started from existing bubble member (A)",
          np.allclose(new.get_parameters()[0], rep[0]),
          f"new={new.get_parameters()[0].flat[0]} rep={rep[0].flat[0]}")

    # Pair with lone isolated client E and warm-start from E.
    server, clients = build_server(tmp_output)
    new = FakeClient(importance_vector([5]), signature_weights(99))
    result = server.add_client("N3", new, save_results=False)
    check("client similar to E forms a new bubble", result["assigned_to"] == "new_bubble" and result["partner"] == "E", result)
    check("E is no longer isolated", "E" not in server.isolated, server.isolated)
    check("warm-started from E's weights",
          np.allclose(new.get_parameters()[0], clients["E"].get_parameters()[0]),
          f"new={new.get_parameters()[0].flat[0]}")

    # add_client also records the new client's importance vector.
    check("new client's importance stored on server", "N3" in server.noisy_importances, list(server.noisy_importances))


# --------------------------------------------------------------------------- #
# 3. get_cluster_report (frontend-facing report)
# --------------------------------------------------------------------------- #

FEATURE_NAMES = [
    "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_28",
    "is_weekend", "price_change_rate", "sell_price", "is_holiday",
    "week_of_year", "is_month_start", "is_month_end",
]


def test_get_cluster_report(tmp_output):
    print("\n[3] BubbleServer.get_cluster_report (frontend report)")

    server, _ = build_server(tmp_output)

    # A is in bubble 0 with B (front-features pattern: indices 0,1,2 strongest).
    report = server.get_cluster_report("A", feature_names=FEATURE_NAMES, top_n=3)

    check("report carries the client id", report["client_id"] == "A", report)
    check("report locates client in a cluster", report["cluster_id"] == 0 and not report["is_isolated"], report)
    check("cluster members include the bubble peers", set(report["cluster_members"]) == {"A", "B"}, report)
    check("my_top_features uses readable names",
          report["my_top_features"][0]["feature"] in FEATURE_NAMES, report["my_top_features"])
    check("front-pattern client's top feature is a front feature (lag_7/14/28)",
          report["my_top_features"][0]["feature"] in {"lag_7", "lag_14", "lag_28"},
          report["my_top_features"])
    check("cluster_common_features present", len(report["cluster_common_features"]) == 3, report)

    # Isolated client E reports as isolated.
    report_e = server.get_cluster_report("E", feature_names=FEATURE_NAMES)
    check("isolated client flagged is_isolated", report_e["is_isolated"] is True, report_e)
    check("isolated client cluster size is 1", report_e["cluster_size"] == 1, report_e)

    # Without feature_names, labels fall back to generic feature_{i}.
    report_generic = server.get_cluster_report("A")
    check("generic labels when feature_names omitted",
          report_generic["my_top_features"][0]["feature"].startswith("feature_"),
          report_generic["my_top_features"])


# --------------------------------------------------------------------------- #

def main():
    tmp_output = os.path.join(PROJECT_ROOT, "outputs", "_test_tmp")
    test_assign_new_client()
    test_add_client(tmp_output)
    test_get_cluster_report(tmp_output)

    print(f"\n==== {_PASSED} passed, {_FAILED} failed ====")
    sys.exit(1 if _FAILED else 0)


if __name__ == "__main__":
    main()
