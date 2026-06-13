import argparse
import json
import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fl.privacy import get_noisy_feature_importance


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute noisy feature importance for one client payload."
    )
    parser.add_argument("--input-npz", required=True, help="Path to client payload .npz")
    parser.add_argument("--output-json", required=True, help="Path to output JSON file")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = np.load(args.input_npz)

    X_train = payload["X_train"].astype(np.float32)
    y_train = payload["y_train"].astype(np.float32)
    epsilon = float(payload["epsilon"][0])

    n_samples = X_train.shape[0]
    seq_len = X_train.shape[1]
    num_features = X_train.shape[2]
    X_flat = X_train.reshape(n_samples, -1)

    noisy_importance, sensitivity = get_noisy_feature_importance(
        X_flat,
        y_train,
        epsilon=epsilon,
        seq_len=seq_len,
        num_features=num_features,
    )

    with open(args.output_json, "w") as f:
        json.dump(
            {
                "importance": noisy_importance.tolist(),
                "sensitivity": float(sensitivity),
                "n_samples": int(n_samples),
                "seq_len": int(seq_len),
                "num_features": int(num_features),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
