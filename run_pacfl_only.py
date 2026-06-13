import argparse
import copy
import os
import sys

from run_fl_baselines import (
    Logger,
    _write_csv,
    _write_json,
    aggregate_metrics,
    create_run_dir,
    flatten_per_client_metrics,
    get_runtime_device,
    load_precomputed_feature_importances,
    save_client_feature_importance_artifacts,
    seed_everything,
    setup_client,
)
from src.dataset import CANDIDATE_FEATURE_COLS
from src.fl.extract_features import compute_anova_feature_selection, save_feature_selection
from src.fl.server import BubbleServer


def build_run_manifest(run_id, run_dir):
    # PA-CFL 전용 파이프라인이 어떤 산출물을 만들었는지 한눈에 추적할 수 있도록
    # 주요 파일 경로를 manifest로 남깁니다.
    files = {
        "config": "config.json",
        "feature_selection": "feature_selection.json",
        "split_summary": "split_summary.json",
        "feature_importances": "feature_importances.json",
        "feature_importances_by_client_csv": "feature_importances_by_client.csv",
        "feature_importance_rankings_json": "feature_importance_rankings.json",
        "clustering_results": "clustering_results.json",
        "metrics_history": "metrics_history.json",
        "per_client_metrics_json": "per_client_metrics.json",
        "per_client_metrics_csv": "per_client_metrics.csv",
        "final_results": "final_results.json",
        "evaluation_report": "evaluation_report.md",
        "models_pacfl": "models_pacfl/",
        "logs": "logs/",
        "run_manifest": "run_manifest.json",
    }
    return {
        "schema_version": 1,
        "run_id": run_id,
        "run_dir": run_dir,
        "files": {
            name: os.path.join(run_dir, relative_path)
            for name, relative_path in files.items()
        },
        "notes": [
            "This run executes only PA-CFL: feature selection, XGBoost feature importance extraction with DP noise, clustering, bubble FL, and personalization.",
            "Reported final metrics use held-out test sequences.",
            "models_pacfl stores the final personalized PA-CFL client models.",
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PA-CFL only: feature selection, noisy importance extraction, clustering, bubble FL, and personalization."
    )
    parser.add_argument(
        "--precomputed-feature-importances",
        default=None,
        help="Reuse a precomputed feature_importances.json instead of recomputing client-side XGBoost importances.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # 데이터 전처리부터 클러스터링, FL까지 전체 실행을 재현 가능하게 맞춥니다.
    seed_everything(seed=42)

    print("=== Starting PA-CFL Only Pipeline ===")
    print(f"=== Runtime Device: {get_runtime_device()} ===")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data", "clients")
    output_base_dir = os.path.join(current_dir, "outputs")
    run_id, run_dir = create_run_dir(output_base_dir)

    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(log_dir, "training_full.log"))

    print(f"Run outputs will be saved to {run_dir}")

    # 아래 하이퍼파라미터들이 이번 PA-CFL 실험 전체를 정의합니다.
    # 시퀀스 생성, feature selection, bubble FL, personalization 설정이 모두 포함됩니다.
    seq_len = 14
    train_ratio = 0.70
    val_ratio = 0.15
    num_rounds = 60
    epochs_per_round = 3
    global_warmup_rounds = 10
    head_finetune_epochs = 10
    recluster_interval = 10
    personalization_epochs = 40
    feature_top_k = 12
    feature_alpha = 0.10

    client_ids = []
    if os.path.exists(data_dir):
        for name in os.listdir(data_dir):
            client_dir = os.path.join(data_dir, name)
            if os.path.isdir(client_dir) and os.path.exists(os.path.join(client_dir, "train.csv")):
                client_ids.append(name)
    client_ids.sort()

    print(f"Found {len(client_ids)} clients in dataset.")
    if not client_ids:
        print("No clients found. Exiting.")
        return

    # 중간에 실행이 실패하더라도 어떤 설정으로 돌렸는지 남도록
    # config를 가장 먼저 저장합니다.
    config = {
        "run_id": run_id,
        "pipeline": "pacfl_only",
        "data_dir": data_dir,
        "output_dir": run_dir,
        "runtime_device": get_runtime_device(),
        "num_clients": len(client_ids),
        "seq_len": seq_len,
        "split_policy": "chronological split inside each item_id",
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1.0 - train_ratio - val_ratio,
        "reported_eval_split": "test",
        "x_scaler_fit_scope": "train rows only",
        "y_scaler_fit_scope": "train rows only",
        "feature_selection_scope": "train rows only",
        "feature_top_k": feature_top_k,
        "feature_alpha": feature_alpha,
        "num_rounds": num_rounds,
        "epochs_per_round": epochs_per_round,
        "global_warmup_rounds": global_warmup_rounds,
        "head_finetune_epochs": head_finetune_epochs,
        "recluster_interval": recluster_interval,
        "personalization_epochs": personalization_epochs,
        "precomputed_feature_importances": args.precomputed_feature_importances,
    }
    _write_json(config, os.path.join(run_dir, "config.json"))

    # 먼저 train split에 대해서만 ANOVA를 적용해 feature 공간을 줄입니다.
    # 여기서 고른 feature들은 이후 모든 client의 XGBoost 중요도 추출과
    # LSTM 학습에 공통으로 사용됩니다.
    feature_selection = compute_anova_feature_selection(
        clients=client_ids,
        data_dir=data_dir,
        candidate_features=CANDIDATE_FEATURE_COLS,
        top_k=feature_top_k,
        alpha=feature_alpha,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    save_feature_selection(feature_selection, os.path.join(run_dir, "feature_selection.json"))
    selected_features = feature_selection["selected_features"]
    config["selected_features"] = selected_features
    _write_json(config, os.path.join(run_dir, "config.json"))
    print(f"Selected {len(selected_features)} ANOVA features: {selected_features}")

    # client 객체를 store별로 하나씩 구성합니다.
    # 이 과정에서 시간순 분할, train-only scaling, item 단위 시퀀스 생성,
    # held-out test loader 구성이 함께 이루어집니다.
    clients_dict = {}
    for cid in client_ids:
        clients_dict[cid] = setup_client(
            cid,
            data_dir,
            seq_len=seq_len,
            selected_features=selected_features,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
    split_summary = {cid: client.split_stats for cid, client in clients_dict.items()}
    _write_json(split_summary, os.path.join(run_dir, "split_summary.json"))

    print("\n=== PA-CFL: Feature Importance, Clustering, FL, Personalization ===")
    # PA-CFL 경로가 독립적으로 모델 상태를 갖도록 client 객체를 복사해서 사용합니다.
    pacfl_clients = {cid: copy.deepcopy(client) for cid, client in clients_dict.items()}
    pacfl_server = BubbleServer(pacfl_clients, output_dir=run_dir)
    if args.precomputed_feature_importances:
        # 이미 다른 실행이나 다른 머신에서 importance를 만들어둔 경우
        # 그 결과를 그대로 불러와 클러스터링 신호로 재사용합니다.
        print(f"Loading precomputed feature importances from {args.precomputed_feature_importances}")
        precomputed_importances = load_precomputed_feature_importances(
            args.precomputed_feature_importances,
            client_ids,
        )
        pacfl_server.cluster_from_noisy_importances(
            noisy_importances=precomputed_importances,
            client_ids=client_ids,
            stage="initial_clustering_precomputed",
            round_num=0,
        )
    else:
        # 기본 PA-CFL 흐름입니다.
        # client 쪽에서 XGBoost 중요도를 뽑고 DP noise를 더한 뒤,
        # server가 이를 받아 클러스터링을 수행합니다.
        pacfl_server.step_1_collect_and_cluster()

    # 나중에 분석하거나 디버깅할 수 있도록 client별 중요도 순위를 저장합니다.
    save_client_feature_importance_artifacts(
        pacfl_server.noisy_importances,
        selected_features,
        run_dir,
    )

    # bubble FL 단계에서는 같은 cluster 안에서 LSTM backbone을 공유하고,
    # 마지막에는 client별 head personalization이 가능하도록 학습합니다.
    pacfl_fed_history = pacfl_server.step_3_federated_learning(
        num_rounds=num_rounds,
        epochs_per_round=epochs_per_round,
        global_warmup_rounds=global_warmup_rounds,
        head_finetune_epochs=head_finetune_epochs,
        personalize_head=True,
        recluster_interval=recluster_interval,
    )
    # clustered FL 이후 마지막으로 각 client가 자기 데이터에 맞게 추가 적응합니다.
    pacfl_pers_history = pacfl_server.step_4_personalized_learning(
        epochs=personalization_epochs
    )

    # head finetune 결과가 있으면 그 지표를 우선 사용하고,
    # 없으면 personalization 직전의 마지막 shared-LSTM round 지표를 사용합니다.
    pacfl_head_metrics = [record for record in pacfl_fed_history if record["stage"] == "head_finetune"]
    pacfl_bubble_metrics = pacfl_head_metrics or [
        record for record in pacfl_fed_history if record["round"] == num_rounds
    ]
    pacfl_final_metrics = pacfl_bubble_metrics + pacfl_pers_history
    pacfl_metrics = aggregate_metrics(pacfl_final_metrics)

    print(
        f"[PA-CFL] RMSE: {pacfl_metrics['rmse']:.4f}, "
        f"SMAPE: {pacfl_metrics['smape']:.4f}, MAE: {pacfl_metrics['mae']:.4f}, "
        f"WMAPE: {pacfl_metrics['wmape']:.4f}, MASE: {pacfl_metrics['mase']:.4f}"
    )

    # 비싼 FL 단계를 다시 돌리지 않아도 되도록 모델 가중치와 metric 결과를 함께 저장합니다.
    pacfl_server.save_models(output_dir=os.path.join(run_dir, "models_pacfl"))

    histories = {
        "PA-CFL": pacfl_fed_history + pacfl_pers_history,
    }
    per_client_metrics = flatten_per_client_metrics(histories)
    _write_json(
        {
            "results": {"PA-CFL": pacfl_metrics},
            "histories": histories,
        },
        os.path.join(run_dir, "metrics_history.json"),
    )
    _write_json(per_client_metrics, os.path.join(run_dir, "per_client_metrics.json"))
    _write_csv(per_client_metrics, os.path.join(run_dir, "per_client_metrics.csv"))
    _write_json(
        {
            "run_id": run_id,
            "results": {"PA-CFL": pacfl_metrics},
            "evaluation_split": "test",
            "metric_weight": "evaluation sequence count",
        },
        os.path.join(run_dir, "final_results.json"),
    )

    report_md = os.path.join(run_dir, "evaluation_report.md")
    with open(report_md, "w") as f:
        f.write("# PA-CFL Only Evaluation Report\n")
        f.write("## Overview\n")
        f.write(f"- **Run ID:** {run_id}\n")
        f.write(f"- **Total Clients:** {len(client_ids)}\n")
        f.write(f"- **Rounds:** {num_rounds}\n")
        f.write(f"- **Epochs per round:** {epochs_per_round}\n")
        f.write("- **Evaluation split:** held-out test split\n")
        f.write("- **Feature pipeline:** ANOVA feature selection, client-side XGBoost importance, DP noise, server-side clustering\n")
        f.write("- **Model pipeline:** bubble FL followed by personalization\n")
        f.write("\n## Result\n")
        f.write(
            f"- **PA-CFL**: RMSE = {pacfl_metrics['rmse']:.4f}, "
            f"SMAPE = {pacfl_metrics['smape']:.4f}, MAE = {pacfl_metrics['mae']:.4f}, "
            f"WMAPE = {pacfl_metrics['wmape']:.4f}, MASE = {pacfl_metrics['mase']:.4f}\n"
        )
    print(f"Evaluation report saved to {report_md}")

    _write_json(build_run_manifest(run_id, run_dir), os.path.join(run_dir, "run_manifest.json"))


if __name__ == "__main__":
    main()
