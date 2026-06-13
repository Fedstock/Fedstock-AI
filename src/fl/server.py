import numpy as np
import flwr as fl
import torch
import os
import json
from concurrent.futures import ThreadPoolExecutor
from src.fl.server_clustering import perform_clustering, assign_new_client

def _emit(message, logger=None):
    if logger:
        logger.info(message)
    else:
        print(message)


class BubbleServer:
    """
    Simulates the central server orchestrating PA-CFL.
    """
    def __init__(self, clients_dict, output_dir="outputs"):
        """
        clients_dict: dict of {client_id: FedStockClient instance}
        """
        self.clients = clients_dict
        self.output_dir = output_dir
        self.bubbles = []
        self.isolated = []
        self.shared_global_weights = None
        self.shared_lstm_weights = None
        self.clustering_history = []
        self.ema_alpha = 0.8  # Decay factor for EMA
        self.ema_dict = {}    # To store EMA of normalized distributions for clustering
        self.noisy_importances = {}  # Store initial noisy feature importances for re-clustering

    @staticmethod
    def _copy_parameters(parameters):
        return [param.copy() for param in parameters]

    @staticmethod
    def _fedavg(round_weights, round_samples):
        total_samples = sum(round_samples)
        aggregated_weights = []
        for layer_idx in range(len(round_weights[0])):
            layer_weighted_sum = sum(
                round_weights[c_idx][layer_idx] * (round_samples[c_idx] / total_samples)
                for c_idx in range(len(round_weights))
            )
            aggregated_weights.append(layer_weighted_sum)
        return aggregated_weights

    @staticmethod
    def _flatten_delta(local_weights, reference_weights):
        delta = np.concatenate(
            [
                (local - reference).reshape(-1)
                for local, reference in zip(local_weights, reference_weights)
            ]
        )
        return delta

    def _cluster_update_vectors(self, update_vectors, current_round, max_clusters=8):
        client_ids = list(update_vectors.keys())
        vectors = np.array([update_vectors[cid] for cid in client_ids])
        
        # Get active noisy feature importances for weighted concatenation
        active_features = []
        for cid in client_ids:
            active_features.append(self.noisy_importances.get(cid, np.zeros(12, dtype=np.float32)))
        active_features = np.array(active_features, dtype=np.float32)
        
        # Decaying feature weight (gamma) scheduling
        # Start at 0.9, decay by 0.008 per round, min value 0.5
        gamma_start = 0.9
        gamma_min = 0.5
        decay_rate = 0.008
        feature_weight = max(gamma_min, gamma_start - decay_rate * current_round)
        
        labels, k_star, _, isolated = perform_clustering(
            vectors,
            max_clusters=max_clusters,
            complexity_penalty=0.15,
            singleton_penalty=0.20,
            ema_dict=self.ema_dict,
            client_ids=client_ids,
            ema_alpha=self.ema_alpha,
            metric='cosine',
            feature_importances=active_features,
            feature_weight=feature_weight,
        )
        bubble_dict = {}
        for idx, label in enumerate(labels):
            bubble_dict.setdefault(int(label), []).append(client_ids[idx])
        return list(bubble_dict.values()), [client_ids[i] for i in isolated], int(k_star)

    def _aggregate_client_weights(self, client_ids, client_weights, client_samples):
        return self._fedavg(
            [client_weights[cid] for cid in client_ids],
            [client_samples[cid] for cid in client_ids],
        )

    def _build_reclustered_common_weights(
        self,
        active_bubbles,
        latest_client_weights,
        latest_client_samples,
        common_global_weights,
    ):
        reclustered_weights = {}
        for b_idx, bubble_cids in enumerate(active_bubbles):
            if len(bubble_cids) == 1:
                reclustered_weights[b_idx] = self._copy_parameters(common_global_weights)
            else:
                reclustered_weights[b_idx] = self._aggregate_client_weights(
                    bubble_cids,
                    latest_client_weights,
                    latest_client_samples,
                )
        return reclustered_weights

    def _build_shared_global_weights(self, global_warmup_rounds, epochs_per_round, logger=None):
        # Exclude isolated clients from global warmup — they only receive the
        # final global model later in step_4.
        non_isolated_ids = [
            cid for cid in self.clients.keys() if cid not in self.isolated
        ]
        if not non_isolated_ids:
            return None

        global_weights = self.clients[non_isolated_ids[0]].get_parameters({})
        _emit(
            f"[Shared Global] warmup_rounds={global_warmup_rounds}, clients={non_isolated_ids} (excluding {len(self.isolated)} isolated)",
            logger,
        )
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            for warmup_round in range(1, global_warmup_rounds + 1):
                futures = []
                for cid in non_isolated_ids:
                    futures.append(executor.submit(
                        self.clients[cid].fit,
                        parameters=global_weights,
                        config={
                            "epochs": epochs_per_round,
                            "current_round": warmup_round,
                            "total_rounds": global_warmup_rounds,
                        }
                    ))
                
                round_weights = []
                round_samples = []
                for future in futures:
                    updated_weights, num_samples, _ = future.result()
                    round_weights.append(updated_weights)
                    round_samples.append(num_samples)
                    
                global_weights = self._fedavg(round_weights, round_samples)
                _emit(f"[Shared Global] warmup round {warmup_round}/{global_warmup_rounds} complete", logger)
        return self._copy_parameters(global_weights)
        
    def load_clustering_results(self, clustering_json_path, logger=None):
        """
        Loads pre-computed cluster assignments from JSON (generated by server_clustering.py).
        """
        import json
        _emit(f"Loading clustering results from {clustering_json_path}...", logger)
        with open(clustering_json_path, 'r') as f:
            data = json.load(f)

        if "records" in data:
            self.clustering_history = data["records"]
            latest_record = self.clustering_history[-1] if self.clustering_history else {}
            record_bubbles = latest_record.get("multi_client_bubbles", [])
            if record_bubbles and any("clients" not in bubble for bubble in record_bubbles):
                raise ValueError(
                    "Cannot load compact clustering history without per-bubble client membership. "
                    "Use a clustering file that includes assignments or clients."
                )
            self.bubbles = [list(bubble["clients"]) for bubble in record_bubbles]
            self.isolated = list(latest_record.get("isolated_clients", []))
        elif "multi_client_bubbles" in data:
            self.clustering_history = [data]
            record_bubbles = data.get("multi_client_bubbles", [])
            if record_bubbles and any("clients" not in bubble for bubble in record_bubbles):
                raise ValueError(
                    "Cannot load compact clustering history without per-bubble client membership. "
                    "Use a clustering file that includes assignments or clients."
                )
            self.bubbles = [list(bubble["clients"]) for bubble in record_bubbles]
            self.isolated = list(data.get("isolated_clients", []))
        else:
            assignments = data["assignments"]
            self.isolated = data["isolated_clients"]

            # Group into bubbles
            bubble_dict = {}
            for cid, cluster_id in assignments.items():
                if cid in self.isolated:
                    continue
                if cluster_id not in bubble_dict:
                    bubble_dict[cluster_id] = []
                bubble_dict[cluster_id].append(cid)

            self.bubbles = list(bubble_dict.values())

        _emit(f"Multi-Client Bubbles: {self.bubbles}", logger)
        _emit(f"Isolated (Single-Client) Bubbles: {self.isolated}", logger)
        
    def step_1_collect_and_cluster(self):
        """
        Collects noisy feature importances and performs agglomerative clustering.
        """
        print("--- PA-CFL Step 1 & 2: Feature Collection & Clustering ---")
        noisy_importances = []
        client_ids = list(self.clients.keys())
        
        for cid in client_ids:
            print(f"Extracting noisy feature importance for {cid}...")
            client = self.clients[cid]
            noisy_imp = client.extract_noisy_importance()
            noisy_importances.append(noisy_imp)
            
        importance_dict = {cid: imp.tolist() for cid, imp in zip(client_ids, noisy_importances)}
        self.noisy_importances = {cid: np.array(imp, dtype=np.float32) for cid, imp in zip(client_ids, noisy_importances)}
        os.makedirs(self.output_dir, exist_ok=True)
        importance_path = os.path.join(self.output_dir, "feature_importances.json")
        with open(importance_path, 'w') as f:
            json.dump(importance_dict, f, indent=4)
        print(f"Feature importances saved to {importance_path}")

        self.cluster_from_noisy_importances(
            noisy_importances=noisy_importances,
            client_ids=client_ids,
            stage="initial_clustering",
            round_num=0,
        )

    def cluster_from_noisy_importances(
        self,
        noisy_importances,
        client_ids=None,
        stage="initial_clustering",
        round_num=0,
    ):
        """
        Cluster clients from precomputed noisy feature importances.

        noisy_importances may be a dict keyed by client id or a sequence aligned
        with client_ids. This supports offline feature extraction on another
        machine, followed by clustering/training in the current environment.
        """
        if client_ids is None:
            client_ids = list(self.clients.keys())

        if isinstance(noisy_importances, dict):
            missing = [cid for cid in client_ids if cid not in noisy_importances]
            extra = [cid for cid in noisy_importances.keys() if cid not in self.clients]
            if missing:
                raise ValueError(f"Missing precomputed feature importances for clients: {missing}")
            if extra:
                print(f"Ignoring precomputed feature importances for unknown clients: {sorted(extra)}")
            ordered_vectors = [
                np.asarray(noisy_importances[cid], dtype=np.float32)
                for cid in client_ids
            ]
            self.noisy_importances = {
                cid: np.asarray(noisy_importances[cid], dtype=np.float32)
                for cid in client_ids
            }
        else:
            ordered_vectors = [np.asarray(vector, dtype=np.float32) for vector in noisy_importances]
            if len(ordered_vectors) != len(client_ids):
                raise ValueError(
                    "Length mismatch between precomputed feature importances and client_ids."
                )
            self.noisy_importances = {
                cid: np.asarray(vector, dtype=np.float32)
                for cid, vector in zip(client_ids, ordered_vectors)
            }

        os.makedirs(self.output_dir, exist_ok=True)
        importance_path = os.path.join(self.output_dir, "feature_importances.json")
        with open(importance_path, "w") as f:
            json.dump(
                {cid: vector.tolist() for cid, vector in self.noisy_importances.items()},
                f,
                indent=4,
            )
        print(f"Feature importances saved to {importance_path}")

        noisy_importances = np.array(ordered_vectors)
        labels, k_star, _, _ = perform_clustering(
            noisy_importances,
            max_clusters=8,
            complexity_penalty=0.15,
            singleton_penalty=0.20,
            ema_dict=self.ema_dict,
            client_ids=client_ids,
            ema_alpha=self.ema_alpha,
            metric='cosine',
        )
        
        print(f"Optimal Clusters (k*): {k_star}")
        
        # Group client IDs into bubbles
        bubble_groups = {}
        for idx, label in enumerate(labels):
            if label not in bubble_groups:
                bubble_groups[label] = []
            bubble_groups[label].append(client_ids[idx])
            
        # Separate into multi-client and single-client
        self.bubbles = []
        self.isolated = []
        for label, cids in bubble_groups.items():
            if len(cids) > 1:
                self.bubbles.append(cids)
            else:
                self.isolated.extend(cids)
                
        print(f"Multi-Client Bubbles: {self.bubbles}")
        print(f"Isolated (Single-Client) Bubbles: {self.isolated}")
        
        # Save clustering results to outputs folder
        self.save_clustering_results(
            os.path.join(self.output_dir, "clustering_results.json"),
            stage=stage,
            round_num=round_num,
            k_star=k_star,
            reset_history=True,
        )

    def add_client(
        self,
        cid,
        client,
        metric='cosine',
        warm_start=True,
        train_epochs=0,
        save_results=True,
        logger=None,
    ):
        """
        Incrementally onboard a single new client (e.g. a store that just
        uploaded its data) into the existing clustering, without re-running the
        full batch clustering over every client.

        Steps:
        1. Register the client and extract its noisy feature importance.
        2. Assign it to the nearest existing bubble — or pair it with a lone
           isolated client, or keep it isolated — via assign_new_client.
        3. Warm-start its model from the joined bubble's already-trained
           weights so it does not learn from scratch.
        4. Optionally fine-tune it locally and persist an updated clustering
           record.

        Returns the assign_new_client result dict (with updated bubbles/isolated).
        """
        self.clients[cid] = client

        importance = np.array(client.extract_noisy_importance(), dtype=np.float32)
        self.noisy_importances[cid] = importance

        existing_importances = {
            other: imp for other, imp in self.noisy_importances.items() if other != cid
        }
        result = assign_new_client(
            importance,
            existing_importances,
            self.bubbles,
            self.isolated,
            metric=metric,
            ema_dict=self.ema_dict,
            new_client_id=cid,
        )
        self.bubbles = result["bubbles"]
        self.isolated = result["isolated"]

        detail = ""
        if result["bubble_index"] is not None:
            detail += f" (bubble {result['bubble_index']})"
        if result["partner"]:
            detail += f", paired with {result['partner']}"
        if result["distance"] is not None and result["threshold"] is not None:
            detail += f", distance={result['distance']:.4f}, threshold={result['threshold']:.4f}"
        _emit(f"[Add Client] {cid} -> {result['assigned_to']}{detail}", logger)

        if warm_start:
            warm_weights = self._warm_start_weights_for(cid, result)
            if warm_weights is not None:
                if self.shared_lstm_weights is not None:
                    client.set_shared_parameters(self._copy_parameters(warm_weights))
                else:
                    client.set_parameters(self._copy_parameters(warm_weights))
                _emit(f"[Add Client] {cid} warm-started from existing weights.", logger)

        if train_epochs > 0:
            if self.shared_lstm_weights is not None:
                client.fit_shared_lstm(
                    parameters=client.get_shared_parameters(),
                    config={"epochs": train_epochs, "current_round": 1, "total_rounds": 1},
                )
            else:
                client.fit(
                    parameters=client.get_parameters({}),
                    config={"epochs": train_epochs, "current_round": 1, "total_rounds": 1},
                )
            _emit(f"[Add Client] {cid} locally trained for {train_epochs} epoch(s).", logger)

        if save_results:
            self.save_clustering_results(
                os.path.join(self.output_dir, "clustering_results.json"),
                stage="incremental_merge",
                round_num=None,
            )

        return result

    def _warm_start_weights_for(self, new_cid, result):
        """
        Pick warm-start weights for a newly assigned client.

        For a bubble (or a freshly formed two-client bubble) we reuse an
        existing member's already-synchronized weights, mirroring how
        save_models treats the first bubble member as the shared representative.
        Isolated clients fall back to the shared backbone / global weights.
        """
        bubble_index = result.get("bubble_index")
        if bubble_index is not None and 0 <= bubble_index < len(self.bubbles):
            for member in self.bubbles[bubble_index]:
                if member != new_cid and member in self.clients:
                    representative = self.clients[member]
                    if self.shared_lstm_weights is not None:
                        return representative.get_shared_parameters()
                    return representative.get_parameters({})

        if self.shared_lstm_weights is not None:
            return self.shared_lstm_weights
        return self.shared_global_weights

    def get_cluster_report(self, cid, feature_names=None, top_n=3):
        """
        Build the "similar-store comparison report" for a single client so the
        frontend can render it directly.

        Returns the client's own dominant features (its "store type"), the
        cluster it belongs to, the other members of that cluster, and the
        cluster's shared dominant features.

        feature_names : optional list mapping importance-vector indices to
            human-readable names (e.g. CANDIDATE_FEATURE_COLS). When omitted,
            generic "feature_{i}" labels are returned so callers can still map
            them later.
        """
        if cid not in self.noisy_importances:
            raise KeyError(f"No feature importance recorded for client: {cid}")

        def _label(idx):
            if feature_names is not None and idx < len(feature_names):
                return feature_names[idx]
            return f"feature_{idx}"

        def _rank(vector, value_key):
            vector = np.asarray(vector, dtype=np.float32).reshape(-1)
            order = np.argsort(vector)[::-1][:top_n]
            return [
                {"feature": _label(int(i)), value_key: round(float(vector[i]), 4)}
                for i in order
            ]

        # Locate the client: a multi-client bubble, or the isolated list.
        cluster_id = None
        members = [cid]
        is_isolated = True
        for b_idx, bubble in enumerate(self.bubbles):
            if cid in bubble:
                cluster_id = b_idx
                members = list(bubble)
                is_isolated = False
                break
        if is_isolated and cid in self.isolated:
            # Isolated clients each form their own cluster after the bubbles.
            cluster_id = len(self.bubbles) + self.isolated.index(cid)

        # Cluster-level shared importance = mean over members we have vectors for.
        member_vectors = [
            np.asarray(self.noisy_importances[m], dtype=np.float32).reshape(-1)
            for m in members
            if m in self.noisy_importances
        ]
        common_features = []
        if member_vectors:
            mean_vector = np.mean(member_vectors, axis=0)
            common_features = _rank(mean_vector, "avg_importance")

        return {
            "client_id": cid,
            "cluster_id": cluster_id,
            "is_isolated": is_isolated,
            "my_top_features": _rank(self.noisy_importances[cid], "importance"),
            "cluster_members": members,
            "cluster_size": len(members),
            "cluster_common_features": common_features,
        }

    def _build_clustering_record(
        self,
        stage="current",
        round_num=None,
        k_star=None,
        bubbles=None,
        isolated=None,
    ):
        active_bubbles = [list(bubble) for bubble in (bubbles if bubbles is not None else self.bubbles)]
        isolated_clients = list(isolated if isolated is not None else self.isolated)
        multi_client_bubbles = [bubble for bubble in active_bubbles if len(bubble) > 1]
        singleton_clients = [
            bubble[0]
            for bubble in active_bubbles
            if len(bubble) == 1 and bubble[0] not in isolated_clients
        ]
        isolated_clients = list(dict.fromkeys(isolated_clients + singleton_clients))

        all_cluster_sizes = [len(bubble) for bubble in multi_client_bubbles] + [1 for _ in isolated_clients]
        total_clients = sum(all_cluster_sizes)
        cluster_size_stats = {
            "min": int(min(all_cluster_sizes)) if all_cluster_sizes else 0,
            "max": int(max(all_cluster_sizes)) if all_cluster_sizes else 0,
            "mean": float(np.mean(all_cluster_sizes)) if all_cluster_sizes else 0.0,
        }

        assignments = {}
        for idx, bubble in enumerate(multi_client_bubbles):
            for cid in bubble:
                assignments[cid] = idx
        isolated_offset = len(multi_client_bubbles)
        for idx, cid in enumerate(isolated_clients):
            assignments[cid] = isolated_offset + idx

        return {
            "sequence": len(self.clustering_history) + 1,
            "stage": stage,
            "round": round_num,
            "k_star": int(k_star if k_star is not None else len(all_cluster_sizes)),
            "total_clients": int(total_clients),
            "num_clusters": len(all_cluster_sizes),
            "num_multi_client_bubbles": len(multi_client_bubbles),
            "num_isolated_clients": len(isolated_clients),
            "cluster_sizes": all_cluster_sizes,
            "cluster_size_stats": cluster_size_stats,
            "multi_client_bubbles": [
                {
                    "bubble_id": idx,
                    "size": len(bubble),
                    "clients": list(bubble),
                }
                for idx, bubble in enumerate(multi_client_bubbles)
            ],
            "isolated_clients": isolated_clients,
            "assignments": assignments,
        }

    def save_clustering_results(
        self,
        output_path,
        stage="current",
        round_num=None,
        k_star=None,
        bubbles=None,
        isolated=None,
        reset_history=False,
    ):
        """
        Saves clustering history to a JSON file.

        Full client membership is included so each run can be audited and
        reproduced without relying on interleaved console logs.
        """
        import json
        import os

        if reset_history:
            self.clustering_history = []

        record = self._build_clustering_record(
            stage=stage,
            round_num=round_num,
            k_star=k_star,
            bubbles=bubbles,
            isolated=isolated,
        )
        self.clustering_history.append(record)

        result = {
            "schema_version": 3,
            "description": "Chronological clustering records with per-bubble clients and client-to-cluster assignments.",
            "records": self.clustering_history,
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"Clustering results saved to {output_path}")

    def save_models(self, output_dir="outputs/models"):
        """
        Saves all client models and bubble weights to the specified directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving all models to {output_dir}...")

        
        # 1. Save individual client models
        client_dir = os.path.join(output_dir, "clients")
        os.makedirs(client_dir, exist_ok=True)
        for cid, client in self.clients.items():
            model_path = os.path.join(client_dir, f"client_{cid}.pt")
            torch.save(client.model.state_dict(), model_path)
            
        # 2. Save bubble/global weights if available
        if self.bubbles:
            bubble_dir = os.path.join(output_dir, "bubbles")
            os.makedirs(bubble_dir, exist_ok=True)
            for b_idx, bubble_cids in enumerate(self.bubbles):
                # We use the weights of the first client in the bubble as a representative of shared weights
                # (since they were just synchronized)
                rep_client = self.clients[bubble_cids[0]]
                bubble_path = os.path.join(bubble_dir, f"bubble_{b_idx}.pt")
                # Only save shared layers (LSTM) for bubbles
                shared_state = {k: v for k, v in rep_client.model.state_dict().items() if k.startswith(("lstm.",))}
                torch.save(shared_state, bubble_path)
                
        print(f"All model weights saved successfully.")


    def step_3_federated_learning(
        self,
        num_rounds=3,
        epochs_per_round=5,
        logger=None,
        global_warmup_rounds=0,
        head_finetune_epochs=1,
        personalize_head=False,
        recluster_interval=10,
    ):
        """
        Performs FedAvg within each multi-client bubble.
        """
        _emit("\n--- PA-CFL Step 3: Federated Learning within Bubbles ---", logger)
        history = []
        self.shared_global_weights = self._build_shared_global_weights(
            global_warmup_rounds=global_warmup_rounds,
            epochs_per_round=epochs_per_round,
            logger=logger,
        )
        first_client = next(iter(self.clients.values()), None)
        self.shared_lstm_weights = (
            first_client.select_parameters(self.shared_global_weights, ("lstm.",))
            if personalize_head and first_client is not None
            else None
        )
        if personalize_head:
            return self._step_3_personalized_dynamic(
                num_rounds=num_rounds,
                epochs_per_round=epochs_per_round,
                head_finetune_epochs=head_finetune_epochs,
                recluster_interval=recluster_interval,
                logger=logger,
            )

        for b_idx, bubble_cids in enumerate(self.bubbles):
            _emit(f"\n[Bubble {b_idx}] Starting FL with clients: {bubble_cids}", logger)

            if personalize_head:
                # Every bubble shares the same LSTM backbone; each client keeps its own output head.
                global_weights = self._copy_parameters(self.shared_lstm_weights)
            else:
                global_weights = self._copy_parameters(self.shared_global_weights)

            for fl_round in range(1, num_rounds + 1):
                _emit(f"  Round {fl_round}/{num_rounds}", logger)
                
                # 1. Distribute global weights and Train locally
                round_weights = []
                round_samples = []
                
                with ThreadPoolExecutor(max_workers=1) as executor:
                    futures = []
                    for cid in bubble_cids:
                        client = self.clients[cid]
                        if personalize_head:
                            futures.append(executor.submit(
                                client.fit_shared_lstm,
                                parameters=global_weights,
                                config={
                                    "epochs": epochs_per_round,
                                    "current_round": fl_round,
                                    "total_rounds": num_rounds,
                                }
                            ))
                        else:
                            futures.append(executor.submit(
                                client.fit,
                                parameters=global_weights,
                                config={
                                    "epochs": epochs_per_round,
                                    "current_round": fl_round,
                                    "total_rounds": num_rounds,
                                }
                            ))
                    
                    for future in futures:
                        updated_weights, num_samples, _ = future.result()
                        round_weights.append(updated_weights)
                        round_samples.append(num_samples)
                    
                # 2. Aggregate using FedAvg
                total_train_samples = sum(round_samples)
                global_weights = self._fedavg(round_weights, round_samples)
                
                # 3. Evaluate on clients (can also be parallelized)
                total_metrics = {}
                total_train_samples = sum(round_samples)
                total_eval_samples = 0
                per_client_metrics = []
                
                with ThreadPoolExecutor(max_workers=1) as executor:
                    eval_futures = []
                    for cid in bubble_cids:
                        client = self.clients[cid]
                        if personalize_head:
                            eval_futures.append(executor.submit(client.evaluate_shared_lstm, parameters=global_weights, config={}))
                        else:
                            eval_futures.append(executor.submit(client.evaluate, parameters=global_weights, config={}))
                    
                    for idx, future in enumerate(eval_futures):
                        cid = bubble_cids[idx]
                        _, eval_samples, metrics = future.result()
                        total_eval_samples += eval_samples
                        
                        client_record = {
                            "client": cid,
                            "num_samples": int(eval_samples),
                            "train_samples": int(round_samples[idx]),
                        }
                        for k, v in metrics.items():
                            client_record[k] = float(v)
                            total_metrics[k] = total_metrics.get(k, 0.0) + float(v) * eval_samples
                        per_client_metrics.append(client_record)
                    
                avg_metrics = {k: v / total_eval_samples for k, v in total_metrics.items()}
                
                history_record = {
                    "stage": "federated",
                    "bubble": b_idx,
                    "clients": bubble_cids,
                    "round": fl_round,
                    "num_samples": int(total_eval_samples),
                    "train_samples": int(total_train_samples),
                    "per_client_metrics": per_client_metrics,
                }
                history_record.update(avg_metrics)
                history.append(history_record)
                _emit(
                    f"    -> Bubble {b_idx} Global metrics: RMSE={avg_metrics.get('rmse', 0):.4f}, SMAPE={avg_metrics.get('smape', 0):.4f}",
                    logger,
                )

            if personalize_head and head_finetune_epochs > 0:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    finetune_futures = []
                    for cid in bubble_cids:
                        client = self.clients[cid]
                        finetune_futures.append(executor.submit(
                            client.fit_head,
                            parameters=global_weights,
                            config={
                                "epochs": head_finetune_epochs,
                                "current_round": num_rounds,
                                "total_rounds": num_rounds,
                            }
                        ))
                    
                    for idx, future in enumerate(finetune_futures):
                        updated_weights, train_samples, _ = future.result()
                        cid = bubble_cids[idx]
                        _, eval_samples, metrics = self.clients[cid].evaluate(parameters=updated_weights, config={})
                        record = {
                            "stage": "head_finetune",
                            "bubble": b_idx,
                            "client": cid,
                            "epochs": head_finetune_epochs,
                            "num_samples": int(eval_samples),
                            "train_samples": int(train_samples),
                        }
                        record.update({k: float(v) for k, v in metrics.items()})
                        history.append(record)
                        _emit(
                            f"    -> Client {cid} personalized head metrics: RMSE={metrics.get('rmse', 0):.4f}, SMAPE={metrics.get('smape', 0):.4f}",
                            logger,
                        )

        _emit("\nFL Training Complete.", logger)
        return history

    def _step_3_personalized_dynamic(
        self,
        num_rounds,
        epochs_per_round,
        head_finetune_epochs,
        recluster_interval,
        logger=None,
    ):
        history = []
        # Isolated clients are completely excluded from FL rounds and
        # reclustering.  They only receive the global model in step_4.
        preserved_isolated = list(self.isolated)
        active_bubbles = [list(bubble) for bubble in self.bubbles]
        common_weights = {
            idx: self._copy_parameters(self.shared_lstm_weights)
            for idx in range(len(active_bubbles))
        }
        reference_weights = self._copy_parameters(self.shared_lstm_weights)

        for fl_round in range(1, num_rounds + 1):
            _emit(f"  Personalized shared-LSTM round {fl_round}/{num_rounds}", logger)
            next_common_weights = {}
            latest_update_vectors = {}
            latest_client_weights = {}
            latest_client_samples = {}

            for b_idx, bubble_cids in enumerate(active_bubbles):
                global_weights = common_weights[b_idx]
                round_weights = []
                round_samples = []
                
                with ThreadPoolExecutor(max_workers=1) as executor:
                    futures = []
                    for cid in bubble_cids:
                        client = self.clients[cid]
                        futures.append(executor.submit(
                            client.fit_shared_lstm,
                            parameters=global_weights,
                            config={
                                "epochs": epochs_per_round,
                                "current_round": fl_round,
                                "total_rounds": num_rounds,
                            }
                        ))
                    
                    for idx, future in enumerate(futures):
                        cid = bubble_cids[idx]
                        updated_weights, num_samples, _ = future.result()
                        round_weights.append(updated_weights)
                        round_samples.append(num_samples)
                        latest_update_vectors[cid] = self._flatten_delta(updated_weights, reference_weights)
                        latest_client_weights[cid] = updated_weights
                        latest_client_samples[cid] = num_samples

                aggregated_weights = self._fedavg(round_weights, round_samples)
                next_common_weights[b_idx] = aggregated_weights

                total_metrics = {}
                total_train_samples = sum(round_samples)
                total_eval_samples = 0
                per_client_metrics = []
                
                with ThreadPoolExecutor(max_workers=1) as executor:
                    eval_futures = []
                    for cid in bubble_cids:
                        eval_futures.append(executor.submit(self.clients[cid].evaluate_shared_lstm, parameters=aggregated_weights, config={}))
                    
                    for idx, future in enumerate(eval_futures):
                        cid = bubble_cids[idx]
                        _, eval_samples, metrics = future.result()
                        total_eval_samples += eval_samples
                        
                        client_record = {
                            "client": cid,
                            "num_samples": int(eval_samples),
                            "train_samples": int(round_samples[idx]),
                        }
                        for k, v in metrics.items():
                            client_record[k] = float(v)
                            total_metrics[k] = total_metrics.get(k, 0.0) + float(v) * eval_samples
                        per_client_metrics.append(client_record)

                avg_metrics = {k: v / total_eval_samples for k, v in total_metrics.items()}
                
                history_record = {
                    "stage": "federated",
                    "bubble": b_idx,
                    "clients": bubble_cids,
                    "round": fl_round,
                    "num_samples": int(total_eval_samples),
                    "train_samples": int(total_train_samples),
                    "per_client_metrics": per_client_metrics,
                }
                history_record.update(avg_metrics)
                history.append(history_record)
                _emit(
                    f"    -> Bubble {b_idx} shared-LSTM metrics: RMSE={avg_metrics.get('rmse', 0):.4f}, SMAPE={avg_metrics.get('smape', 0):.4f}",
                    logger,
                )

            common_weights = next_common_weights
            
            # Calculate the single global reference weights representing average of all client updates
            reference_weights = self._aggregate_client_weights(
                list(latest_client_weights.keys()),
                latest_client_weights,
                latest_client_samples,
            )

            should_recluster = (
                recluster_interval > 0
                and fl_round % recluster_interval == 0
                and fl_round < num_rounds
            )
            if should_recluster:
                # Filter out preserved isolated clients from reclustering input
                recluster_vectors = {
                    cid: vec for cid, vec in latest_update_vectors.items()
                    if cid not in preserved_isolated
                }
                if len(recluster_vectors) > 2:
                    active_bubbles, new_isolated, k_star = self._cluster_update_vectors(recluster_vectors, current_round=fl_round)
                else:
                    # Not enough non-isolated clients to recluster
                    new_isolated = []
                    k_star = len(active_bubbles)
                common_global_weights = reference_weights
                common_weights = self._build_reclustered_common_weights(
                    active_bubbles=active_bubbles,
                    latest_client_weights=latest_client_weights,
                    latest_client_samples=latest_client_samples,
                    common_global_weights=common_global_weights,
                )
                self.bubbles = [bubble for bubble in active_bubbles if len(bubble) > 1]
                # Merge newly isolated clients from reclustering with the
                # original isolated list (preserved from initial clustering).
                dynamic_isolated = [cid for bubble in active_bubbles if len(bubble) == 1 for cid in bubble]
                self.isolated = list(dict.fromkeys(preserved_isolated + dynamic_isolated))
                self.shared_lstm_weights = self._copy_parameters(common_global_weights)
                _emit(
                    f"[Dynamic Recluster] round={fl_round}, k_star={k_star}, bubbles={active_bubbles}, isolated_preserved={preserved_isolated}, isolated_new={dynamic_isolated}",
                    logger,
                )
                self.save_clustering_results(
                    os.path.join(self.output_dir, "clustering_results.json"),
                    stage="dynamic_recluster",
                    round_num=fl_round,
                    k_star=k_star,
                    bubbles=active_bubbles,
                    isolated=self.isolated,
                )


        if head_finetune_epochs > 0:
            for b_idx, bubble_cids in enumerate(active_bubbles):
                if len(bubble_cids) == 1:
                    continue  # Isolated clients will be handled in step 4
                global_weights = common_weights[b_idx]
                
                with ThreadPoolExecutor(max_workers=1) as executor:
                    finetune_futures = []
                    for cid in bubble_cids:
                        client = self.clients[cid]
                        finetune_futures.append(executor.submit(
                            client.fit_head,
                            parameters=global_weights,
                            config={
                                "epochs": head_finetune_epochs,
                                "current_round": num_rounds,
                                "total_rounds": num_rounds,
                            }
                        ))
                    
                    for idx, future in enumerate(finetune_futures):
                        cid = bubble_cids[idx]
                        updated_weights, train_samples, _ = future.result()
                        _, eval_samples, metrics = self.clients[cid].evaluate(parameters=updated_weights, config={})
                        record = {
                            "stage": "head_finetune",
                            "bubble": b_idx,
                            "client": cid,
                            "epochs": head_finetune_epochs,
                            "num_samples": int(eval_samples),
                            "train_samples": int(train_samples),
                        }
                        record.update({k: float(v) for k, v in metrics.items()})
                        history.append(record)
                        _emit(
                            f"    -> Client {cid} personalized head metrics: RMSE={metrics.get('rmse', 0):.4f}, SMAPE={metrics.get('smape', 0):.4f}",
                            logger,
                        )

        self.bubbles = [bubble for bubble in active_bubbles if len(bubble) > 1]
        # Preserve the original isolated clients plus any newly isolated from
        # the last reclustering iteration.
        dynamic_isolated = [bubble[0] for bubble in active_bubbles if len(bubble) == 1]
        self.isolated = list(dict.fromkeys(preserved_isolated + dynamic_isolated))
        _emit("\nFL Training Complete.", logger)
        return history

    def step_4_personalized_learning(self, epochs=10, logger=None):
        """
        Performs local training only (Personalized FL) for isolated clients.
        """
        if not self.isolated:
            _emit("\nNo isolated clients to process.", logger)
            return []
            
        _emit("\n--- PA-CFL Step 4: Personalized Learning for Isolated Clients ---", logger)
        history = []
        
        def train_client(cid):
            _emit(f"\n[Personalized] Starting local training for client: {cid}", logger)
            client = self.clients[cid]
            if self.shared_lstm_weights is None:
                current_weights = client.get_parameters({})
                updated_weights, train_samples, _ = client.fit(
                    parameters=current_weights,
                    config={"epochs": epochs}
                )
            else:
                current_weights = self._copy_parameters(self.shared_lstm_weights)
                updated_weights, train_samples, _ = client.fit_head(
                    parameters=current_weights,
                    config={"epochs": epochs}
                )
            _, eval_samples, metrics = client.evaluate(parameters=updated_weights, config={})
            record = {
                "stage": "personalized",
                "client": cid,
                "epochs": epochs,
                "num_samples": int(eval_samples),
                "train_samples": int(train_samples),
            }
            record.update({k: float(v) for k, v in metrics.items()})
            return record

        from concurrent.futures import as_completed
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = {executor.submit(train_client, cid): cid for cid in self.isolated}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    _emit(
                        f"    -> Client {result['client']} metrics after {epochs} epochs: RMSE={result['rmse']:.4f}, SMAPE={result['smape']:.4f}",
                        logger,
                    )
                    history.append(result)

        return history
