import numpy as np
import flwr as fl
from src.fl.server_clustering import perform_clustering

class BubbleServer:
    """
    Simulates the central server orchestrating PA-CFL.
    """
    def __init__(self, clients_dict):
        """
        clients_dict: dict of {client_id: FedStockClient instance}
        """
        self.clients = clients_dict
        self.bubbles = []
        self.isolated = []
        
    def step_1_collect_and_cluster(self):
        """
        Collects noisy feature importances and performs agglomerative clustering.
        """
        print("--- PA-CFL Step 1 & 2: Feature Collection & Clustering ---")
        noisy_importances = []
        client_ids = list(self.clients.keys())
        
        for cid in client_ids:
            client = self.clients[cid]
            noisy_imp = client.extract_noisy_importance()
            noisy_importances.append(noisy_imp)
            
        noisy_importances = np.array(noisy_importances)
        labels, k_star, multi_bubbles, single_bubbles = perform_clustering(noisy_importances)
        
        print(f"Optimal Clusters (k*): {k_star}")
        
        # Group client IDs into bubbles
        bubble_groups = {}
        for idx, label in enumerate(labels):
            if label not in bubble_groups:
                bubble_groups[label] = []
            bubble_groups[label].append(client_ids[idx])
            
        # Separate into multi-client and single-client
        for label, cids in bubble_groups.items():
            if len(cids) > 1:
                self.bubbles.append(cids)
            else:
                self.isolated.extend(cids)
                
        print(f"Multi-Client Bubbles: {self.bubbles}")
        print(f"Isolated (Single-Client) Bubbles: {self.isolated}")
        
    def step_3_federated_learning(self, num_rounds=3, epochs_per_round=5):
        """
        Performs FedAvg within each multi-client bubble.
        """
        print("\n--- PA-CFL Step 3: Federated Learning within Bubbles ---")
        
        for b_idx, bubble_cids in enumerate(self.bubbles):
            print(f"\n[Bubble {b_idx}] Starting FL with clients: {bubble_cids}")
            
            # Initialize global model weights (using the first client's initial weights)
            global_weights = self.clients[bubble_cids[0]].get_parameters({})
            
            for fl_round in range(1, num_rounds + 1):
                print(f"  Round {fl_round}/{num_rounds}")
                
                # 1. Distribute global weights and Train locally
                round_weights = []
                round_samples = []
                
                for cid in bubble_cids:
                    client = self.clients[cid]
                    # fit(parameters, config) -> returns updated_parameters, num_samples, metrics
                    updated_weights, num_samples, _ = client.fit(
                        parameters=global_weights, 
                        config={"epochs": epochs_per_round}
                    )
                    round_weights.append(updated_weights)
                    round_samples.append(num_samples)
                    
                # 2. Aggregate using FedAvg
                total_samples = sum(round_samples)
                aggregated_weights = []
                
                for layer_idx in range(len(global_weights)):
                    layer_weighted_sum = sum(
                        round_weights[c_idx][layer_idx] * (round_samples[c_idx] / total_samples)
                        for c_idx in range(len(bubble_cids))
                    )
                    aggregated_weights.append(layer_weighted_sum)
                    
                global_weights = aggregated_weights
                
                # 3. Evaluate on clients
                total_rmse = 0.0
                total_smape = 0.0
                
                for cid in bubble_cids:
                    client = self.clients[cid]
                    _, _, metrics = client.evaluate(parameters=global_weights, config={})
                    total_rmse += metrics["rmse"] * round_samples[bubble_cids.index(cid)]
                    total_smape += metrics["smape"] * round_samples[bubble_cids.index(cid)]
                    
                avg_rmse = total_rmse / total_samples
                avg_smape = total_smape / total_samples
                
                print(f"    -> Bubble {b_idx} Global metrics: RMSE={avg_rmse:.4f}, SMAPE={avg_smape:.4f}")
                
        print("\nFL Training Complete. Isolated clients did not participate.")
