import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score

def normalize_importance(noisy_importances):
    """
    Normalize noisy feature importances into a distribution.
    """
    normalized = []
    for imp in noisy_importances:
        # Prevent negative values from Laplace noise
        imp_non_negative = np.maximum(imp, 0)
        sum_imp = np.sum(imp_non_negative)
        if sum_imp > 0:
            normalized.append(imp_non_negative / sum_imp)
        else:
            # Fallback to uniform distribution if sum is 0
            normalized.append(np.ones_like(imp) / len(imp))
    return np.array(normalized)

def compute_emd_distance_matrix(normalized_importances):
    """
    Compute Earth Mover's Distance (1D Wasserstein) between all pairs.
    """
    n_clients = len(normalized_importances)
    dist_matrix = np.zeros((n_clients, n_clients))
    
    # In 1D, Earth Mover's Distance is equivalent to Wasserstein distance
    # Here we are comparing distributions, so we can use indices as support points
    features_idx = np.arange(normalized_importances.shape[1])
    
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            dist = wasserstein_distance(
                u_values=features_idx, 
                v_values=features_idx, 
                u_weights=normalized_importances[i], 
                v_weights=normalized_importances[j]
            )
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
            
    return dist_matrix

def perform_clustering(noisy_importances):
    """
    Perform Agglomerative Clustering and find the optimal number of clusters 
    using the Davies-Bouldin Index.
    
    Returns:
    - optimal_labels: Cluster assignment for each client
    - k_star: Optimal number of clusters
    - multi_client_bubbles: List of clusters with >1 clients
    - isolated_clients: List of clients in clusters of size 1
    """
    normalized = normalize_importance(noisy_importances)
    dist_matrix = compute_emd_distance_matrix(normalized)
    
    n_clients = len(noisy_importances)
    if n_clients <= 2:
        # Trivial case, assign to one cluster
        return np.zeros(n_clients, dtype=int), 1, [0], []
        
    best_k = 2
    best_dbi = float('inf')
    best_labels = None
    
    # Test k from 2 to n_clients - 1
    max_k = min(n_clients - 1, 10)  # Limiting max clusters
    
    for k in range(2, max_k + 1):
        # We must use 'precomputed' affinity for custom distance matrix
        clustering = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
        labels = clustering.fit_predict(dist_matrix)
        
        # Calculate Davies-Bouldin Index
        # DBI requires original coordinates or we can use the distance matrix indirectly.
        # But sklearn's davies_bouldin_score expects feature array, not distance matrix.
        # So we pass the normalized distributions as features.
        try:
            dbi = davies_bouldin_score(normalized, labels)
            if dbi < best_dbi:
                best_dbi = dbi
                best_k = k
                best_labels = labels
        except ValueError:
            continue
            
    if best_labels is None:
        best_labels = np.zeros(n_clients, dtype=int)
        best_k = 1
        
    # Analyze bubbles
    labels_counts = np.bincount(best_labels)
    multi_client_bubbles = np.where(labels_counts > 1)[0].tolist()
    single_client_bubbles = np.where(labels_counts == 1)[0].tolist()
    
    isolated_clients = []
    for cluster_id in single_client_bubbles:
        client_idx = np.where(best_labels == cluster_id)[0][0]
        isolated_clients.append(int(client_idx))
        
    return best_labels, best_k, multi_client_bubbles, isolated_clients
