import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist, squareform
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

def compute_cosine_distance_matrix(vectors):
    """
    Compute Cosine Distance (1 - Cosine Similarity) between all pairs of vectors.
    """
    dists = pdist(vectors, metric='cosine')
    # If any NaN in dists (e.g. due to zero vector), replace with 1.0 (completely dissimilar)
    dists = np.nan_to_num(dists, nan=1.0)
    dist_matrix = squareform(dists)
    np.fill_diagonal(dist_matrix, 0.0)
    return dist_matrix

def _fit_agglomerative(dist_matrix, n_clusters):
    """
    Fit agglomerative clustering with a precomputed distance matrix.
    """
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average',
    )
    return clustering.fit_predict(dist_matrix)

def _compact_labels(labels):
    """
    Re-map cluster labels to a dense 0..k-1 range.
    """
    unique_labels = sorted(set(labels))
    mapping = {old: new for new, old in enumerate(unique_labels)}
    return np.array([mapping[label] for label in labels], dtype=int)

def _resolve_max_clusters(n_clients, max_clusters):
    """
    Keep the k search space conservative. Searching up to n_clients - 1 tends to
    favor fragmented clusters when client distributions are heterogeneous.
    """
    if max_clusters is None:
        max_clusters = int(np.ceil(np.sqrt(n_clients)))
    return max(2, min(n_clients - 1, max_clusters))

def _compute_filtered_dbi(X, labels):
    """
    Compute Davies-Bouldin Index excluding singleton clusters (size == 1) from the sum/average.
    """
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    
    centroids = []
    dispersions = []
    sizes = []
    
    for label in unique_labels:
        cluster_points = X[labels == label]
        size = len(cluster_points)
        sizes.append(size)
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
        # Dispersion: average Euclidean distance of points in cluster to centroid
        disp = np.mean(np.linalg.norm(cluster_points - centroid, axis=1))
        dispersions.append(disp)
        
    centroids = np.array(centroids)
    dispersions = np.array(dispersions)
    sizes = np.array(sizes)
    
    # Identify non-singleton clusters (size > 1)
    non_singletons = np.where(sizes > 1)[0]
    
    # Fallback to standard DBI if no non-singleton clusters
    if len(non_singletons) == 0:
        return davies_bouldin_score(X, labels)
        
    R = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                dist = np.linalg.norm(centroids[i] - centroids[j])
                if dist > 1e-8:
                    R[i, j] = (dispersions[i] + dispersions[j]) / dist
                else:
                    R[i, j] = 0.0
                    
    # Calculate D_i (max similarity) for non-singleton clusters only
    D = []
    for i in non_singletons:
        max_r = 0.0
        for j in range(k):
            if i != j:
                if R[i, j] > max_r:
                    max_r = R[i, j]
        D.append(max_r)
        
    return float(np.mean(D))

def _candidate_score(normalized, labels, k, complexity_penalty, singleton_penalty):
    """
    DBI alone does not penalize singleton-heavy solutions enough for this PA-CFL
    use case. Add small penalties so FL bubbles stay useful unless the data
    strongly supports a split.
    """
    dbi = _compute_filtered_dbi(normalized, labels)
    labels_counts = np.bincount(labels)
    singleton_count = int(np.sum(labels_counts == 1))
    score = dbi + complexity_penalty * max(0, k - 2) + singleton_penalty * singleton_count
    return score, dbi, singleton_count

def _merge_non_isolated_singletons(labels, dist_matrix, isolation_std_multiplier):
    """
    Merge singleton clusters back into the nearest multi-client cluster unless
    the singleton is far enough to be treated as a true isolated client.
    """
    labels = np.array(labels, dtype=int).copy()
    nonzero_distances = dist_matrix[dist_matrix > 0]
    if nonzero_distances.size == 0:
        return _compact_labels(labels)

    isolation_threshold = (
        float(nonzero_distances.mean())
        + float(nonzero_distances.std()) * isolation_std_multiplier
    )

    for cluster_id in sorted(set(labels)):
        members = np.where(labels == cluster_id)[0]
        if len(members) != 1:
            continue

        client_idx = members[0]
        target_cluster = None
        target_distance = float('inf')

        for other_cluster_id in sorted(set(labels)):
            if other_cluster_id == cluster_id:
                continue

            other_members = np.where(labels == other_cluster_id)[0]
            if len(other_members) == 0:
                continue

            avg_distance = float(dist_matrix[client_idx, other_members].mean())
            if avg_distance < target_distance:
                target_distance = avg_distance
                target_cluster = other_cluster_id

        if target_cluster is not None and target_distance <= isolation_threshold:
            labels[client_idx] = target_cluster

    return _compact_labels(labels)

def perform_clustering(
    noisy_importances,
    max_clusters=None,
    complexity_penalty=0.03,
    singleton_penalty=0.15,
    isolation_std_multiplier=0.75,
    ema_dict=None,
    client_ids=None,
    ema_alpha=0.8,
    metric='cosine',
    feature_importances=None,
    feature_weight=0.5,
):
    """
    Perform Agglomerative Clustering and find the optimal number of clusters 
    using a regularized Davies-Bouldin Index.
    
    Returns:
    - optimal_labels: Cluster assignment for each client
    - k_star: Optimal number of clusters
    - multi_client_bubbles: List of clusters with >1 clients
    - isolated_clients: List of clients in clusters of size 1
    """
    if metric == 'emd':
        normalized = normalize_importance(noisy_importances)
        
        # Apply EMA on the normalized distributions if requested
        if ema_dict is not None and client_ids is not None:
            for idx, cid in enumerate(client_ids):
                if cid in ema_dict:
                    # Check for shape mismatch (e.g., transitioning from feature importances to model weights)
                    if ema_dict[cid].shape == normalized[idx].shape:
                        normalized[idx] = ema_alpha * ema_dict[cid] + (1 - ema_alpha) * normalized[idx]
                ema_dict[cid] = normalized[idx].copy()
        dist_matrix = compute_emd_distance_matrix(normalized)
        features = normalized
    elif metric == 'cosine':
        vectors = np.array(noisy_importances, dtype=np.float32)
        
        # Apply EMA on the raw vectors if requested
        if ema_dict is not None and client_ids is not None:
            for idx, cid in enumerate(client_ids):
                if cid in ema_dict:
                    if ema_dict[cid].shape == vectors[idx].shape:
                        vectors[idx] = ema_alpha * ema_dict[cid] + (1 - ema_alpha) * vectors[idx]
                ema_dict[cid] = vectors[idx].copy()
                
        # L2-normalize vectors for cosine compatibility in Euclidean DBI
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Weight delta scale correction: divide by the mean norm across all clients
        # to preserve the relative scale of updates for active vs converged clients.
        # This correction should only apply to weight deltas (i.e., when feature_importances is not None).
        # If feature_importances is None, this is a pure feature importance clustering, where standard L2 normalization is correct.
        if feature_importances is not None:
            mean_norm = np.mean(norms)
            if mean_norm > 1e-8:
                scaled_vectors = vectors / mean_norm
            else:
                scaled_vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 1e-8)
        else:
            scaled_vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 1e-8)
        
        if feature_importances is not None:
            # Feature importances L2-normalization
            feats = np.array(feature_importances, dtype=np.float32)
            f_norms = np.linalg.norm(feats, axis=1, keepdims=True)
            normalized_feats = np.divide(feats, f_norms, out=np.zeros_like(feats), where=f_norms > 1e-8)
            
            # Weighted concatenation
            w_feat = np.sqrt(feature_weight)
            w_weight = np.sqrt(1.0 - feature_weight)
            
            features = np.hstack([
                w_feat * normalized_feats,
                w_weight * scaled_vectors
            ])
        else:
            features = scaled_vectors
        
        dist_matrix = compute_cosine_distance_matrix(features)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    n_clients = len(noisy_importances)
    if n_clients <= 2:
        # Trivial case, assign to one cluster
        return np.zeros(n_clients, dtype=int), 1, [0], []
        
    best_k = 2
    best_score = float('inf')
    best_labels = None
    
    max_k = _resolve_max_clusters(n_clients, max_clusters)
    
    for k in range(2, max_k + 1):
        if metric == 'cosine':
            clustering = AgglomerativeClustering(
                n_clusters=k,
                metric='euclidean',
                linkage='ward',
            )
            labels = clustering.fit_predict(features)
        else:
            labels = _fit_agglomerative(dist_matrix, k)
        
        # Calculate Davies-Bouldin Index
        # DBI requires original coordinates or we can use the distance matrix indirectly.
        # But sklearn's davies_bouldin_score expects feature array, not distance matrix.
        # So we pass the normalized distributions/raw vectors as features.
        try:
            score, _, _ = _candidate_score(
                features,
                labels,
                k,
                complexity_penalty,
                singleton_penalty,
            )
            if score < best_score:
                best_score = score
                best_k = k
                best_labels = labels
        except ValueError:
            continue
            
    if best_labels is None:
        best_labels = np.zeros(n_clients, dtype=int)
        best_k = 1
    else:
        best_labels = _merge_non_isolated_singletons(
            best_labels,
            dist_matrix,
            isolation_std_multiplier,
        )
        best_k = len(set(best_labels))
        
    # Analyze bubbles
    labels_counts = np.bincount(best_labels)
    multi_client_bubbles = np.where(labels_counts > 1)[0].tolist()
    single_client_bubbles = np.where(labels_counts == 1)[0].tolist()
    
    isolated_clients = []
    for cluster_id in single_client_bubbles:
        client_idx = np.where(best_labels == cluster_id)[0][0]
        isolated_clients.append(int(client_idx))
        
    return best_labels, best_k, multi_client_bubbles, isolated_clients


def _client_representation(importance, metric):
    """
    Build the single-vector representation used for distance comparisons,
    matching how perform_clustering treats feature importances per metric.
    """
    vector = np.asarray(importance, dtype=np.float32).reshape(-1)
    if metric == 'emd':
        return normalize_importance([vector])[0]
    return vector


def _point_distance(u, v, metric):
    """
    Distance between two client representations, consistent with the matrices
    built in compute_emd_distance_matrix / compute_cosine_distance_matrix.
    """
    if metric == 'emd':
        support = np.arange(len(u))
        return float(wasserstein_distance(support, support, u_weights=u, v_weights=v))

    # cosine distance (1 - cosine similarity); zero vectors are fully dissimilar
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u < 1e-12 or norm_v < 1e-12:
        return 1.0
    similarity = float(np.dot(u, v) / (norm_u * norm_v))
    return float(np.clip(1.0 - similarity, 0.0, 2.0))


def _isolation_threshold(existing_representations, metric, isolation_std_multiplier):
    """
    Reproduce the batch isolation threshold used in _merge_non_isolated_singletons
    so incremental assignment stays consistent with the initial clustering.
    """
    if len(existing_representations) < 2:
        return None

    vectors = np.array(existing_representations, dtype=np.float32)
    if metric == 'emd':
        dist_matrix = compute_emd_distance_matrix(vectors)
    else:
        dist_matrix = compute_cosine_distance_matrix(vectors)

    nonzero_distances = dist_matrix[dist_matrix > 0]
    if nonzero_distances.size == 0:
        return None
    return (
        float(nonzero_distances.mean())
        + float(nonzero_distances.std()) * isolation_std_multiplier
    )


def assign_new_client(
    new_importance,
    existing_importances,
    bubbles,
    isolated,
    metric='cosine',
    isolation_std_multiplier=0.75,
    ema_dict=None,
    new_client_id=None,
    ema_alpha=0.8,
):
    """
    Incrementally place a single newly-arrived client into the existing
    clustering without re-running the full batch perform_clustering.

    The new client's noisy feature importance is compared (average linkage,
    same metric as the initial clustering) against every existing multi-client
    bubble and every isolated client. It joins the nearest cluster when that
    cluster is within the batch isolation threshold; otherwise it stays
    isolated for personalized FL. Matching a lone isolated client forms a new
    two-client bubble, mirroring _merge_non_isolated_singletons.

    Parameters
    ----------
    new_importance : array-like
        Noisy feature importance vector of the new client.
    existing_importances : dict
        {client_id: importance vector} for all currently known clients.
    bubbles : list[list[str]]
        Current multi-client bubbles (each a list of client ids).
    isolated : list[str]
        Current isolated client ids.

    Returns
    -------
    dict with keys:
        assigned_to    : 'bubble' | 'new_bubble' | 'isolated'
        bubble_index   : index into the returned bubbles, or None if isolated
        partner        : the isolated client paired with (for 'new_bubble')
        distance       : nearest cluster distance (None if no comparison made)
        threshold      : isolation threshold used (None if not computable)
        bubbles        : updated list of multi-client bubbles
        isolated       : updated list of isolated client ids
    """
    bubbles = [list(bubble) for bubble in bubbles]
    isolated = list(isolated)

    new_repr = _client_representation(new_importance, metric)

    # Pre-compute every existing client's representation once.
    existing_repr = {
        cid: _client_representation(imp, metric)
        for cid, imp in existing_importances.items()
    }

    if ema_dict is not None and new_client_id is not None:
        ema_dict[new_client_id] = new_repr.copy()

    # No prior clients to compare against -> nothing to merge into.
    if not existing_repr:
        if new_client_id is not None and new_client_id not in isolated:
            isolated.append(new_client_id)
        return {
            "assigned_to": "isolated",
            "bubble_index": None,
            "partner": None,
            "distance": None,
            "threshold": None,
            "bubbles": bubbles,
            "isolated": isolated,
        }

    threshold = _isolation_threshold(
        list(existing_repr.values()), metric, isolation_std_multiplier
    )

    # Candidate clusters: every multi-client bubble, then every isolated client.
    candidates = []
    for b_idx, bubble in enumerate(bubbles):
        members = [cid for cid in bubble if cid in existing_repr]
        if not members:
            continue
        avg_distance = float(
            np.mean([_point_distance(new_repr, existing_repr[cid], metric) for cid in members])
        )
        candidates.append({"kind": "bubble", "index": b_idx, "distance": avg_distance})

    for cid in isolated:
        if cid not in existing_repr:
            continue
        distance = _point_distance(new_repr, existing_repr[cid], metric)
        candidates.append({"kind": "isolated", "partner": cid, "distance": distance})

    if not candidates:
        if new_client_id is not None and new_client_id not in isolated:
            isolated.append(new_client_id)
        return {
            "assigned_to": "isolated",
            "bubble_index": None,
            "partner": None,
            "distance": None,
            "threshold": threshold,
            "bubbles": bubbles,
            "isolated": isolated,
        }

    nearest = min(candidates, key=lambda c: c["distance"])
    nearest_distance = nearest["distance"]

    # Without a data-driven threshold (fewer than 2 prior clients) we cannot
    # judge similarity reliably, so we keep the new client isolated.
    close_enough = threshold is not None and nearest_distance <= threshold

    if close_enough and nearest["kind"] == "bubble":
        b_idx = nearest["index"]
        if new_client_id is not None and new_client_id not in bubbles[b_idx]:
            bubbles[b_idx].append(new_client_id)
        return {
            "assigned_to": "bubble",
            "bubble_index": b_idx,
            "partner": None,
            "distance": nearest_distance,
            "threshold": threshold,
            "bubbles": bubbles,
            "isolated": isolated,
        }

    if close_enough and nearest["kind"] == "isolated":
        partner = nearest["partner"]
        if partner in isolated:
            isolated.remove(partner)
        new_bubble = [partner]
        if new_client_id is not None:
            new_bubble.append(new_client_id)
        bubbles.append(new_bubble)
        return {
            "assigned_to": "new_bubble",
            "bubble_index": len(bubbles) - 1,
            "partner": partner,
            "distance": nearest_distance,
            "threshold": threshold,
            "bubbles": bubbles,
            "isolated": isolated,
        }

    if new_client_id is not None and new_client_id not in isolated:
        isolated.append(new_client_id)
    return {
        "assigned_to": "isolated",
        "bubble_index": None,
        "partner": None,
        "distance": nearest_distance,
        "threshold": threshold,
        "bubbles": bubbles,
        "isolated": isolated,
    }


def run_clustering_pipeline(feature_json_path, output_json_path=None):
    """
    Load extracted feature importances from JSON, perform Cosine Distance clustering,
    and return/save the cluster assignments.
    """
    import json
    import os
    
    print(f"Loading feature importances from {feature_json_path}...")
    with open(feature_json_path, 'r') as f:
        client_data = json.load(f)
        
    client_ids = list(client_data.keys())
    noisy_importances = np.array([client_data[cid] for cid in client_ids])
    
    print(f"Loaded {len(client_ids)} clients. Running Agglomerative Clustering with Cosine Distance...")
    labels, k_star, multi_bubbles, isolated = perform_clustering(noisy_importances)
    
    # Create output dictionary
    cluster_assignments = {}
    for i, cid in enumerate(client_ids):
        cluster_assignments[cid] = int(labels[i])
        
    print(f"\n--- Clustering Results ---")
    print(f"Optimal number of clusters (k*): {k_star}")
    
    # Print clients in each cluster
    for k in range(k_star):
        members = [client_ids[i] for i, label in enumerate(labels) if label == k]
        print(f"Cluster {k}: {members}")
        
    print(f"\nMulti-client bubbles (for standard FL): {multi_bubbles}")
    print(f"Isolated clients (for personalized FL): {[client_ids[i] for i in isolated]}")
    
    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        isolated_clients = [client_ids[i] for i in isolated]
        cluster_sizes = [len(bubble) for bubble in multi_bubbles] + [1 for _ in isolated_clients]
        record = {
            "sequence": 1,
            "stage": "initial_clustering",
            "round": 0,
            "k_star": int(k_star),
            "total_clients": int(sum(cluster_sizes)),
            "num_clusters": len(cluster_sizes),
            "num_multi_client_bubbles": len(multi_bubbles),
            "num_isolated_clients": len(isolated_clients),
            "cluster_sizes": cluster_sizes,
            "cluster_size_stats": {
                "min": int(min(cluster_sizes)) if cluster_sizes else 0,
                "max": int(max(cluster_sizes)) if cluster_sizes else 0,
                "mean": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
            },
            "multi_client_bubbles": [
                {
                    "bubble_id": idx,
                    "size": len(bubble),
                }
                for idx, bubble in enumerate(multi_bubbles)
            ],
            "isolated_clients": isolated_clients,
        }
        with open(output_json_path, 'w') as f:
            json.dump({
                "schema_version": 2,
                "description": "Chronological clustering records. Client-to-cluster assignments and per-bubble client lists are omitted.",
                "records": [record],
            }, f, indent=4)
        print(f"\nSaved clustering results to {output_json_path}")
        
    return cluster_assignments

if __name__ == "__main__":
    import os
    
    # 프로젝트 루트 경로 (현재 파일 기준)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    INPUT_JSON = os.path.join(project_root, "outputs", "feature_importances.json")
    OUTPUT_JSON = os.path.join(project_root, "outputs", "clustering_results.json")
    
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found. Please run extract_features.py first.")
    else:
        run_clustering_pipeline(INPUT_JSON, OUTPUT_JSON)
