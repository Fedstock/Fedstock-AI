import numpy as np

def calculate_feature_importance(X, y):
    """
    Train an XGBoost model and return the feature importance scores.
    """
    import xgboost as xgb

    model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, tree_method='hist')
    model.fit(X, y)
    return model.feature_importances_

def get_noisy_feature_importance(X, y, epsilon=10.0, clip_norm=0.15, seq_len=14, num_features=12):
    """
    Calculate feature importance and apply Differential Privacy.
    Uses 'Gradient Clipping' equivalent on feature importances to bound the sensitivity.
    """
    importance = calculate_feature_importance(X, y)
    
    # 14 타임스텝에 걸쳐 피처별로 중요도를 평균(Averaging)하여 12차원 벡터로 축소
    if seq_len > 1 and num_features > 0 and importance.shape[0] == seq_len * num_features:
        importance = importance.reshape(seq_len, num_features).mean(axis=0)
        
    # 1. Gradient Clipping: Limit the L2 norm of the importance vector
    norm = np.linalg.norm(importance)
    if norm > clip_norm:
        importance = importance * (clip_norm / norm)
        
    # 2. Now the global sensitivity is bounded by clip_norm
    sensitivity = clip_norm
    scale = sensitivity / epsilon if epsilon > 0 else 0
    
    # 3. Add Laplace noise
    if scale > 0:
        noise = np.random.laplace(loc=0.0, scale=scale, size=importance.shape)
        noisy_importance = importance + noise
    else:
        noisy_importance = importance
        
    return noisy_importance, sensitivity
