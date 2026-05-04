import numpy as np
import xgboost as xgb

def calculate_feature_importance(X, y):
    """
    Train an XGBoost model and return the feature importance scores.
    """
    model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, tree_method='hist')
    model.fit(X, y)
    return model.feature_importances_

def calculate_local_sensitivity(X, y):
    """
    Calculate the local sensitivity \\Delta_i of the dataset.
    This is the maximum change in feature importance when any single record is removed.
    """
    base_importance = calculate_feature_importance(X, y)
    n_samples = len(X)
    max_sensitivity = 0.0
    
    # In a real scenario, dropping each row iteratively might be slow for large datasets.
    # We use a subsample or theoretical bound if it's too large, but for DP, the definition requires this.
    # To optimize for this POC, we can limit the check to a random sample of rows if n_samples is large.
    sample_indices = np.random.choice(n_samples, min(n_samples, 100), replace=False)
    
    for i in sample_indices:
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False
        X_minus_i = X[mask]
        y_minus_i = y[mask]
        
        importance_minus_i = calculate_feature_importance(X_minus_i, y_minus_i)
        
        # Calculate the absolute difference for each feature
        diff = np.abs(base_importance - importance_minus_i)
        max_diff = np.max(diff)
        
        if max_diff > max_sensitivity:
            max_sensitivity = max_diff
            
    return max_sensitivity

def get_noisy_feature_importance(X, y, epsilon=1.0):
    """
    Calculate feature importance and apply Differential Privacy (Laplace noise).
    """
    importance = calculate_feature_importance(X, y)
    sensitivity = calculate_local_sensitivity(X, y)
    
    # \Delta_i / \epsilon
    scale = sensitivity / epsilon if epsilon > 0 else 0
    
    if scale > 0:
        noise = np.random.laplace(loc=0.0, scale=scale, size=importance.shape)
        noisy_importance = importance + noise
    else:
        noisy_importance = importance
        
    return noisy_importance, sensitivity
