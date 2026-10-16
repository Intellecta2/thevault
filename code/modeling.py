"""
Modeling Module
================
Trains an ensemble of gradient-boosted trees to predict post_hire_score
and rank candidates for the Corporate Heist shortlist.

Ensemble: LightGBM + XGBoost + CatBoost
Training: Regression on post_hire_score + Classification on top-5% label
Validation: Ledger (dev.csv + dev_winners.csv) — Precision@150 & NDCG
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import ndcg_score


SEED = 42


# ============================================================
# 1. TRAINING
# ============================================================

def train_lightgbm_regressor(X_train, y_train, feature_names=None):
    """Train a LightGBM regressor to predict post_hire_score."""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 128,
        'learning_rate': 0.032,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'seed': SEED,
        'n_jobs': 2,
    }
    
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    model = lgb.train(
        params, dtrain,
        num_boost_round=1400,
        callbacks=[lgb.log_evaluation(0)],  # suppress output
    )
    return model


def train_xgboost_regressor(X_train, y_train):
    """Train an XGBoost regressor to predict post_hire_score."""
    model = xgb.XGBRegressor(
        n_estimators=1400,
        max_depth=6,
        learning_rate=0.0058,
        subsample=0.866,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        random_state=SEED,
        n_jobs=2,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def train_catboost_regressor(X_train, y_train):
    """Train a CatBoost regressor to predict post_hire_score."""
    model = CatBoostRegressor(
        iterations=1400,
        depth=9,
        learning_rate=0.0059,
        l2_leaf_reg=3.845,
        random_seed=SEED,
        verbose=0,
        thread_count=2,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm_classifier(X_train, y_train_binary, feature_names=None):
    """Train a LightGBM classifier for top-5% membership."""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 128,
        'learning_rate': 0.032,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'seed': SEED,
        'n_jobs': 2,
        'is_unbalance': True,
    }
    
    dtrain = lgb.Dataset(X_train, label=y_train_binary, feature_name=feature_names)
    model = lgb.train(
        params, dtrain,
        num_boost_round=1400,
        callbacks=[lgb.log_evaluation(0)],
    )
    return model


def train_xgboost_classifier(X_train, y_train_binary):
    """Train an XGBoost classifier for top-5% membership."""
    model = xgb.XGBClassifier(
        n_estimators=1400,
        max_depth=6,
        learning_rate=0.0058,
        subsample=0.866,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        random_state=SEED,
        n_jobs=2,
        verbosity=0,
        scale_pos_weight=(y_train_binary == 0).sum() / max((y_train_binary == 1).sum(), 1),
    )
    model.fit(X_train, y_train_binary)
    return model


# ============================================================
# 2. ENSEMBLE PREDICTION
# ============================================================

def train_ensemble(X_train, y_train, feature_names=None):
    """Train all models in the ensemble.
    
    Returns:
        dict of trained models with keys describing each model
    """
    # Prepare binary labels for classification models
    threshold = np.percentile(y_train, 95)
    y_binary = (y_train >= threshold).astype(int)
    
    print(f"  Training data: {len(X_train)} samples")
    print(f"  Top 5% threshold: {threshold:.1f}")
    print(f"  Positive class: {y_binary.sum()} ({y_binary.mean()*100:.1f}%)")
    
    models = {}
    
    # Regression models
    print("  Training LightGBM regressor...")
    models['lgb_reg'] = train_lightgbm_regressor(X_train, y_train, feature_names)
    
    print("  Training XGBoost regressor...")
    models['xgb_reg'] = train_xgboost_regressor(X_train, y_train)
    
    print("  Training CatBoost regressor...")
    models['cat_reg'] = train_catboost_regressor(X_train, y_train)
    
    # Classification models
    print("  Training LightGBM classifier...")
    models['lgb_cls'] = train_lightgbm_classifier(X_train, y_binary, feature_names)
    
    print("  Training XGBoost classifier...")
    models['xgb_cls'] = train_xgboost_classifier(X_train, y_binary)
    
    print("  Training CatBoost classifier...")
    cat_cls = CatBoostClassifier(
        iterations=1400,
        depth=9,
        learning_rate=0.0059,
        l2_leaf_reg=3.845,
        auto_class_weights='Balanced',
        random_seed=SEED,
        verbose=0,
        thread_count=2,
    )
    cat_cls.fit(X_train, y_binary)
    models['cat_cls'] = cat_cls
    
    return models


def predict_ensemble(models, X, feature_names=None):
    """Get ensemble prediction scores.
    
    Combines regression predictions (score) and classification probabilities
    into a unified ranking score.
    
    Returns:
        numpy array of ensemble scores (higher = better candidate)
    """
    predictions = {}
    
    # Regression predictions (predict post_hire_score)
    predictions['lgb_reg'] = models['lgb_reg'].predict(X)
    predictions['xgb_reg'] = models['xgb_reg'].predict(X)
    predictions['cat_reg'] = models['cat_reg'].predict(np.array(X))
    
    # Classification predictions (predict P(top 5%))
    predictions['lgb_cls'] = models['lgb_cls'].predict(X)
    predictions['xgb_cls'] = models['xgb_cls'].predict_proba(X)[:, 1]
    predictions['cat_cls'] = models['cat_cls'].predict_proba(np.array(X))[:, 1]
    
    # Normalize each to 0-1 range using rank-based normalization
    # (more robust than min-max for ranking)
    from scipy.stats import rankdata
    normalized = {}
    for key, preds in predictions.items():
        normalized[key] = rankdata(preds) / len(preds)
    
    # Weighted ensemble — classifiers get more weight as they
    # directly optimize for top-5% membership
    weights = {
        'lgb_reg': 0.302,
        'xgb_reg': 0.353,
        'cat_reg': 0.136,
        'lgb_cls': 0.032,
        'xgb_cls': 0.039,
        'cat_cls': 0.138,
    }
    
    ensemble_score = np.zeros(len(X))
    for key, w in weights.items():
        ensemble_score += w * normalized[key]
    
    return ensemble_score


# ============================================================
# 3. VALIDATION
# ============================================================

def validate_on_dev(models, X_dev, dev_df, dev_winners_df, k=150):
    """Validate model on the Ledger (dev set).
    
    Compares predicted top-k with actual dev_winners.
    
    Returns:
        dict with precision@k, recall@k, and overlap count
    """
    # Get ensemble predictions
    scores = predict_ensemble(models, X_dev)
    
    # Get top-k predicted candidates
    dev_df = dev_df.copy()
    dev_df['predicted_score'] = scores
    top_k = dev_df.nlargest(k, 'predicted_score')
    predicted_ids = set(top_k['candidate_id'].values)
    
    # Get actual winners
    actual_ids = set(dev_winners_df['candidate_id'].values)
    
    # Compute metrics
    overlap = predicted_ids & actual_ids
    precision = len(overlap) / k
    recall = len(overlap) / len(actual_ids)
    
    print(f"\n  === Dev Set Validation (top {k}) ===")
    print(f"  Overlap: {len(overlap)} / {len(actual_ids)} actual winners")
    print(f"  Precision@{k}: {precision:.4f}")
    print(f"  Recall@{k}: {recall:.4f}")
    print(f"  F1@{k}: {2*precision*recall/(precision+recall+1e-10):.4f}")
    
    return {
        'overlap': len(overlap),
        'precision': precision,
        'recall': recall,
        'predicted_ids': predicted_ids,
        'actual_ids': actual_ids,
    }


def get_feature_importance(models, feature_names):
    """Get combined feature importance from ensemble models.
    
    Returns:
        DataFrame with feature names and importance scores
    """
    importances = {}
    
    # LightGBM feature importance
    lgb_imp = models['lgb_reg'].feature_importance(importance_type='gain')
    for i, name in enumerate(feature_names):
        if i < len(lgb_imp):
            importances[name] = importances.get(name, 0) + lgb_imp[i]
    
    # XGBoost feature importance
    xgb_imp = models['xgb_reg'].feature_importances_
    for i, name in enumerate(feature_names):
        if i < len(xgb_imp):
            importances[name] = importances.get(name, 0) + xgb_imp[i]
    
    # Sort by importance
    imp_df = pd.DataFrame([
        {'feature': k, 'importance': v}
        for k, v in importances.items()
    ]).sort_values('importance', ascending=False)
    
    return imp_df
