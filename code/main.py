"""
=============================================================================
THE CORPORATE HEIST — Main Pipeline
=============================================================================
Entry point for the complete prediction pipeline.

Usage:
    python code/main.py

Expects in the working directory:
    train.csv, dev.csv, dev_winners.csv, test.csv

Produces:
    submission.csv (500 ranked candidate IDs)

Runtime: < 5 minutes on 2 cores, 4 GB RAM, CPU only
Random seed: 42 (all models, all operations)

Pipeline:
    Phase 1: Data Cleaning & Parsing
    Phase 2: Fabrication & Duplicate Detection
    Phase 3: Feature Engineering
    Phase 4: Model Training (ensemble) + Dev Validation
    Phase 5: Cycle-Specific Adjustments (insider intelligence)
    Phase 6: Ranking & Submission
=============================================================================
"""

import os
import sys
import time
import random
import warnings

import numpy as np
import pandas as pd

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# ============================================================
# REPRODUCIBILITY: Set ALL random seeds
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Ensure imports work from both code/ and project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_cleaning import clean_dataframe, impute_missing
from fraud_detection import detect_fabrication, detect_duplicates, resolve_duplicates
from feature_engineering import build_feature_matrix, add_test_specific_features
from modeling import train_ensemble, predict_ensemble, validate_on_dev, get_feature_importance
from cycle_adjustments import apply_hard_filters, apply_cycle_adjustments, generate_final_ranking


def main():
    """Run the complete Corporate Heist prediction pipeline."""
    start_time = time.time()
    
    print("=" * 70)
    print("  THE CORPORATE HEIST — Prediction Pipeline")
    print("=" * 70)
    
    # ============================================================
    # PHASE 0: Load Data
    # ============================================================
    def _load(fname):
        if os.path.exists(fname):
            return pd.read_csv(fname)
        alt = os.path.join('corporate_heist_data', fname)
        if os.path.exists(alt):
            return pd.read_csv(alt)
        raise FileNotFoundError(f"Could not find {fname}")
    
    train_raw = _load('train.csv')
    dev_raw = _load('dev.csv')
    dev_winners = _load('dev_winners.csv')
    test_raw = _load('test.csv')
    
    print(f"  Train: {train_raw.shape[0]} rows, {train_raw.shape[1]} cols")
    print(f"  Dev:   {dev_raw.shape[0]} rows, {dev_raw.shape[1]} cols")
    print(f"  Dev winners: {dev_winners.shape[0]} rows")
    print(f"  Test:  {test_raw.shape[0]} rows, {test_raw.shape[1]} cols")
    
    elapsed = time.time() - start_time
    print(f"  [Time: {elapsed:.1f}s]")
    
    # ============================================================
    # PHASE 1: Data Cleaning & Parsing
    # ============================================================
    print("\n[Phase 1] Cleaning and parsing data...")
    
    train = clean_dataframe(train_raw, is_test=False)
    dev = clean_dataframe(dev_raw, is_test=False)
    test = clean_dataframe(test_raw, is_test=True)
    
    # Impute missing values (using train as reference)
    train = impute_missing(train, reference_df=train)
    dev = impute_missing(dev, reference_df=train)
    test = impute_missing(test, reference_df=train)
    
    elapsed = time.time() - start_time
    print(f"  Cleaning complete. [Time: {elapsed:.1f}s]")
    
    # ============================================================
    # PHASE 2: Fabrication & Duplicate Detection
    # ============================================================
    print("\n[Phase 2] Detecting fabricated profiles and duplicates...")
    
    # Detect fabrication in all datasets
    train = detect_fabrication(train)
    dev = detect_fabrication(dev)
    test = detect_fabrication(test)
    
    print(f"  Train fabricated: {train['is_fabricated'].sum()} / {len(train)}")
    print(f"  Dev fabricated:   {dev['is_fabricated'].sum()} / {len(dev)}")
    print(f"  Test fabricated:  {test['is_fabricated'].sum()} / {len(test)}")
    
    # Detect duplicates in test set
    test = detect_duplicates(test)
    dup_exclude_ids = resolve_duplicates(test)
    
    print(f"  Test duplicates found: {test['is_duplicate'].sum()}")
    print(f"  Duplicate entries to exclude: {len(dup_exclude_ids)}")
    
    elapsed = time.time() - start_time
    print(f"  [Time: {elapsed:.1f}s]")
    
    # ============================================================
    # PHASE 3: Feature Engineering
    # ============================================================
    print("\n[Phase 3] Engineering features...")
    
    # Build feature matrices
    X_train, feature_names, role_encoder = build_feature_matrix(train)
    X_dev, _, _ = build_feature_matrix(dev, fit_encoder=role_encoder)
    X_test, _, _ = build_feature_matrix(test, fit_encoder=role_encoder)
    
    # Align columns (ensure all datasets have the same features)
    # Add missing columns as 0
    all_cols = sorted(set(X_train.columns) | set(X_dev.columns) | set(X_test.columns))
    for col in all_cols:
        if col not in X_train.columns:
            X_train[col] = 0
        if col not in X_dev.columns:
            X_dev[col] = 0
        if col not in X_test.columns:
            X_test[col] = 0
    
    # Reorder columns to match
    X_train = X_train[all_cols]
    X_dev = X_dev[all_cols]
    X_test = X_test[all_cols]
    
    feature_names = list(all_cols)
    
    y_train = train['post_hire_score'].values
    
    print(f"  Feature matrix shape: {X_train.shape}")
    print(f"  Features: {len(feature_names)}")
    
    elapsed = time.time() - start_time
    print(f"  [Time: {elapsed:.1f}s]")
    
    # ============================================================
    # PHASE 4: Model Training & Validation
    # ============================================================
    print("\n[Phase 4] Training ensemble models...")
    
    models = train_ensemble(X_train.values, y_train, feature_names)
    
    # Validate on dev set (Ledger)
    print("\n  Validating on Ledger (dev set)...")
    dev_results = validate_on_dev(models, X_dev.values, dev, dev_winners, k=150)
    
    # Feature importance
    imp_df = get_feature_importance(models, feature_names)
    print("\n  Top 15 features by importance:")
    for _, row in imp_df.head(15).iterrows():
        print(f"    {row['feature']}: {row['importance']:.1f}")
    
    elapsed = time.time() - start_time
    print(f"\n  [Time: {elapsed:.1f}s]")
    
    # ============================================================
    # PHASE 5: Cycle-Specific Adjustments
    # ============================================================
    print("\n[Phase 5] Applying cycle-specific adjustments...")
    
    # Get base predictions for test set
    base_scores = predict_ensemble(models, X_test.values, feature_names)
    
    # Apply hard filters
    print("\n  Applying hard filters...")
    eligible_mask = apply_hard_filters(test, exclude_ids=dup_exclude_ids)
    
    # Apply cycle adjustments to all candidates (filtered later)
    print("\n  Applying insider intelligence adjustments...")
    adjusted_scores = apply_cycle_adjustments(test, base_scores)
    
    # Set ineligible candidates to -inf so they can't be selected
    adjusted_scores[~eligible_mask.values] = -np.inf
    
    elapsed = time.time() - start_time
    print(f"\n  [Time: {elapsed:.1f}s]")
    
    # ============================================================
    # PHASE 6: Generate Final Ranking & Submission
    # ============================================================
    print("\n[Phase 6] Generating final ranking...")
    
    submission = generate_final_ranking(test, adjusted_scores, k=500)
    
    # Write submission
    submission.to_csv('submission.csv', index=False)
    print(f"  Written submission.csv with {len(submission)} rows")
    
    # Verify format
    print("\n  Verifying submission format...")
    assert len(submission) == 500, f"Expected 500 rows, got {len(submission)}"
    assert list(submission.columns) == ['rank', 'candidate_id'], "Wrong columns"
    assert sorted(submission['rank'].tolist()) == list(range(1, 501)), "Rank not 1..500"
    assert submission['candidate_id'].duplicated().sum() == 0, "Duplicate candidates"
    test_ids = set(test_raw['candidate_id'].values)
    assert all(cid in test_ids for cid in submission['candidate_id']), "Unknown candidates"
    print("  ✓ All format checks passed!")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total runtime: {total_time:.1f}s")
    print(f"  Dev set overlap: {dev_results['overlap']} / 150")
    print(f"  Dev Precision@150: {dev_results['precision']:.4f}")
    print(f"  Submission: 500 candidates ranked in submission.csv")
    print("=" * 70)
    
    # Sanity check: print some stats about the shortlist
    shortlisted = test[test['candidate_id'].isin(set(submission['candidate_id']))]
    print(f"\n  Shortlist statistics:")
    print(f"    Avg tech assessment: {shortlisted['tech_assessment'].mean():.1f}")
    print(f"    Avg aptitude: {shortlisted['aptitude'].mean():.1f}")
    print(f"    Avg experience: {shortlisted['experience_years'].mean():.1f} years")
    if 'code_contributions' in shortlisted.columns:
        print(f"    Avg code contributions: {shortlisted['code_contributions'].mean():.1f}")
    print(f"    Non-traditional institute: {shortlisted['is_non_traditional'].mean()*100:.1f}%")
    print(f"    Old-boys network: {shortlisted['is_old_boys'].mean()*100:.1f}%")
    print(f"    Can join < 2 months: {shortlisted['can_join_2months'].mean()*100:.1f}%")
    print(f"    Fabrication score (avg): {shortlisted['fabrication_score'].mean():.3f}")


if __name__ == '__main__':
    main()
