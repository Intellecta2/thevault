"""
Cycle-Specific Adjustments Module
====================================
Applies insider intelligence as post-model score adjustments
to account for the regime change between old and new hiring cycles.

Key adjustments (from insider voice note):
1. Hard filter: notice_period > 2 months → excluded
2. Hard filter: fabricated profiles → excluded
3. Hard filter: duplicate candidates → keep only best entry
4. Boost: high public_code_contributions (fast-track signal)
5. Heavy penalty: title inflation (instant rejection)
6. Boost: non-pedigree college + high technical_assessment
7. Mild boost: old-boys' network colleges
8. De-weight: old biases (big metros, big employers, proper degrees)
"""

import numpy as np
import pandas as pd


def apply_hard_filters(df, exclude_ids=None):
    """Apply hard exclusion filters.
    
    Removes:
    1. Candidates with notice_period > 60 days (can't join within 2 months)
    2. Fabricated profiles
    3. Duplicate candidates (inferior entries)
    
    Args:
        df: DataFrame with all parsed columns
        exclude_ids: Set of candidate_ids to exclude (from duplicate resolution)
    
    Returns:
        Boolean mask (True = eligible, False = excluded)
    """
    eligible = pd.Series(True, index=df.index)
    
    # 1. Notice period > 2 months → EXCLUDE
    # Insider: "Anyone who can't join inside two months is dead to them"
    if 'notice_period_days' in df.columns:
        too_slow = df['notice_period_days'] > 60
        n_excluded = too_slow.sum()
        eligible &= ~too_slow
        print(f"  Excluded {n_excluded} candidates with notice > 60 days")
    
    # 2. Fabricated profiles → EXCLUDE
    if 'is_fabricated' in df.columns:
        n_faked = (df['is_fabricated'] == 1).sum()
        eligible &= (df['is_fabricated'] == 0)
        print(f"  Excluded {n_faked} fabricated profiles")
    
    # 3. Duplicate candidates → EXCLUDE inferior entries
    if exclude_ids:
        dup_mask = df['candidate_id'].isin(exclude_ids)
        n_duped = dup_mask.sum()
        eligible &= ~dup_mask
        print(f"  Excluded {n_duped} duplicate entries")
    
    total_excluded = (~eligible).sum()
    print(f"  Total excluded: {total_excluded} / {len(df)}")
    print(f"  Remaining eligible: {eligible.sum()}")
    
    return eligible


def apply_cycle_adjustments(df, base_scores):
    """Apply cycle-specific score adjustments based on insider intelligence.
    
    These adjustments modify the model's base prediction to account for
    the known regime changes in this hiring cycle.
    
    Args:
        df: DataFrame with parsed columns (including test-specific features)
        base_scores: numpy array of ensemble model scores
    
    Returns:
        adjusted_scores: numpy array of adjusted scores
    """
    scores = base_scores.copy()
    
    # ============================================================
    # ADJUSTMENT A: Public Code Contributions (test.csv only)
    # ============================================================
    # Insider: "dozens of merged contributions or more get fast-tracked,
    # even if the rest of the profile is only good rather than great"
    if 'code_contributions' in df.columns:
        pcc = df['code_contributions'].fillna(0).values
        
        # Massive boost for sustained contributors (≥30)
        very_high = pcc >= 30
        scores[very_high] *= 1.35
        
        # Significant boost for decent contributors (≥15)
        high = (pcc >= 15) & (pcc < 30)
        scores[high] *= 1.15
        
        # Slight boost for any contributions (≥5)
        some = (pcc >= 5) & (pcc < 15)
        scores[some] *= 1.05
        
        print(f"  Code contribution boosts: "
              f"≥30: {very_high.sum()}, ≥15: {high.sum()}, ≥5: {some.sum()}")
    
    # ============================================================
    # ADJUSTMENT B: Title Inflation Penalty
    # ============================================================
    # Insider: "Head of something 3 years out of college goes straight
    # to the bin, however shiny the rest looks"
    if 'max_title_level' in df.columns and 'experience_years' in df.columns:
        title_level = df['max_title_level'].values
        exp = df['experience_years'].values
        
        # Severe inflation: Director+ with <5 years
        severe = (title_level >= 6) & (exp < 5)
        scores[severe] *= 0.15
        
        # Moderate inflation: Manager/Head with <5 years
        moderate = (title_level >= 5) & (exp < 5) & ~severe
        scores[moderate] *= 0.25
        
        # Mild inflation: Lead with <3 years
        mild = (title_level >= 4) & (exp < 3) & ~severe & ~moderate
        scores[mild] *= 0.40
        
        n_penalized = severe.sum() + moderate.sum() + mild.sum()
        print(f"  Title inflation penalties: {n_penalized} candidates "
              f"(severe: {severe.sum()}, moderate: {moderate.sum()}, mild: {mild.sum()})")
    
    # ============================================================
    # ADJUSTMENT C: Non-Pedigree + High Tech Assessment Boost
    # ============================================================
    # Insider: "Their best people this cycle came from colleges they had
    # never hired from before. Candidates who topped the technical
    # assessment and genuinely fit the role."
    if 'is_non_traditional' in df.columns and 'tech_assessment' in df.columns:
        non_trad = df['is_non_traditional'].values == 1
        tech = df['tech_assessment'].values
        
        # Non-traditional + very high tech (≥80)
        star_nontrad = non_trad & (tech >= 80)
        scores[star_nontrad] *= 1.20
        
        # Non-traditional + high tech (≥65)
        good_nontrad = non_trad & (tech >= 65) & (tech < 80)
        scores[good_nontrad] *= 1.08
        
        print(f"  Non-pedigree boosts: top tech: {star_nontrad.sum()}, "
              f"high tech: {good_nontrad.sum()}")
    
    # ============================================================
    # ADJUSTMENT D: Old-Boys' Network (Still Active)
    # ============================================================
    # Insider: "a handful of colleges where Nightingale's leadership
    # studied still get a leg up, same as always"
    if 'is_old_boys' in df.columns:
        old_boys = df['is_old_boys'].values == 1
        scores[old_boys] *= 1.05
        print(f"  Old-boys' network boost: {old_boys.sum()} candidates")
    
    # ============================================================
    # ADJUSTMENT E: De-weight Old Panel Biases
    # ============================================================
    # Insider: "Big-name colleges, candidates from the big metros,
    # glowing referrals, big-brand employers, 'proper' degrees,
    # CVs without a single gap: the old panel quietly inflated all of it.
    # The new panel ignores every one of those."
    #
    # This is partially handled by the model already learning from
    # historical data. But we explicitly de-weight metro advantage
    # and any remaining prestige bias (except old-boys' network).
    #
    # For IIT candidates who are NOT in the old-boys' network,
    # remove the prestige boost the model may have learned.
    if 'is_iit' in df.columns and 'is_old_boys' in df.columns:
        iit_not_oldboys = (df['is_iit'].values == 1) & (df['is_old_boys'].values == 0)
        scores[iit_not_oldboys] *= 0.95  # Slight reduction
        print(f"  IIT (non-old-boys) de-boost: {iit_not_oldboys.sum()} candidates")
    
    return scores


def generate_final_ranking(df, adjusted_scores, k=500):
    """Generate the final ranked shortlist of top-k candidates.
    
    Args:
        df: DataFrame with candidate_id
        adjusted_scores: numpy array of adjusted scores
        k: Number of candidates to select (default 500)
    
    Returns:
        submission DataFrame with columns [rank, candidate_id]
    """
    df = df.copy()
    df['final_score'] = adjusted_scores
    
    # Sort by score descending
    df_sorted = df.sort_values('final_score', ascending=False)
    
    # Take top k
    top_k = df_sorted.head(k)
    
    # Create submission
    submission = pd.DataFrame({
        'rank': range(1, k + 1),
        'candidate_id': top_k['candidate_id'].values,
    })
    
    return submission
