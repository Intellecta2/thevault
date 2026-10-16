"""
Fabrication & Duplicate Detection Module
==========================================
Detects fabricated profiles and duplicate candidates.

Fabrication signals:
- Experience vs age impossible mismatch
- Title inflation (senior titles too early in career)
- Career path duration vs reported experience mismatch
- Impossibly perfect profiles

Duplicate detection:
- Email normalization + exact match
- Phone normalization + exact match
- Fuzzy name matching with same graduation_year
"""

import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz


# ============================================================
# 1. CAREER PATH PARSING
# ============================================================

def parse_career_path(val):
    """Parse career_path into list of (title, duration_months).
    
    Handles formats:
    - "Title [23 mo] > Title [34 mo]"
    - "Title (1.0y) -> Title (2.4y)"
    - "Title (1.5 yrs) -> Title (0.8 yrs)"
    """
    if pd.isna(val):
        return []
    
    val = str(val).strip()
    entries = []
    
    # Split by common delimiters: >, ->, →, |, –>
    parts = re.split(r'\s*(?:->|→|–>|>|\|)\s*', val)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Try "[X mo]" format
        m = re.search(r'\[(\d+)\s*mo\]', part)
        if m:
            months = int(m.group(1))
            title = re.sub(r'\s*\[\d+\s*mo\]', '', part).strip()
            entries.append((title, months))
            continue
        
        # Try "(X.Xy)" or "(X.X yrs)" format
        m = re.search(r'\(([\d.]+)\s*(?:y|yrs?|years?)\)', part)
        if m:
            years = float(m.group(1))
            months = int(years * 12)
            title = re.sub(r'\s*\([\d.]+\s*(?:y|yrs?|years?)\)', '', part).strip()
            entries.append((title, months))
            continue
        
        # No duration found — add title with 0 months
        entries.append((part, 0))
    
    return entries


# Title seniority levels
TITLE_LEVELS = {
    'intern': 0, 'trainee': 0, 'apprentice': 0, 'fresher': 0,
    'junior': 1, 'associate': 1, 'entry': 1,
    'engineer': 2, 'developer': 2, 'analyst': 2, 'scientist': 2, 'designer': 2,
    'senior': 3, 'sr.': 3, 'sr': 3,
    'lead': 4, 'staff': 4, 'principal': 4,
    'manager': 5, 'head': 5,
    'director': 6,
    'vp': 7, 'vice president': 7,
    'cto': 8, 'ceo': 8, 'cfo': 8, 'coo': 8,
}


def get_title_level(title):
    """Map a job title string to a seniority level (0-8)."""
    if pd.isna(title):
        return 2  # default mid-level
    title_lower = str(title).strip().lower()
    
    # Check each keyword — return highest matching level
    max_level = 2  # default
    for keyword, level in TITLE_LEVELS.items():
        if keyword in title_lower:
            max_level = max(max_level, level)
    
    return max_level


def analyze_career_path(val):
    """Analyze career path and return trajectory metrics.
    
    Returns dict with:
    - num_steps: Number of career positions
    - total_months: Sum of months across all positions
    - avg_tenure_months: Average time per position
    - max_title_level: Highest seniority achieved
    - min_title_level: Starting seniority
    - progression_rate: Title level gain per year
    - title_inflation: True if suspicious rapid escalation
    """
    entries = parse_career_path(val)
    
    if not entries:
        return {
            'num_steps': 0, 'career_total_months': 0,
            'avg_tenure_months': 0, 'max_title_level': 2,
            'min_title_level': 2, 'progression_rate': 0,
            'title_jump_count': 0,
        }
    
    num_steps = len(entries)
    total_months = sum(m for _, m in entries)
    avg_tenure = total_months / num_steps if num_steps > 0 else 0
    
    levels = [get_title_level(t) for t, _ in entries]
    max_level = max(levels)
    min_level = levels[0] if levels else 2
    
    # Progression rate: level gain per year
    total_years = total_months / 12.0 if total_months > 0 else 1.0
    progression_rate = (max_level - min_level) / total_years if total_years > 0 else 0
    
    # Count large jumps (≥2 levels in one step)
    title_jump_count = 0
    for i in range(1, len(levels)):
        if levels[i] - levels[i-1] >= 2:
            title_jump_count += 1
    
    return {
        'num_steps': num_steps,
        'career_total_months': total_months,
        'avg_tenure_months': avg_tenure,
        'max_title_level': max_level,
        'min_title_level': min_level,
        'progression_rate': progression_rate,
        'title_jump_count': title_jump_count,
    }


# ============================================================
# 2. FABRICATION DETECTION
# ============================================================

def detect_fabrication(df):
    """Flag fabricated profiles based on multiple inconsistency signals.
    
    Returns:
        DataFrame with 'fabrication_score' (0-1) and 'is_fabricated' columns.
    """
    df = df.copy()
    
    # Initialize fabrication score
    fab_score = pd.Series(0.0, index=df.index)
    
    # --- Rule 1: Experience vs Age mismatch ---
    # Cannot have more experience than (age - 18)
    if 'experience_years' in df.columns and 'age' in df.columns:
        max_possible_exp = df['age'] - 18
        exp_exceeds = (df['experience_years'] > max_possible_exp + 2)  # 2 year tolerance
        fab_score += exp_exceeds.fillna(False).astype(float) * 0.4
    
    # --- Rule 2: Graduation year vs experience ---
    # Experience should roughly equal (current_year - graduation_year)
    if 'experience_years' in df.columns and 'graduation_year' in df.columns:
        current_year = 2026
        expected_exp = current_year - df['graduation_year']
        exp_gap = abs(df['experience_years'] - expected_exp)
        fab_score += (exp_gap > 8).fillna(False).astype(float) * 0.2
    
    # --- Rule 3: Title inflation (insider hint!) ---
    # "Head of something 3 years out of college goes straight to the bin"
    if 'experience_years' in df.columns:
        current_level = df['current_title'].apply(get_title_level)
        
        # VP/Director/Head with < 5 years experience
        title_inflation = (
            (current_level >= 5) & (df['experience_years'] < 5)
        )
        fab_score += title_inflation.fillna(False).astype(float) * 0.5
        
        # Senior/Lead with < 2 years
        mild_inflation = (
            (current_level >= 4) & (df['experience_years'] < 3)
        )
        fab_score += mild_inflation.fillna(False).astype(float) * 0.2
    
    # --- Rule 4: Career path vs experience mismatch ---
    career_metrics = df['career_path'].apply(analyze_career_path)
    career_df = pd.DataFrame(career_metrics.tolist(), index=df.index)
    
    if 'experience_years' in df.columns:
        career_exp_years = career_df['career_total_months'] / 12.0
        exp_mismatch = abs(career_exp_years - df['experience_years'])
        fab_score += (exp_mismatch > 5).fillna(False).astype(float) * 0.2
    
    # --- Rule 5: Rapid career progression (from career path analysis) ---
    fab_score += (career_df['progression_rate'] > 1.5).astype(float) * 0.3
    fab_score += (career_df['title_jump_count'] >= 2).astype(float) * 0.2
    
    # --- Rule 6: CTC anomalies ---
    if 'ctc_hike_ratio' in df.columns:
        extreme_hike = df['ctc_hike_ratio'] > 5  # Expecting 5x salary
        fab_score += extreme_hike.fillna(False).astype(float) * 0.15
    
    # Normalize to 0-1
    fab_score = fab_score.clip(0, 1)
    
    df['fabrication_score'] = fab_score
    df['is_fabricated'] = (fab_score >= 0.5).astype(int)
    
    # Add career path metrics to dataframe
    for col in career_df.columns:
        df[col] = career_df[col]
    
    return df


# ============================================================
# 3. DUPLICATE DETECTION
# ============================================================

def normalize_email(email):
    """Normalize email for dedup matching."""
    if pd.isna(email):
        return None
    return str(email).strip().lower()


def normalize_phone(phone):
    """Normalize phone number for dedup matching."""
    if pd.isna(phone):
        return None
    # Keep only digits
    digits = re.sub(r'\D', '', str(phone))
    # Remove country code if present
    if digits.startswith('91') and len(digits) > 10:
        digits = digits[2:]
    # Take last 10 digits
    if len(digits) >= 10:
        return digits[-10:]
    return digits


def detect_duplicates(df):
    """Detect duplicate candidates (same person, different entries).
    
    Returns:
        DataFrame with 'duplicate_group' column (group ID, -1 if unique)
        and 'is_duplicate' flag.
    """
    df = df.copy()
    
    # Normalize contact info
    df['email_norm'] = df['email'].apply(normalize_email)
    df['phone_norm'] = df['phone'].apply(normalize_phone)
    
    # Union-Find for grouping
    n = len(df)
    parent = list(range(n))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Build index for quick lookups
    email_to_idx = {}
    phone_to_idx = {}
    
    for i in range(n):
        # Email matching
        email = df.iloc[i]['email_norm']
        if email and email in email_to_idx:
            union(i, email_to_idx[email])
        elif email:
            email_to_idx[email] = i
        
        # Phone matching
        phone = df.iloc[i]['phone_norm']
        if phone and phone in phone_to_idx:
            union(i, phone_to_idx[phone])
        elif phone:
            phone_to_idx[phone] = i
    
    # Assign group IDs
    groups = [find(i) for i in range(n)]
    df['duplicate_group'] = groups
    
    # Mark duplicates (groups with more than one member)
    group_counts = pd.Series(groups).value_counts()
    duplicate_groups = set(group_counts[group_counts > 1].index)
    df['is_duplicate'] = df['duplicate_group'].isin(duplicate_groups).astype(int)
    
    # Clean up temp columns
    df.drop(columns=['email_norm', 'phone_norm'], inplace=True)
    
    return df


def resolve_duplicates(df, score_col=None):
    """For duplicate groups, keep the better profile.
    
    If score_col is provided, keep the one with higher predicted score.
    Otherwise, keep the one with more complete data and higher tech assessment.
    
    Returns:
        Set of candidate_ids to EXCLUDE (inferior duplicates).
    """
    exclude_ids = set()
    
    if 'is_duplicate' not in df.columns:
        return exclude_ids
    
    dup_df = df[df['is_duplicate'] == 1]
    
    for group_id in dup_df['duplicate_group'].unique():
        group = dup_df[dup_df['duplicate_group'] == group_id]
        
        if len(group) <= 1:
            continue
        
        # Score each profile by completeness and quality
        if score_col and score_col in group.columns:
            best_idx = group[score_col].idxmax()
        else:
            # Heuristic: prefer higher tech_assessment, more complete data
            completeness = group.notna().sum(axis=1)
            if 'tech_assessment' in group.columns:
                tech = group['tech_assessment'].fillna(0)
            elif 'technical_assessment' in group.columns:
                tech = pd.to_numeric(group['technical_assessment'], errors='coerce').fillna(0)
            else:
                tech = 0
            combined = completeness + tech * 10  # weight tech assessment
            best_idx = combined.idxmax()
        
        # Exclude all but the best
        for idx in group.index:
            if idx != best_idx:
                exclude_ids.add(df.loc[idx, 'candidate_id'])
    
    return exclude_ids
