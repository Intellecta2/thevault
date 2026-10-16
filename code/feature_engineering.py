"""
Feature Engineering Module (OPTIMIZED)
=========================================
Constructs a rich feature matrix for model training and prediction.
Groups features into:
  1. Core performance (tech_assessment, aptitude, rating, kpi, etc.)
  2. Career trajectory (progression, tenure, title level, stability)
  3. Education (degree level, institute tier)
  4. Compensation (CTC, hike ratio, market alignment)
  5. Availability (notice period, can_join)
  6. Profile quality (skills count, certifications, recruiter note)
  7. NLP features from recruiter_note
  8. Skill quality scoring
  9. Target-encoded features
 10. City tier & recruitment channel
 11. Fabrication signals
 12. Role encoding
"""

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Feature columns that exist in all datasets (train, dev, test)
COMMON_FEATURES = [
    # Core performance
    'tech_assessment',
    'aptitude',
    'rating',
    'kpi',
    'has_awards',
    'trainings_last_year',
    'training_hours',
    
    # Career trajectory
    'experience_years',
    'num_employers',
    'num_steps',
    'career_total_months',
    'avg_tenure_months',
    'max_title_level',
    'min_title_level',
    'progression_rate',
    'title_jump_count',
    
    # Education
    'degree_level',
    'is_iit',
    'is_old_boys',
    'is_top_institute',
    'is_non_traditional',
    
    # Compensation
    'current_ctc_lakhs',
    'expected_ctc_lakhs',
    'ctc_hike_ratio',
    
    # Availability
    'notice_period_days',
    'can_join_2months',
    
    # Profile quality
    'num_skills',
    'num_certifications',
    'has_recruiter_note',
    'overtime',
    'enrolled',
    'last_job_change_years',
    'company_size_num',
    
    # Demographics
    'age',
    'graduation_year',
    
    # Fabrication signals
    'fabrication_score',
]

# Role encoding (will be added dynamically)
ROLE_PREFIX = 'role_'

# ============================================================
# In-demand skills for quality scoring
# ============================================================
HIGH_VALUE_SKILLS = {
    # Languages
    'python', 'java', 'javascript', 'typescript', 'golang', 'go', 'rust', 'scala',
    'kotlin', 'swift', 'c++', 'cpp',
    # Data / ML
    'sql', 'spark', 'hadoop', 'kafka', 'airflow', 'tensorflow', 'pytorch',
    'pandas', 'numpy', 'scikit-learn', 'sklearn', 'ml', 'machine learning',
    'deep learning', 'nlp', 'computer vision', 'cv',
    # Cloud
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'k8s', 'terraform',
    'ci/cd', 'jenkins', 'github actions',
    # Web
    'react', 'angular', 'vue', 'node', 'nodejs', 'express', 'django',
    'flask', 'fastapi', 'spring', 'graphql', 'rest', 'api',
    # Data Engineering
    'etl', 'data pipeline', 'dbt', 'snowflake', 'redshift', 'bigquery',
    'postgresql', 'mongodb', 'redis', 'elasticsearch',
    # DevOps / Infra
    'linux', 'git', 'agile', 'scrum', 'microservices',
}

HIGH_VALUE_CERTS = {
    'aws', 'azure', 'gcp', 'google cloud', 'certified', 'professional',
    'associate', 'solutions architect', 'devops', 'kubernetes', 'ckad',
    'cka', 'istqb', 'pmp', 'scrum master', 'csm', 'six sigma',
    'data engineer', 'machine learning', 'tensorflow', 'deep learning',
}

# City tier mapping
TIER1_CITIES = {
    'mumbai', 'delhi', 'bangalore', 'bengaluru', 'hyderabad', 'chennai',
    'kolkata', 'pune', 'gurugram', 'gurgaon', 'noida', 'ghaziabad',
    'new delhi', 'madras',
}

TIER2_CITIES = {
    'ahmedabad', 'jaipur', 'lucknow', 'chandigarh', 'indore', 'bhopal',
    'nagpur', 'coimbatore', 'kochi', 'cochin', 'trivandrum', 'visakhapatnam',
    'vadodara', 'surat', 'mysore', 'mysuru', 'mangalore', 'mangaluru',
    'thiruvananthapuram', 'thrissur', 'bhubaneswar',
}

# Recruiter note keywords
POSITIVE_KEYWORDS = [
    'shipped', 'built', 'led', 'strong', 'excellent', 'impressive',
    'outstanding', 'talent', 'stellar', 'top', 'exceptional', 'great',
    'recommend', 'fast learner', 'proactive', 'passionate', 'driven',
    'innovative', 'expert', 'deep knowledge', 'solid', '1m+', 'million',
    'ownership', 'initiative', 'mentor',
]

NEGATIVE_KEYWORDS = [
    'lukewarm', 'concern', 'weak', 'poor', 'gap', 'inconsistent',
    'unreliable', 'late', 'slow', 'overconfident', 'arrogant',
    'not a team', 'cultural fit', 'red flag', 'caution', 'average',
    'mediocre', 'below', 'underperform',
]

NEUTRAL_KEYWORDS = [
    'prefers', 'hobby', 'cricket', 'music', 'travel', 'volunteer',
    'side project', 'open source', 'remote', 'hybrid', 'relocat',
]


def extract_recruiter_note_features(df):
    """Extract NLP features from recruiter_note column."""
    features = pd.DataFrame(index=df.index)
    
    notes = df['recruiter_note'].fillna('')
    
    # Basic text features
    features['note_length'] = notes.str.len()
    features['note_word_count'] = notes.apply(lambda x: len(str(x).split()) if x else 0)
    
    # Positive / negative keyword counts
    notes_lower = notes.str.lower()
    
    features['note_positive_count'] = notes_lower.apply(
        lambda x: sum(1 for kw in POSITIVE_KEYWORDS if kw in x) if x else 0
    )
    features['note_negative_count'] = notes_lower.apply(
        lambda x: sum(1 for kw in NEGATIVE_KEYWORDS if kw in x) if x else 0
    )
    features['note_sentiment'] = features['note_positive_count'] - features['note_negative_count']
    
    # Has shipped/built something (strong positive signal)
    features['note_has_shipped'] = notes_lower.apply(
        lambda x: 1 if any(kw in x for kw in ['shipped', 'built', 'launched', 'deployed', '1m+', 'million']) else 0
    )
    
    # Leadership signals
    features['note_has_leadership'] = notes_lower.apply(
        lambda x: 1 if any(kw in x for kw in ['led', 'mentor', 'managed', 'team lead', 'ownership']) else 0
    )
    
    return features


def score_skills_quality(df):
    """Score skills based on in-demand skill matching."""
    features = pd.DataFrame(index=df.index)
    
    skills_raw = df['skills'].fillna('')
    
    def count_high_value(skills_str):
        if not skills_str:
            return 0
        skills_lower = str(skills_str).lower()
        # Split by common delimiters
        tokens = re.split(r'[;,|]', skills_lower)
        tokens = [t.strip() for t in tokens if t.strip()]
        count = 0
        for token in tokens:
            for hv in HIGH_VALUE_SKILLS:
                if hv in token:
                    count += 1
                    break
        return count
    
    features['high_value_skill_count'] = skills_raw.apply(count_high_value)
    features['skill_quality_ratio'] = np.where(
        df['num_skills'] > 0,
        features['high_value_skill_count'] / df['num_skills'],
        0
    )
    
    return features


def score_certification_quality(df):
    """Score certifications based on value."""
    features = pd.DataFrame(index=df.index)
    
    certs_raw = df['certifications'].fillna('')
    
    def count_high_value_certs(cert_str):
        if not cert_str:
            return 0
        cert_lower = str(cert_str).lower()
        count = 0
        for kw in HIGH_VALUE_CERTS:
            if kw in cert_lower:
                count += 1
        return min(count, 5)  # cap
    
    features['high_value_cert_count'] = certs_raw.apply(count_high_value_certs)
    
    return features


def encode_city_tier(df):
    """Encode city into tiers."""
    features = pd.DataFrame(index=df.index)
    
    cities = df['current_city'].fillna('').str.strip().str.lower()
    
    features['is_tier1_city'] = cities.isin(TIER1_CITIES).astype(int)
    features['is_tier2_city'] = cities.isin(TIER2_CITIES).astype(int)
    features['is_tier3_city'] = (~cities.isin(TIER1_CITIES) & ~cities.isin(TIER2_CITIES) & (cities != '')).astype(int)
    
    return features


def compute_target_encodings(train_df, target_col='post_hire_score', n_folds=5):
    """Compute target encoding maps from train data using global mean.
    
    Returns dict of {column_name: {value: encoded_mean}}.
    """
    encodings = {}
    global_mean = train_df[target_col].mean()
    
    # Columns to target-encode
    encode_cols = ['applied_role', 'degree_canonical']
    
    for col in encode_cols:
        if col not in train_df.columns:
            continue
        # Smoothed target encoding: (count * mean + m * global_mean) / (count + m)
        m = 10  # smoothing parameter
        stats = train_df.groupby(col)[target_col].agg(['mean', 'count'])
        encoding = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
        encodings[col] = encoding.to_dict()
    
    encodings['_global_mean'] = global_mean
    return encodings


def apply_target_encodings(df, encodings):
    """Apply pre-computed target encodings to a dataframe."""
    features = pd.DataFrame(index=df.index)
    global_mean = encodings.get('_global_mean', 50.0)
    
    for col, mapping in encodings.items():
        if col.startswith('_'):
            continue
        if col in df.columns:
            features[f'te_{col}'] = df[col].map(mapping).fillna(global_mean)
    
    return features


def build_feature_matrix(df, target_encodings=None, fit_encoder=None):
    """Build the feature matrix from a cleaned + fraud-detected dataframe.
    
    Args:
        df: Cleaned dataframe with all parsed columns
        target_encodings: Pre-computed target encodings dict (from train)
        fit_encoder: Fitted LabelEncoder for applied_role (None = fit new one)
    
    Returns:
        (X, feature_names) where X is a DataFrame
    """
    # Start with common features
    feature_cols = [c for c in COMMON_FEATURES if c in df.columns]
    
    X = df[feature_cols].copy()
    
    # --- Role one-hot encoding ---
    if 'applied_role' in df.columns:
        role_dummies = pd.get_dummies(df['applied_role'], prefix='role', dtype=int)
        X = pd.concat([X, role_dummies], axis=1)
    
    # --- Recruitment channel one-hot ---
    if 'recruitment_channel' in df.columns:
        channel_dummies = pd.get_dummies(df['recruitment_channel'], prefix='channel', dtype=int)
        X = pd.concat([X, channel_dummies], axis=1)
    
    # --- Company type one-hot ---
    if 'company_type' in df.columns:
        ctype_dummies = pd.get_dummies(df['company_type'], prefix='ctype', dtype=int)
        X = pd.concat([X, ctype_dummies], axis=1)
    
    # --- Major one-hot ---
    if 'major' in df.columns:
        major_dummies = pd.get_dummies(df['major'], prefix='major', dtype=int)
        X = pd.concat([X, major_dummies], axis=1)
    
    # --- City tier features ---
    city_features = encode_city_tier(df)
    X = pd.concat([X, city_features], axis=1)
    
    # --- Recruiter note NLP features ---
    note_features = extract_recruiter_note_features(df)
    X = pd.concat([X, note_features], axis=1)
    
    # --- Skills quality features ---
    skill_features = score_skills_quality(df)
    X = pd.concat([X, skill_features], axis=1)
    
    # --- Certification quality ---
    cert_features = score_certification_quality(df)
    X = pd.concat([X, cert_features], axis=1)
    
    # --- Target-encoded features ---
    if target_encodings is not None:
        te_features = apply_target_encodings(df, target_encodings)
        X = pd.concat([X, te_features], axis=1)
    
    # ============================================================
    # INTERACTION FEATURES
    # ============================================================
    
    # Non-pedigree + high tech assessment (insider hint)
    if 'is_non_traditional' in X.columns and 'tech_assessment' in X.columns:
        X['non_trad_x_tech'] = X['is_non_traditional'] * X['tech_assessment']
    
    # Old boys network + tech assessment
    if 'is_old_boys' in X.columns and 'tech_assessment' in X.columns:
        X['old_boys_x_tech'] = X['is_old_boys'] * X['tech_assessment']
    
    # Experience per employer (stability)
    if 'experience_years' in X.columns and 'num_employers' in X.columns:
        X['exp_per_employer'] = np.where(
            X['num_employers'] > 0,
            X['experience_years'] / X['num_employers'],
            X['experience_years']
        )
    
    # CTC per experience year (compensation efficiency)
    if 'current_ctc_lakhs' in X.columns and 'experience_years' in X.columns:
        X['ctc_per_exp'] = np.where(
            X['experience_years'] > 0,
            X['current_ctc_lakhs'] / X['experience_years'],
            X['current_ctc_lakhs']
        )
    
    # Title level vs experience (inflation signal)
    if 'max_title_level' in X.columns and 'experience_years' in X.columns:
        X['title_per_exp'] = np.where(
            X['experience_years'] > 0,
            X['max_title_level'] / X['experience_years'],
            X['max_title_level']
        )
    
    # Training intensity
    if 'training_hours' in X.columns and 'trainings_last_year' in X.columns:
        X['hrs_per_training'] = np.where(
            X['trainings_last_year'] > 0,
            X['training_hours'] / X['trainings_last_year'],
            0
        )
    
    # --- Winner-analysis-inspired features ---
    
    # Composite performance score
    if all(c in X.columns for c in ['tech_assessment', 'aptitude', 'rating', 'kpi']):
        X['perf_composite'] = (
            X['tech_assessment'] / 100.0 *
            X['aptitude'] / 10.0 *
            X['rating'] / 5.0 *
            (X['kpi'] * 0.5 + 0.5)
        )
    
    # KPI × tech assessment interaction
    if 'kpi' in X.columns and 'tech_assessment' in X.columns:
        X['kpi_x_tech'] = X['kpi'] * X['tech_assessment']
    
    # KPI × rating interaction
    if 'kpi' in X.columns and 'rating' in X.columns:
        X['kpi_x_rating'] = X['kpi'] * X['rating']
    
    # Aptitude × tech
    if 'aptitude' in X.columns and 'tech_assessment' in X.columns:
        X['apt_x_tech'] = X['aptitude'] * X['tech_assessment']
    
    # Experience sweet spot (mid-career)
    if 'experience_years' in X.columns:
        X['exp_mid_career'] = np.where(
            (X['experience_years'] >= 5) & (X['experience_years'] <= 15), 1, 0
        )
    
    # CTC efficiency
    if 'current_ctc_lakhs' in X.columns and 'experience_years' in X.columns:
        X['ctc_efficiency'] = np.where(
            X['current_ctc_lakhs'] > 0,
            X['experience_years'] / X['current_ctc_lakhs'] * 10,
            0
        )
    
    # Skills per year of experience
    if 'num_skills' in X.columns and 'experience_years' in X.columns:
        X['skills_per_exp'] = np.where(
            X['experience_years'] > 0,
            X['num_skills'] / X['experience_years'],
            X['num_skills']
        )
    
    # Career stability
    if 'title_jump_count' in X.columns and 'avg_tenure_months' in X.columns:
        X['career_stability'] = (
            (X['title_jump_count'] == 0).astype(float) * 0.5 +
            np.clip(X['avg_tenure_months'] / 36.0, 0, 1) * 0.5
        )
    
    # ============================================================
    # NEW INTERACTION FEATURES
    # ============================================================
    
    # Rating × aptitude (both indicators of quality)
    if 'rating' in X.columns and 'aptitude' in X.columns:
        X['rating_x_aptitude'] = X['rating'] * X['aptitude']
    
    # Tech assessment squared (non-linear importance)
    if 'tech_assessment' in X.columns:
        X['tech_squared'] = X['tech_assessment'] ** 2 / 100.0
    
    # Aptitude squared
    if 'aptitude' in X.columns:
        X['aptitude_squared'] = X['aptitude'] ** 2 / 10.0
    
    # Experience bins (non-linear)
    if 'experience_years' in X.columns:
        X['exp_0_3'] = ((X['experience_years'] >= 0) & (X['experience_years'] < 3)).astype(int)
        X['exp_3_7'] = ((X['experience_years'] >= 3) & (X['experience_years'] < 7)).astype(int)
        X['exp_7_12'] = ((X['experience_years'] >= 7) & (X['experience_years'] < 12)).astype(int)
        X['exp_12_plus'] = (X['experience_years'] >= 12).astype(int)
    
    # CTC bins
    if 'current_ctc_lakhs' in X.columns:
        X['ctc_low'] = (X['current_ctc_lakhs'] < 15).astype(int)
        X['ctc_mid'] = ((X['current_ctc_lakhs'] >= 15) & (X['current_ctc_lakhs'] < 40)).astype(int)
        X['ctc_high'] = (X['current_ctc_lakhs'] >= 40).astype(int)
    
    # Age-experience alignment (should be roughly age - 22 ≈ experience)
    if 'age' in X.columns and 'experience_years' in X.columns:
        X['age_exp_gap'] = X['age'] - 22 - X['experience_years']
        X['age_exp_ratio'] = np.where(
            X['age'] > 20,
            X['experience_years'] / (X['age'] - 20),
            0
        )
    
    # Graduation recency
    if 'graduation_year' in X.columns:
        X['years_since_grad'] = 2026 - X['graduation_year']
        X['recent_grad'] = (X['years_since_grad'] <= 5).astype(int)
    
    # Tenure consistency (std of tenure — lower is more stable)
    # Already have avg_tenure_months; add more
    if 'career_total_months' in X.columns and 'num_steps' in X.columns:
        X['career_density'] = np.where(
            X['career_total_months'] > 0,
            X['num_steps'] / (X['career_total_months'] / 12.0 + 1),
            0
        )
    
    # Notice period normalized
    if 'notice_period_days' in X.columns:
        X['notice_immediate'] = (X['notice_period_days'] == 0).astype(int)
        X['notice_short'] = ((X['notice_period_days'] > 0) & (X['notice_period_days'] <= 30)).astype(int)
        X['notice_medium'] = ((X['notice_period_days'] > 30) & (X['notice_period_days'] <= 60)).astype(int)
        X['notice_long'] = (X['notice_period_days'] > 60).astype(int)
    
    # Skill + cert combined quality
    if 'high_value_skill_count' in X.columns and 'high_value_cert_count' in X.columns:
        X['total_quality_signals'] = X['high_value_skill_count'] + X['high_value_cert_count'] * 2
    
    # Awards × KPI × rating triple interaction
    if all(c in X.columns for c in ['has_awards', 'kpi', 'rating']):
        X['awards_kpi_rating'] = X['has_awards'] * X['kpi'] * X['rating']
    
    # Non-traditional × high aptitude
    if 'is_non_traditional' in X.columns and 'aptitude' in X.columns:
        X['non_trad_x_aptitude'] = X['is_non_traditional'] * X['aptitude']
    
    # Hike ratio bins (candidates expecting reasonable hike vs outrageous)
    if 'ctc_hike_ratio' in X.columns:
        X['reasonable_hike'] = ((X['ctc_hike_ratio'] >= 1.0) & (X['ctc_hike_ratio'] <= 1.5)).astype(int)
        X['moderate_hike'] = ((X['ctc_hike_ratio'] > 1.5) & (X['ctc_hike_ratio'] <= 2.5)).astype(int)
        X['aggressive_hike'] = (X['ctc_hike_ratio'] > 2.5).astype(int)
    
    # Training engagement score
    if 'trainings_last_year' in X.columns and 'training_hours' in X.columns:
        X['training_engagement'] = (
            np.clip(X['trainings_last_year'] / 3.0, 0, 1) * 0.5 +
            np.clip(X['training_hours'] / 100.0, 0, 1) * 0.5
        )
    
    # Replace infinities with NaN, then fill
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Fill remaining NaN with 0 for model
    X = X.fillna(0)
    
    feature_names = list(X.columns)
    return X, feature_names, None


def add_test_specific_features(X_test, test_df):
    """Add test-specific features (public_code_contributions).
    
    These features don't exist in train/dev data, so we add them
    as post-model adjustments rather than model features.
    
    Returns:
        Updated X_test with code contribution features
    """
    if 'code_contributions' in test_df.columns:
        X_test['code_contributions'] = test_df['code_contributions'].fillna(0).values
        X_test['high_code_contributor'] = (X_test['code_contributions'] >= 25).astype(int)
        X_test['very_high_code_contributor'] = (X_test['code_contributions'] >= 50).astype(int)
    
    return X_test
