"""
Data Cleaning & Parsing Module (ACCURATE & EFFICIENT)
======================================================
Parses and normalizes all messy fields in the Corporate Heist dataset.
Handles:
- CTC: USD conversion (at 83.5 INR/USD), Crores (100 LPA), Lakhs/LPA/L,
       and full Indian rupee amounts with commas (e.g., ₹12,59,000 -> 12.59 LPA)
- total_experience: years, months, fresher, ranges, >, <, +
- notice_period: immediate/0, serving notice days, days, months, weeks
- last_rating: 1-5 scale, text ratings ('Unsatisfactory' -> 1.0, 'Meets', 'Outstanding'),
               unrated new joiners -> NaN
- kpi_met: binary (1/0)
- currently_enrolled: 'Full time course' and 'Part time course' -> 1, 'no_enrollment' -> 0
- awards: binary (1/0)
- overtime_history: binary (1/0)
- technical_assessment: /100, %, missing values ('not taken', 'absent', '-')
- aptitude_score: /10, %, plain numbers
- last_job_change: 'never' -> 0.0, '>4' -> 5.0, plain numbers
- degree & degree_level: canonical mapping including 12th/Diploma (level 0)
- institute: robust punctuation-agnostic matching for all IITs,
             Old-Boys' Network classic IITs (Delhi, Bombay, Madras, Kanpur, Kharagpur),
             and Top Institutes (NITs, IIITs, BITS Pilani, IITs)
- skills: splits on ';', ',', and '|' (accounting for ~24% of candidates using '|')
- imputation: vectorized median by applied_role + global median fallback
"""

import re
import numpy as np
import pandas as pd


# Helper for high-efficiency unique value parsing
def _map_series(series, parse_fn):
    """Parse series efficiently via unique values lookup table."""
    uniques = series.dropna().unique()
    lut = {u: parse_fn(u) for u in uniques}
    return series.map(lut)


# ============================================================
# 1. NUMERIC FIELD PARSERS
# ============================================================

def parse_total_experience(val):
    """Parse total_experience into float years.
    
    Handles: '17.6 years', '8+ yrs', '>20', 'Fresher', '<1 year',
             '69 months', '2+ yrs', '5.8', etc.
    """
    if pd.isna(val):
        return np.nan
    val = str(val).strip().lower()
    
    # Fresher
    if 'fresher' in val or val == '0':
        return 0.0
    
    # <1 year
    if val.startswith('<1') or val == 'less than 1 year':
        return 0.5
    
    # >20 or 20+
    if val in ('>20', '20+', '20+ years', '20+ yrs'):
        return 22.0
    
    # "69 months" pattern
    m = re.match(r'^([\d.]+)\s*months?$', val)
    if m:
        return float(m.group(1)) / 12.0
    
    # "X+ yrs" or "X+ years" pattern
    m = re.match(r'^([\d.]+)\+?\s*(yrs?|years?)$', val)
    if m:
        return float(m.group(1))
    
    # "X years" or "X yrs"
    m = re.match(r'^([\d.]+)\s*(yrs?|years?)$', val)
    if m:
        return float(m.group(1))
    
    # ">X" pattern
    m = re.match(r'^>([\d.]+)$', val)
    if m:
        return float(m.group(1)) + 2.0
    
    # Plain number
    m = re.match(r'^([\d.]+)$', val)
    if m:
        return float(m.group(1))
    
    return np.nan


def parse_ctc(val):
    """Parse CTC into lakhs per annum (LPA).
    
    Handles:
    - USD salaries: '$31,230', '$17,788' -> converted to LPA via 83.5 INR/USD
    - Crores: '7.53 Cr', '10.18 Cr' -> 753.0 LPA
    - LPA/Lakh/Lac/L: '58.3 LPA', 'INR 29.1 lakh', '16.48L' -> 58.3, 29.1, 16.48 LPA
    - Full rupee amounts with commas: '₹12,59,000' -> 12.59 LPA, '₹41,20,000' -> 41.20 LPA
    - Plain numbers: '728000' -> 7.28 LPA, '28.6' -> 28.6 LPA
    """
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    
    # 1. USD: contains $
    if '$' in val:
        clean_num = re.sub(r'[^\d.]', '', val)
        if clean_num:
            return float(clean_num) * 83.5 / 100000.0
        return np.nan
    
    val_lower = val.lower()
    
    # 2. Crore: contains cr
    if 'cr' in val_lower:
        clean_num = re.search(r'([\d.]+)', val.replace(',', ''))
        if clean_num:
            return float(clean_num.group(1)) * 100.0
        return np.nan
    
    # 3. Contains LPA / lakh / lac / L suffix (e.g. '16.48L', '29.1 lakh', '58.3 LPA')
    if re.search(r'\b(?:lpa|lakhs?|lac)\b|[\d.]+\s*(?:lpa|lakhs?|lac|l)\b', val_lower):
        m = re.search(r'([\d.]+)\s*(?:lpa|lakhs?|lac|l)\b', val_lower)
        if m:
            return float(m.group(1))
    
    # 4. Remove currency symbols and non-numeric chars except '.'
    val_clean = re.sub(r'(?:inr|rs\.?|₹)\s*', '', val_lower)
    val_num_str = re.sub(r'[^\d.]', '', val_clean)
    if not val_num_str:
        return np.nan
    num = float(val_num_str)
    
    # Full rupee amounts (> 1000 e.g. 728000 or 1259000 from '₹12,59,000')
    if num > 1000:
        return num / 100000.0
    return num


def parse_notice_period(val):
    """Parse notice_period into days.
    
    Handles: '15 days', '1 month', '2 months', 'Immediate',
             'Available now', 'Immediate joiner', '0 days',
             'Serving notice - 30 days', etc.
    """
    if pd.isna(val):
        return np.nan
    val = str(val).strip().lower()
    
    # Immediate variants
    if any(x in val for x in ['immediate', 'available now', 'right away']):
        return 0
    
    # "Serving notice - X days"
    m = re.search(r'serving\s+notice\s*[-–]\s*(\d+)\s*days?', val)
    if m:
        return int(m.group(1))
    
    # "X days"
    m = re.match(r'^(\d+)\s*days?$', val)
    if m:
        return int(m.group(1))
    
    # "X month(s)"
    m = re.match(r'^(\d+)\s*months?$', val)
    if m:
        return int(m.group(1)) * 30
    
    # "X weeks"
    m = re.match(r'^(\d+)\s*weeks?$', val)
    if m:
        return int(m.group(1)) * 7
    
    return np.nan


def parse_aptitude_score(val):
    """Parse aptitude_score to float (out of 10).
    
    Handles: '5.7', '5.7/10', 'N/A', etc.
    """
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    
    # Remove "/10" suffix
    val = re.sub(r'\s*/\s*10$', '', val)
    
    # Handle percentage
    if '%' in val:
        val = re.sub(r'\s*%\s*$', '', val)
        try:
            score = float(val) / 10.0
            if 0 <= score <= 10:
                return score
        except:
            pass
    
    try:
        score = float(val)
        if 0 <= score <= 10:
            return score
    except (ValueError, TypeError):
        pass
    
    return np.nan


def parse_technical_assessment(val):
    """Parse technical_assessment to float (out of 100).
    
    Handles: '50', 'not taken', 'absent', 'N/A', '-', etc.
    """
    if pd.isna(val):
        return np.nan
    val = str(val).strip().lower()
    
    if val in ('not taken', 'absent', 'n/a', '-', 'na', 'none', ''):
        return np.nan
    
    # Remove "/100" suffix
    val = re.sub(r'\s*/\s*100$', '', val)
    
    # Handle percentage
    if '%' in val:
        val = re.sub(r'\s*%\s*$', '', val)
    
    try:
        score = float(val)
        if 0 <= score <= 100:
            return score
    except (ValueError, TypeError):
        pass
    
    return np.nan


def parse_last_rating(val):
    """Parse last_rating to float (1-5 scale).
    
    Handles: '3', '3/5', '3.0', 'Meets Expectations', 'Outstanding',
             'Exceeds Expectations', 'Below Expectations', 'Poor',
             'Unsatisfactory', 'New joiner - not rated', etc.
    """
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    
    # Text mappings
    text_map = {
        'outstanding': 5.0,
        'exceeds expectations': 4.0,
        'exceeds': 4.0,
        'meets expectations': 3.0,
        'meets': 3.0,
        'below expectations': 2.0,
        'below': 2.0,
        'needs improvement': 1.5,
        'poor': 1.0,
        'unsatisfactory': 1.0,
        'new joiner - not rated': np.nan,
    }
    
    val_lower = val.lower()
    if val_lower in text_map:
        return text_map[val_lower]
    
    # "X/5" pattern
    m = re.match(r'^([\d.]+)\s*/\s*5$', val)
    if m:
        return float(m.group(1))
    
    # Plain number
    try:
        score = float(val)
        if 0 <= score <= 5:
            return score
    except (ValueError, TypeError):
        pass
    
    return np.nan


def parse_kpi_met(val):
    """Parse kpi_met to binary (1/0).
    
    Handles: 'Y', 'Yes', 'yes', '1', 'TRUE', 'N', 'No', 'no', '0', 'FALSE', etc.
    """
    if pd.isna(val):
        return np.nan
    val = str(val).strip().lower()
    
    if val in ('y', 'yes', '1', 'true', 'met'):
        return 1
    if val in ('n', 'no', '0', 'false', 'not met'):
        return 0
    
    return np.nan


def parse_awards(val):
    """Parse awards to binary (1=has award, 0=no award)."""
    if pd.isna(val):
        return 0
    val = str(val).strip().lower()
    
    if val in ('0', '-', 'none', 'no', 'na', 'n/a', ''):
        return 0
    
    return 1


def parse_overtime(val):
    """Parse overtime_history to binary (1/0)."""
    if pd.isna(val):
        return np.nan
    val = str(val).strip().lower()
    
    if val in ('yes', 'y', '1', 'true'):
        return 1
    if val in ('no', 'n', '0', 'false'):
        return 0
    
    return np.nan


def parse_currently_enrolled(val):
    """Parse currently_enrolled to binary (1=enrolled, 0=not).
    
    Handles: 'no_enrollment', 'Full time course', 'Part time course', etc.
    """
    if pd.isna(val):
        return np.nan
    val = str(val).strip().lower()
    
    if val in ('no_enrollment', 'no', 'n', '0', 'false', 'not enrolled'):
        return 0
    if any(k in val for k in ('full time', 'part time', 'course', 'yes', 'y', '1', 'true', 'enrolled')):
        return 1
    
    return np.nan


def parse_public_code_contributions(val):
    """Parse public_code_contributions to integer count.
    
    Handles: '3', '~3', '10+', '3 merged PRs', 'not tracked', NaN, etc.
    """
    if pd.isna(val):
        return np.nan
    val = str(val).strip().lower()
    
    if val in ('not tracked', 'n/a', '-', 'none', ''):
        return np.nan
    
    # Extract numeric part
    m = re.search(r'(\d+)', val)
    if m:
        num = int(m.group(1))
        # Handle "10+" or "more than X"
        if '+' in val or 'more' in val:
            num += 5
        return num
    
    return np.nan


def parse_last_job_change(val):
    """Parse last_job_change to float years.
    
    Handles: 'never' -> 0.0 (first company), '>4' -> 5.0, '1', '2', etc.
    """
    if pd.isna(val):
        return np.nan
    val = str(val).strip().lower()
    
    if val == 'never':
        return 0.0
    if val in ('>4', '4+'):
        return 5.0
    
    try:
        return float(val)
    except (ValueError, TypeError):
        pass
    
    m = re.search(r'([\d.]+)', val)
    if m:
        return float(m.group(1))
    
    return np.nan


# ============================================================
# 2. CATEGORICAL FIELD NORMALIZERS
# ============================================================

DEGREE_MAP = {
    'bachelor of technology': 'BTech',
    'b.tech': 'BTech',
    'btech': 'BTech',
    'b.e.': 'BTech',
    'be': 'BTech',
    'b.tech.': 'BTech',
    'bachelor of engineering': 'BTech',
    'bca': 'BCA',
    'b.c.a': 'BCA',
    'b.sc': 'BSc',
    'bsc': 'BSc',
    'b.sc.': 'BSc',
    'bachelor of science': 'BSc',
    'mca': 'MCA',
    'm.c.a': 'MCA',
    'ms': 'MS',
    'm.s.': 'MS',
    'm.s': 'MS',
    'msc': 'MS',
    'm.sc': 'MS',
    'm.sc.': 'MS',
    'master of science': 'MS',
    'mba': 'MBA',
    'm.b.a': 'MBA',
    'm.b.a.': 'MBA',
    'master of business administration': 'MBA',
    'phd': 'PhD',
    'ph.d': 'PhD',
    'ph.d.': 'PhD',
    'doctorate': 'PhD',
    'm.tech': 'MTech',
    'mtech': 'MTech',
    'm.tech.': 'MTech',
    'master of technology': 'MTech',
    '12th': 'HighSchool',
    'diploma': 'Diploma',
    'polytechnic diploma': 'Diploma',
}

DEGREE_LEVEL_MAP = {
    'HighSchool': 0, 'Diploma': 0,
    'BTech': 1, 'BSc': 1, 'BCA': 1, 'BE': 1,
    'MCA': 2, 'MS': 2, 'MBA': 2, 'MTech': 2,
    'PhD': 3,
}


def normalize_degree(val):
    """Normalize degree to canonical form."""
    if pd.isna(val):
        return 'Unknown'
    val = str(val).strip().lower()
    return DEGREE_MAP.get(val, 'Other')


def get_degree_level(canonical_degree):
    """Get numeric degree level (0=school/diploma, 1=undergrad, 2=postgrad, 3=PhD)."""
    return DEGREE_LEVEL_MAP.get(canonical_degree, 1)


# ============================================================
# IIT / Institute normalization
# ============================================================

def _norm_inst(val):
    """Normalize institute string by stripping punctuation and collapsing spaces."""
    if pd.isna(val):
        return ''
    s = re.sub(r'[,\.\-–\(\)]', ' ', str(val).lower())
    return ' '.join(s.split())


OLD_BOYS_CITIES = {'delhi', 'bombay', 'madras', 'kanpur', 'kharagpur'}
OLD_BOYS_ABBRS = {'iitd', 'iit-d', 'iitb', 'iitm', 'iitk', 'iitkgp'}


def is_iit(val):
    """Check if institute is any IIT (excluding non-IITs like KIIT)."""
    if pd.isna(val):
        return False
    raw = str(val).strip().lower()
    if any(k in raw.split() for k in ['iit', 'iit-d', 'iitb', 'iitd', 'iitm', 'iitk', 'iitkgp']):
        return True
    norm = _norm_inst(val)
    if 'indian institute of technology' in norm:
        return True
    tokens = norm.split()
    return 'iit' in tokens


def is_old_boys_network(val):
    """Check if institute is in the old-boys' network (top 5 classic IITs)."""
    if pd.isna(val):
        return False
    raw = str(val).strip().lower()
    for abbr in OLD_BOYS_ABBRS:
        if abbr in raw:
            return True
    if is_iit(val):
        norm = _norm_inst(val)
        tokens = set(norm.split())
        if tokens & OLD_BOYS_CITIES:
            return True
    return False


def is_top_institute(val):
    """Check if institute is a recognized top-tier school (IIT, NIT, IIIT, BITS)."""
    if pd.isna(val):
        return False
    if is_iit(val):
        return True
    norm = _norm_inst(val)
    tokens = set(norm.split())
    if 'nit' in tokens or 'iiit' in tokens or 'bits' in tokens:
        return True
    if 'national institute of technology' in norm:
        return True
    if 'indian institute of information technology' in norm:
        return True
    if 'birla institute of technology and science' in norm:
        return True
    return False


def is_non_traditional_institute(val):
    """Check if institute is NOT a traditional top school."""
    return not is_top_institute(val)


# ============================================================
# Company size parsing
# ============================================================

COMPANY_SIZE_MAP = {
    '<10': 5, '1-10': 5, '10-49': 30, '10-50': 30, '10/49': 30,
    '50-99': 75, '50-100': 75, '100-499': 300, '100-500': 300,
    '500-999': 750, '500-1000': 750, '1000-4999': 3000, '1000-5000': 3000,
    '5000-9999': 7500, '5000-10000': 7500, '10000+': 15000, '10000': 15000,
}


def parse_company_size(val):
    """Parse company_size to numeric midpoint."""
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    
    if val in COMPANY_SIZE_MAP:
        return COMPANY_SIZE_MAP[val]
    
    nums = re.findall(r'\d+', val)
    if len(nums) >= 2:
        return (int(nums[0]) + int(nums[1])) / 2.0
    elif len(nums) == 1:
        return float(nums[0])
    
    return np.nan


# ============================================================
# 3. MASTER CLEANING FUNCTION
# ============================================================

def clean_dataframe(df, is_test=False):
    """Apply all cleaning and parsing to a dataframe efficiently.
    
    Args:
        df: Raw dataframe (train, dev, or test)
        is_test: If True, parse public_code_contributions
    
    Returns:
        Cleaned dataframe with parsed columns
    """
    df = df.copy()
    
    # --- Numeric fields (parsed via fast unique value lookup) ---
    df['experience_years'] = _map_series(df['total_experience'], parse_total_experience)
    df['current_ctc_lakhs'] = _map_series(df['current_ctc'], parse_ctc)
    df['expected_ctc_lakhs'] = _map_series(df['expected_ctc'], parse_ctc)
    df['notice_period_days'] = _map_series(df['notice_period'], parse_notice_period)
    df['aptitude'] = _map_series(df['aptitude_score'], parse_aptitude_score)
    df['tech_assessment'] = _map_series(df['technical_assessment'], parse_technical_assessment)
    df['rating'] = _map_series(df['last_rating'], parse_last_rating)
    df['last_job_change_years'] = _map_series(df['last_job_change'], parse_last_job_change)
    
    # --- Binary fields ---
    df['kpi'] = _map_series(df['kpi_met'], parse_kpi_met)
    df['has_awards'] = _map_series(df['awards'], parse_awards)
    df['overtime'] = _map_series(df['overtime_history'], parse_overtime)
    df['enrolled'] = _map_series(df['currently_enrolled'], parse_currently_enrolled)
    df['enrolled_full_time'] = df['currently_enrolled'].fillna('').str.lower().str.contains('full time').astype(int)
    
    # --- Categorical fields ---
    df['degree_canonical'] = _map_series(df['degree'], normalize_degree)
    df['degree_level'] = _map_series(df['degree_canonical'], get_degree_level)
    df['is_iit'] = _map_series(df['institute'], is_iit).fillna(False).astype(int)
    df['is_old_boys'] = _map_series(df['institute'], is_old_boys_network).fillna(False).astype(int)
    df['is_top_institute'] = _map_series(df['institute'], is_top_institute).fillna(False).astype(int)
    df['is_non_traditional'] = (df['is_top_institute'] == 0).astype(int)
    df['company_size_num'] = _map_series(df['company_size'], parse_company_size)
    
    # --- Test-only: public code contributions ---
    if is_test and 'public_code_contributions' in df.columns:
        df['code_contributions'] = _map_series(
            df['public_code_contributions'],
            parse_public_code_contributions
        )
    
    # --- CTC ratio ---
    df['ctc_hike_ratio'] = np.where(
        (df['current_ctc_lakhs'] > 0) & df['current_ctc_lakhs'].notna(),
        df['expected_ctc_lakhs'] / df['current_ctc_lakhs'],
        np.nan
    )
    
    # --- Skills count (handling ';', ',', and '|' delimiters) ---
    df['num_skills'] = df['skills'].apply(
        lambda x: len([s for s in re.split(r'[;,|]', str(x)) if s.strip()]) if pd.notna(x) else 0
    )
    
    # --- Certifications count ---
    df['num_certifications'] = df['certifications'].apply(
        lambda x: len([s for s in re.split(r'[;,]', str(x)) if s.strip()]) if pd.notna(x) else 0
    )
    
    # --- Has recruiter note ---
    df['has_recruiter_note'] = df['recruiter_note'].notna().astype(int)
    
    # --- Can join within 2 months ---
    df['can_join_2months'] = (df['notice_period_days'] <= 60).astype(int)
    
    return df


def impute_missing(df, reference_df=None):
    """Impute missing values using median per applied_role vectorially.
    
    Args:
        df: Dataframe to impute
        reference_df: Optional reference for computing medians (e.g., train)
    
    Returns:
        Dataframe with imputed numeric columns
    """
    df = df.copy()
    ref = reference_df if reference_df is not None else df
    
    numeric_cols = [
        'experience_years', 'current_ctc_lakhs', 'expected_ctc_lakhs',
        'notice_period_days', 'aptitude', 'tech_assessment', 'rating',
        'kpi', 'overtime', 'enrolled', 'age', 'last_job_change_years',
        'company_size_num', 'ctc_hike_ratio',
    ]
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
        if df[col].isna().any():
            role_medians = ref.groupby('applied_role')[col].median()
            global_median = ref[col].median()
            # Vectorized fill: role median first, global median fallback
            df[col] = df[col].fillna(df['applied_role'].map(role_medians)).fillna(global_median)
    
    return df
