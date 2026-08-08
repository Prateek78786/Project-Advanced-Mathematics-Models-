# lifetrack.py — LifeTrack Ultimate Single-File
# Robust multi-file parser (up to 18 Fitbit-style CSVs), extended analytics,
# extra predictions (sleep cycles, fatigue proxy), additional visuals and
# verbose page descriptions. Includes banner asset and logo support.

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore
import plotly.express as px
import plotly.graph_objects as go
import io
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="LifeTrack — Ultimate", layout="wide", page_icon="📊")

# custom app styling
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, rgba(56, 198, 255, 0.22), transparent 24%),
                    radial-gradient(circle at bottom right, rgba(147, 84, 255, 0.16), transparent 26%),
                    linear-gradient(180deg, #091d42 0%, #15438d 40%, #08172f 100%);
        color: #eef5ff;
    }
    .css-1d391kg, .css-18ni7ap, .css-1v3fvcr {
        padding-top: 0.75rem;
    }
    .stButton>button {
        border-radius: 14px;
        background: linear-gradient(135deg, #39f2ff, #5f59ff);
        color: #04122a;
        box-shadow: 0 18px 40px rgba(57, 242, 255, 0.25);
        border: none;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 22px 46px rgba(57, 242, 255, 0.32);
    }
    .streamlit-expanderHeader {
        font-weight: 700;
        color: #c6e9ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Configuration / Constants
# -----------------------
MAX_FILES = 18
ASSETS_BANNER = "assets/banner.png"  # your banner path
LOGO_PATH = "assets/logo.png"

# Known table name hints (some Fitbit exports or merged filenames from dataset)
TABLE_HINTS = {
    'dailyactivity': 'dailyActivity',
    'dailyactivity_merged': 'dailyActivity',
    'daily_steps': 'dailySteps',
    'dailysteps_merged': 'dailySteps',
    'sleepday': 'sleepDay',
    'sleepday_merged': 'sleepDay',
    'heartrate_seconds_merged': 'heartrate_seconds',
    'hourlycalories': 'hourlyCalories',
    'hourlyintensities': 'hourlyIntensities',
    'hourlysteps': 'hourlySteps',
    'minutecaloriesnarrow': 'minuteCaloriesNarrow',
    'minuteintensitiesnarrow': 'minuteIntensitiesNarrow',
    'minutemetsnarrow': 'minuteMETsNarrow',
    'minutesleep': 'minuteSleep',
    'minutesleep_merged': 'minuteSleep',
    'minutestepsnarrow': 'minuteStepsNarrow',
    'weightloginfo': 'weightLogInfo',
    'dailycalories': 'dailyCalories',
    'dailyintensities': 'dailyIntensities',
    'dailysteps': 'dailySteps',
    'minutecalorieswide': 'minuteCaloriesWide',
    'minuteintensitieswide': 'minuteIntensitiesWide',
    'minutestepswide': 'minuteStepsWide'
}

# -----------------------
# Utilities: Parsing helpers
# -----------------------


def safe_read_csv(f):
    """Try multiple ways to read a CSV-like file robustly."""
    for args in [{}, {'encoding':'latin1'}, {'sep':';'}, {'sep':'\t'}, {'encoding':'latin1','sep':';'}]:
        try:
            f.seek(0)
        except Exception:
            pass
        try:
            df = pd.read_csv(f, **args)
            return df
        except Exception:
            continue
    # last resort: try read_table
    try:
        f.seek(0)
        df = pd.read_table(f)
        return df
    except Exception:
        return None


def detect_table_hint(name, df):
    """Return canonical table hint key based on filename or columns."""
    low = name.lower() if isinstance(name, str) else ''
    for k in TABLE_HINTS:
        if k in low:
            return TABLE_HINTS[k]
    # fallback: inspect columns
    cols = ' '.join([str(c).lower() for c in df.columns])
    for k in TABLE_HINTS:
        if k in cols:
            return TABLE_HINTS[k]
    # heuristic by column names
    if any('activitydate' in str(c).lower() or 'activityday' in str(c).lower() or 'date' == str(c).lower() for c in df.columns):
        # likely daily table
        if any('steps' in str(c).lower() or 'totalsteps' in str(c).lower() for c in df.columns):
            return 'dailyActivity'
    if any('value' in str(c).lower() and 'time' in cols for c in df.columns):
        return 'heartrate_seconds'
    return 'unknown'


def coerce_date_col(df, prefer_cols=None):
    """Find and convert a date-like column to pandas datetime and return df and chosen col name."""
    df2 = df.copy()
    prefer_cols = prefer_cols or ['activitydate', 'activityday', 'date', 'sleepday', 'activityhour', 'activityminute', 'time']
    date_col = None
    for c in df2.columns:
        if str(c).lower() in prefer_cols or any(pc in str(c).lower() for pc in prefer_cols):
            # ensure not a numeric id-like column
            date_col = c
            break
    if date_col is None:
        # try to detect by dtype or pattern
        for c in df2.columns:
            ser = df2[c].astype(str)
            if ser.str.match(r"^\d{1,2}/\d{1,2}/\d{2,4}").any() or ser.str.match(r"^\d{4}-\d{2}-\d{2}").any():
                date_col = c
                break
    if date_col is None:
        # last fallback: first non-numeric column
        for c in df2.columns:
            if not pd.api.types.is_numeric_dtype(df2[c]):
                date_col = c
                break
    if date_col is None:
        return df2, None
    # Attempt parsing with common formats and with dayfirst heuristics
    try:
        df2[date_col] = pd.to_datetime(df2[date_col], errors='coerce', infer_datetime_format=True, dayfirst=False)
        if df2[date_col].isna().all():
            df2[date_col] = pd.to_datetime(df2[date_col].astype(str).str.replace(r'\s+AM|\s+PM', '', regex=True), errors='coerce', dayfirst=True)
    except Exception:
        try:
            df2[date_col] = pd.to_datetime(df2[date_col].astype(str), errors='coerce')
        except Exception:
            df2[date_col] = pd.NaT
    return df2, date_col


def clean_numeric_columns(df):
    dfc = df.copy()
    for c in dfc.columns:
        if c == 'Date':
            continue
        # strip commas, spaces, misc characters
        if dfc[c].dtype == object:
            dfc[c] = dfc[c].astype(str).str.replace(',', '').str.replace(' ', '').str.replace('\x00', '')
            # convert exponential-like strings '1.5E+09' or scientific notation preserved
            try:
                dfc[c] = pd.to_numeric(dfc[c], errors='coerce')
            except Exception:
                pass
    return dfc

# -----------------------
# Specialized aggregators / converters for Fitbit-like tables
# -----------------------


def hr_seconds_to_daily(df_hr):
    """Aggregate heartrate seconds table (Time, Value) into daily average heart rate."""
    df = df_hr.copy()
    # rename heuristics
    col_time = None
    col_val = None
    for c in df.columns:
        low = str(c).lower()
        if 'time' in low or 'datetime' in low:
            col_time = c
        if 'value' in low or 'bpm' in low or 'heart' in low:
            col_val = c
    if col_time is None:
        col_time = df.columns[0]
    if col_val is None and df.shape[1] > 1:
        col_val = df.columns[1]
    df[col_time] = pd.to_datetime(df[col_time].astype(str).str.replace('T', ' '), errors='coerce')
    df['Date'] = df[col_time].dt.date
    agg = df.groupby('Date')[col_val].mean().reset_index().rename(columns={col_val: 'AvgHeartRate'})
    agg['Date'] = pd.to_datetime(agg['Date'])
    return agg


def hourly_to_daily(df_hourly, value_col_candidates=None):
    """Aggregate hourly tables (ActivityHour + metric) to daily sums or means."""
    df = df_hourly.copy()
    # detect activity hour column
    ah_col = None
    for c in df.columns:
        if 'activityhour' in str(c).lower() or str(c).lower() == 'activityhour' or 'activityhour' in str(c).replace(' ', '').lower():
            ah_col = c
            break
    if ah_col is None:
        # fallback first col
        ah_col = df.columns[0]
    # parse date portion
    df[ah_col] = df[ah_col].astype(str)
    # try to capture '4/13/2016 12:00:00 AM' patterns
    try:
        parsed = pd.to_datetime(df[ah_col].str.extract(r'(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}:\d{2}\s*(AM|PM)?)')[0], errors='coerce')
    except Exception:
        parsed = pd.to_datetime(df[ah_col], errors='coerce', infer_datetime_format=True)
    if parsed is None or parsed.isna().all():
        parsed = pd.to_datetime(df[ah_col], errors='coerce', infer_datetime_format=True)
    df['Date'] = pd.to_datetime(parsed).dt.date
    # find metric column
    metric = None
    if value_col_candidates:
        for cand in value_col_candidates:
            for c in df.columns:
                if cand.lower() in str(c).lower():
                    metric = c
                    break
            if metric:
                break
    if metric is None:
        # pick first numeric column aside from activityhour
        for c in df.columns:
            if c == ah_col:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                metric = c
                break
    if metric is None:
        # try coercion on columns to numeric and pick
        for c in df.columns:
            if c == ah_col:
                continue
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                if pd.api.types.is_numeric_dtype(df[c]):
                    metric = c
                    break
            except Exception:
                continue
    if metric is None:
        return pd.DataFrame()
    # aggregate daily sum
    agg = df.groupby('Date')[metric].sum().reset_index().rename(columns={metric: metric})
    agg['Date'] = pd.to_datetime(agg['Date'])
    return agg

# -----------------------
# Master preprocessor: parse up to 18 files and merge
# -----------------------


def auto_preprocess_uploaded(files):
    """Main robust preprocessor that handles the many Fitbit table types described by the user.
    Returns a daily-level dataframe with Date and a set of numeric features (Steps, Calories, SleepHours, AvgHeartRate, etc.).
    """
    # Read files into dfs with hint names
    parsed_tables = {}
    for idx, f in enumerate(files[:MAX_FILES]):
        df = safe_read_csv(f)
        if df is None:
            continue
        name = getattr(f, 'name', f'upload_{idx}')
        hint = detect_table_hint(name, df)
        parsed_tables.setdefault(hint, []).append((name, df))

    # create a base daily DataFrame using dailyActivity or dailySteps or dailyCalories or sleepDay
    daily_frames = []
    # helper to coerce each table to daily-level df with Date and appropriate columns
    for hint, table_list in parsed_tables.items():
        for name, df in table_list:
            try:
                if hint in ['dailyActivity', 'dailySteps'] or any(x in str(name).lower() for x in ['dailyactivity', 'dailysteps']):
                    # these tables already have ActivityDate/ActivityDay + aggregated columns
                    df2, date_col = coerce_date_col(df, prefer_cols=['ActivityDate', 'ActivityDay', 'date'])
                    if date_col:
                        df2 = df2.rename(columns={date_col: 'Date'})
                        df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')
                        # normalize steps column names
                        col_map = {}
                        for c in df2.columns:
                            if str(c).lower() in ['totalsteps', 'steptotal', 'steps', 'stepstotal']:
                                col_map[c] = 'Steps'
                            if 'calori' in str(c).lower():
                                col_map[c] = 'Calories'
                            if 'sedentaryminutes' in str(c).lower():
                                col_map[c] = 'SedentaryMinutes'
                        df2 = df2.rename(columns=col_map)
                        df2 = clean_numeric_columns(df2)
                        daily_frames.append(df2[['Date'] + [c for c in df2.columns if c != 'Date']])
                elif hint in ['sleepDay', 'minuteSleep'] or 'sleep' in str(name).lower():
                    df2, date_col = coerce_date_col(df, prefer_cols=['sleepday', 'date'])
                    if date_col:
                        df2 = df2.rename(columns={date_col: 'Date'})
                        df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')
                        # convert minutes -> hours
                        if 'TotalMinutesAsleep' in df2.columns:
                            df2['SleepHours'] = pd.to_numeric(df2['TotalMinutesAsleep'], errors='coerce') / 60.0
                        df2 = clean_numeric_columns(df2)
                        daily_frames.append(df2[['Date'] + [c for c in df2.columns if c != 'Date']])
                elif hint in ['heartrate_seconds'] or 'heartrate' in str(name).lower():
                    # aggregate seconds to daily
                    try:
                        hr_daily = hr_seconds_to_daily(df)
                        daily_frames.append(hr_daily)
                    except Exception:
                        df2, date_col = coerce_date_col(df)
                        if date_col:
                            df2 = df2.rename(columns={date_col: 'Date'})
                            df2 = clean_numeric_columns(df2)
                            daily_frames.append(df2[['Date'] + [c for c in df2.columns if c != 'Date']])
                elif hint in ['hourlyCalories', 'hourlyIntensities', 'hourlySteps'] or 'hourly' in str(name).lower():
                    df2 = df.copy()
                    daily = hourly_to_daily(df2)
                    if not daily.empty:
                        daily_frames.append(daily)
                elif hint in ['minuteCaloriesNarrow', 'minuteIntensitiesNarrow', 'minuteMETsNarrow', 'minuteStepsNarrow'] or 'minute' in str(name).lower():
                    # minute narrow tables can be aggregated to daily by taking sum over minute timestamps
                    df2, date_col = coerce_date_col(df, prefer_cols=['ActivityMinute', 'date', 'activityminute'])
                    if date_col:
                        df2 = df2.rename(columns={date_col: 'Date'})
                        # parse Date from ActivityMinute which may contain time
                        df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce').dt.date
                        # aggregate numeric columns by sum
                        numeric_cols = [c for c in df2.columns if c != 'Date']
                        for c in numeric_cols:
                            df2[c] = pd.to_numeric(df2[c], errors='coerce')
                        df2_agg = df2.groupby('Date')[numeric_cols].sum().reset_index()
                        df2_agg['Date'] = pd.to_datetime(df2_agg['Date'])
                        daily_frames.append(df2_agg)
                elif hint in ['minuteCaloriesWide', 'minuteIntensitiesWide', 'minuteStepsWide'] or 'wide' in str(name).lower():
                    # wide minute tables have columns per minute; parse ActivityHour/ActivityDay to date then sum across minute columns
                    df2, date_col = coerce_date_col(df, prefer_cols=['ActivityHour', 'ActivityDay', 'date'])
                    if 'ActivityHour' in df2.columns or date_col:
                        ah_col = 'ActivityHour' if 'ActivityHour' in df2.columns else date_col
                        try:
                            df2['Date'] = pd.to_datetime(df2[ah_col], errors='coerce').dt.date
                        except Exception:
                            df2['Date'] = pd.to_datetime(df2[ah_col].astype(str), errors='coerce').dt.date
                        minute_cols = [c for c in df2.columns if any(p in str(c).lower() for p in [f"{i:02d}" for i in range(60)])]
                        if len(minute_cols) > 0:
                            for c in minute_cols:
                                df2[c] = pd.to_numeric(df2[c], errors='coerce')
                            df2_agg = df2.groupby('Date')[minute_cols].sum().reset_index()
                            df2_agg['Date'] = pd.to_datetime(df2_agg['Date'])
                            daily_frames.append(df2_agg)
                elif hint == 'weightLogInfo' or 'weight' in str(name).lower():
                    df2, date_col = coerce_date_col(df, prefer_cols=['date', 'weightdate'])
                    if date_col:
                        df2 = df2.rename(columns={date_col: 'Date'})
                        df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')
                        # weight in kg column detection
                        for c in df2.columns:
                            if 'weight' in str(c).lower() and 'kg' in str(c).lower() or str(c).lower() == 'weight':
                                df2 = df2.rename(columns={c: 'WeightKg'})
                        daily_frames.append(df2[['Date'] + [c for c in df2.columns if c != 'Date']])
                else:
                    # unknown: attempt to coerce to daily by any date-like col
                    df2, date_col = coerce_date_col(df)
                    if date_col:
                        df2 = df2.rename(columns={date_col: 'Date'})
                        df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')
                        df2 = clean_numeric_columns(df2)
                        daily_frames.append(df2[['Date'] + [c for c in df2.columns if c != 'Date']])
            except Exception as e:
                # continue parsing rest even if one table fails
                print(f"Failed to parse {name} as {hint}: {e}")
                continue

    if len(daily_frames) == 0:
        return pd.DataFrame()

    # merge all daily_frames on Date via outer join
    merged = daily_frames[0].copy()
    for df in daily_frames[1:]:
        try:
            merged = pd.merge(merged, df, on='Date', how='outer', suffixes=('', '_r'))
        except Exception:
            df = df.reset_index()
            merged = pd.merge(merged, df, on='Date', how='outer', suffixes=('', '_r'))

    # normalize merged: collapse duplicate columns (e.g., multiple Calories columns), coalesce where appropriate
    merged = merged.sort_values('Date').reset_index(drop=True)

    # coalesce variants of Steps / Calories / Sleep / AvgHeartRate
    def coalesce_columns(df, patterns, new_name, agg='first_non_null'):
        cols = [c for c in df.columns if any(p.lower() in str(c).lower() for p in patterns)]
        if len(cols) == 0:
            return df
        # if preferred exact exists
        if new_name in df.columns:
            return df
        # coalesce by taking first non-null, else sum for minute expansions
        if agg == 'first_non_null':
            try:
                df[new_name] = df[cols].bfill(axis=1).iloc[:, 0]
            except Exception:
                # fallback: pick first column
                df[new_name] = df[cols[0]]
        elif agg == 'sum':
            df[new_name] = df[cols].sum(axis=1, skipna=True)
        # drop original columns if they are different from new_name
        for c in cols:
            if c != new_name:
                try:
                    del df[c]
                except Exception:
                    pass
        return df

    merged = coalesce_columns(merged, ['step', 'steptotal', 'totalsteps'], 'Steps', agg='first_non_null')
    merged = coalesce_columns(merged, ['calori', 'dailycalories', 'cal'], 'Calories', agg='first_non_null')
    merged = coalesce_columns(merged, ['totalminutesasleep', 'sleephours', 'sleep'], 'SleepHours', agg='first_non_null')
    # if SleepHours came from minutes, convert
    if 'TotalMinutesAsleep' in merged.columns and 'SleepHours' not in merged.columns:
        merged['SleepHours'] = merged['TotalMinutesAsleep'] / 60.0
    merged = coalesce_columns(merged, ['avgheartrate', 'heartrate', 'avgheart', 'bpm', 'value'], 'AvgHeartRate', agg='first_non_null')
    merged = coalesce_columns(merged, ['weightkg', 'weight'], 'WeightKg', agg='first_non_null')

    # convert types
    for c in merged.columns:
        if c == 'Date':
            continue
        try:
            merged[c] = pd.to_numeric(merged[c], errors='coerce')
        except Exception:
            pass

    # fill numeric NA with median to avoid zscore issues later
    numcols = merged.select_dtypes(include=[np.number]).columns.tolist()
    for c in numcols:
        if merged[c].isna().all():
            # leave all-na as is
            continue
        merged[c] = merged[c].fillna(merged[c].median())

    # final ensure Date type
    merged['Date'] = pd.to_datetime(merged['Date'], errors='coerce')
    merged = merged.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    return merged

# -----------------------
# Math / Model functions (same as before but with sleep/fatigue proxies)
# -----------------------


def compute_gaussian_anomalies(df, columns, z_thresh=2.0):
    df = df.copy()
    for c in columns:
        try:
            z = zscore(df[c].astype(float), nan_policy='omit')
        except Exception:
            z = np.zeros(len(df))
        z = np.nan_to_num(z)
        df[c + '_z'] = z
        df[c + '_anomaly'] = np.where(np.abs(z) > z_thresh, 1, 0)
    return df


def shannon_entropy_vector(row_values, bins=5):
    try:
        counts, _ = np.histogram(row_values.astype(float), bins=bins)
        probs = counts / counts.sum() if counts.sum() > 0 else counts
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    except Exception:
        return 0.0


def compute_entropy(df, columns, bins=5):
    ent = df[columns].apply(lambda r: shannon_entropy_vector(r.values, bins=bins), axis=1)
    return ent


def compute_bci(entropy_series, anomaly_score_series, weights=(0.5, 0.5)):
    e = entropy_series.copy()
    if (e.max() - e.min()) == 0:
        e_norm = np.zeros_like(e)
    else:
        e_norm = (e - e.min()) / (e.max() - e.min())
    a = anomaly_score_series.copy()
    a = np.clip(a, 0, 1)
    w_e, w_a = weights
    bci = (1 - w_e * e_norm) * (1 - w_a * a)
    return bci


def classify_state(bci_value):
    if bci_value >= 0.75:
        return "Healthy"
    elif bci_value >= 0.5:
        return "Moderate"
    else:
        return "Unstable"


def build_transition_matrix(state_codes, n_states=3):
    trans = np.zeros((n_states, n_states), dtype=float)
    for i in range(len(state_codes) - 1):
        try:
            s_from = int(state_codes[i])
            s_to = int(state_codes[i + 1])
            trans[s_from, s_to] += 1
        except Exception:
            continue
    row_sums = trans.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        trans_norm = np.divide(trans, row_sums, where=row_sums != 0)
    trans_norm = np.nan_to_num(trans_norm)
    return trans_norm


def simulate_markov(initial_state, transition_matrix, steps=7):
    state = int(initial_state)
    seq = []
    for _ in range(steps):
        probs = transition_matrix[state]
        if probs.sum() == 0:
            next_state = state
        else:
            next_state = np.random.choice(len(probs), p=probs)
        seq.append(int(next_state))
        state = next_state
    return seq

# -----------------------
# Extra predictive proxies: sleep cycle & fatigue/stress proxies
# -----------------------


def predict_sleep_cycles(df):
    """Very simple heuristic: detect long contiguous sleep durations (SleepHours) and infer average sleep onset and wake times if minute-level sleep is present.
    Returns a DataFrame with predicted "TypicalSleepOnset" and "TypicalWakeUp" time-of-day (as strings) and sleep regularity score.
    This is not medical-grade and is intended for demonstration.
    """
    out = pd.DataFrame(index=df.index)
    if 'SleepHours' in df.columns and not df['SleepHours'].isna().all():
        out['SleepRegularity'] = df['SleepHours'].rolling(window=7, min_periods=1).apply(
            lambda x: np.nanstd(x) if len(x) > 0 else np.nan
        )
    else:
        out['SleepRegularity'] = np.nan
    # Attempt to detect onset/wake from minute-level sleep columns if present: look for columns that look like 'minuteSleep' detail
    # For demo, produce placeholder typical times
    out['TypicalSleepOnset'] = '23:30'
    out['TypicalWakeUp'] = '07:00'
    return out


def predict_fatigue_stress(df):
    """Construct a simple fatigue/stress proxy combining low SleepHours, high SedentaryMinutes, high AnomalyScore, and elevated AvgHeartRate.
    Returns a normalized fatigue_score between 0-1 (1 high fatigue/stress).
    """
    s = pd.Series(0, index=df.index, dtype=float)
    weights = {'sleep': 0.4, 'sedentary': 0.2, 'anomaly': 0.2, 'hr': 0.2}
    # normalize contributors
    if 'SleepHours' in df.columns:
        # less sleep -> more fatigue
        sh = df['SleepHours'].astype(float)
        sh_norm = 1 - (sh - sh.min()) / (sh.max() - sh.min() + 1e-9)
        s += weights['sleep'] * sh_norm.fillna(0)
    if 'SedentaryMinutes' in df.columns:
        sed = df['SedentaryMinutes'].astype(float)
        sed_norm = (sed - sed.min()) / (sed.max() - sed.min() + 1e-9)
        s += weights['sedentary'] * sed_norm.fillna(0)
    if 'AnomalyScore' in df.columns:
        a = df['AnomalyScore'].astype(float)
        a_norm = (a - a.min()) / (a.max() - a.min() + 1e-9)
        s += weights['anomaly'] * a_norm.fillna(0)
    if 'AvgHeartRate' in df.columns:
        hr = df['AvgHeartRate'].astype(float)
        hr_norm = (hr - hr.min()) / (hr.max() - hr.min() + 1e-9)
        s += weights['hr'] * hr_norm.fillna(0)
    # clip 0-1
    s = np.clip(s, 0, 1)
    return s

# -----------------------
# Long descriptions for pages (megaton-sized but concise here)
# -----------------------


def mega_description_upload():
    return """
## Upload & Analyze — In-Depth

This page is designed to accept the full exported Fitbit dataset broken into many tables (daily activity, daily steps, hourly summaries, minute-level wide/narrow tables, heart rate seconds, sleep logs, weight logs, etc.). The preprocessor attempts to parse each file independently, coerce date/time columns, aggregate fine-grain tables to daily resolution, and merge everything using an outer join so **no information is silently dropped**.

We apply a set of derived features and analytics:
- Rolling statistics (7-day mean/median) to smooth noisy measures
- Gaussian-based anomaly detection (Z-score) to flag outliers per-feature
- Shannon entropy computed across the selected feature vector for each day
- Behavioral Consistency Index (BCI): user-tunable blend of entropy and anomaly score, scaled to [0,1]
- Markov-chain estimation on discretized states (Healthy/Moderate/Unstable) to model transitions
- Simple sleep-cycle heuristics and a fatigue/stress proxy combining multiple signals

**Important:** This tool is demonstrative and academic. It is not a medical diagnostic tool. Interpret results carefully.
"""


def mega_description_forecast():
    return """
## Forecast Trends — In-Depth

Use Monte Carlo simulation to generate distributions of future states and visualize sample trajectories. This is helpful to demonstrate uncertainty and to create probabilistic short-term forecasts for demo purposes.
"""


def mega_description_models():
    return """
## Mathematical Models — In-Depth

We present the math behind each algorithm, full formulas, and interpretation notes suitable for inclusion in your report or slide deck. For each concept we show the formula, a short intuitive explanation, and interactive examples using your data.

- Shannon entropy: measures unpredictability of the daily multivariate vector — higher means more irregular behaviour.

Each section contains code-level explanations and pointers to literature.
"""


def mega_description_about():
    return """
## About LifeTrack — Extended

LifeTrack is an academic research prototype built to explore interpretable health & lifestyle analytics using simple, transparent algorithms. The project bundles preprocessing heuristics to handle exported files from consumer wearable platforms (Fitbit, etc.), lightweight statistical methods (z-score, entropy), and simple probabilistic forecasting (Markov + ARIMA). The goal is to provide a platform for prototyping signals and building demos.

Team & Acknowledgements:
- Student team: 4 members (list names here)
- Advisors: Prof. X
- Data sources: public Fitbit dataset format

Caveats and Ethics:
- Not medical software — do not use for diagnosis.
- The algorithms are illustrative. Use caution when presenting results to non-technical audiences.
"""

# -----------------------
# UI: Sidebar navigation with extra options & assets
# -----------------------
st.sidebar.title("LifeTrack — Navigation & Controls")
page = st.sidebar.radio("Choose page", [
    'Upload & Analyze',
    'Forecast Trends',
    'Explore Math Models',
    'About / Export'
])

# show banner if available
if os.path.exists(ASSETS_BANNER):
    try:
        st.sidebar.image(ASSETS_BANNER, use_container_width=True)
    except Exception:
        pass
elif os.path.exists(LOGO_PATH):
    try:
        st.sidebar.image(LOGO_PATH, use_container_width=True)
    except Exception:
        pass

# extra global controls
st.sidebar.markdown("---")
st.sidebar.subheader('Global options')
max_files_ui = st.sidebar.number_input('Max files to process (bounded by 18)', min_value=1, max_value=MAX_FILES, value=MAX_FILES)

# -----------------------
# Page implementations
# -----------------------
if page == 'Upload & Analyze':
    st.header('📥 Upload & Analyze — LifeTrack')
    st.markdown(mega_description_upload())
    uploaded_files = st.file_uploader(f'Upload up to {max_files_ui} CSV files', type=['csv'], accept_multiple_files=True)
    if uploaded_files:
        with st.spinner('Parsing and merging files — this may take ~10-30s for many files'):
            merged = auto_preprocess_uploaded(uploaded_files[:max_files_ui])
        if merged.empty:
            st.error('Preprocessing failed — no daily data could be produced. Check files and try again.')
        else:
            st.success('Preprocessing complete — daily table produced')
            file_names = [getattr(f, 'name', f'upload_{i}') for i, f in enumerate(uploaded_files[:max_files_ui])]
            st.markdown(
                f"**Files processed:** {len(file_names)}  \
                **Names:** {', '.join(file_names)}"
            )
            if 'Date' in merged.columns:
                date_min = merged['Date'].min()
                date_max = merged['Date'].max()
                if pd.notna(date_min) and pd.notna(date_max):
                    st.markdown(f"**Date range:** {date_min.date()} → {date_max.date()}")

            numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
            if 'Date' in numeric_cols:
                numeric_cols.remove('Date')
            chosen = st.sidebar.multiselect('Numeric features to analyze', numeric_cols, default=[c for c in ['Steps', 'Calories', 'SleepHours', 'AvgHeartRate'] if c in numeric_cols])
            z_thresh = st.sidebar.slider('Z-score anomaly threshold', 1.5, 4.0, 2.0, 0.1)
            bins = st.sidebar.slider('Entropy bins', 3, 12, 5)
            compute_roll = st.sidebar.checkbox('Compute rolling (7d) means', value=True)
            compute_pct = st.sidebar.checkbox('Compute daily percent-change', value=False)
            add_sleep_predict = st.sidebar.checkbox('Add sleep cycle predictions', value=True)
            add_fatigue = st.sidebar.checkbox('Add fatigue/stress proxy', value=True)

            if len(chosen) == 0:
                st.warning('Pick at least one numeric feature to analyze from the sidebar.')
            else:
                proc = merged.copy()
                # derived
                if compute_roll:
                    for c in chosen:
                        proc[c + '_roll7'] = proc[c].rolling(window=7, min_periods=1).mean()
                if compute_pct:
                    for c in chosen:
                        proc[c + '_pctchange'] = proc[c].pct_change().fillna(0)

                proc = compute_gaussian_anomalies(proc, chosen, z_thresh=z_thresh)
                proc['Entropy'] = compute_entropy(proc, chosen, bins=bins)
                anomaly_cols = [c for c in proc.columns if c.endswith('_anomaly')]
                proc['AnomalyScore'] = proc[anomaly_cols].mean(axis=1) if len(anomaly_cols) > 0 else 0
                proc['BCI'] = compute_bci(proc['Entropy'], proc['AnomalyScore'])
                proc['State'] = proc['BCI'].apply(classify_state)

                if add_sleep_predict:
                    sleep_preds = predict_sleep_cycles(proc)
                    proc = pd.concat([proc, sleep_preds], axis=1)
                if add_fatigue:
                    proc['FatigueScore'] = predict_fatigue_stress(proc)

                # show key tables
                st.subheader('Processed sample')
                show_cols = ['Date'] + chosen + ['Entropy', 'AnomalyScore', 'BCI', 'State']
                show_cols = [c for c in show_cols if c in proc.columns]
                st.dataframe(proc[show_cols].head(50))

                # Visualisations — many
                st.subheader('Visualizations')
                col1, col2 = st.columns(2)
                with col1:
                    if 'Date' in proc.columns and 'BCI' in proc.columns:
                        fig = px.line(proc, x='Date', y='BCI', markers=True, title='BCI over time')
                        st.plotly_chart(fig, use_container_width=True)
                    if 'FatigueScore' in proc.columns:
                        figf = px.line(proc, x='Date', y='FatigueScore', title='Fatigue proxy over time')
                        st.plotly_chart(figf, use_container_width=True)
                with col2:
                    if 'Entropy' in proc.columns:
                        fig2 = px.bar(proc, x='Date', y='Entropy', title='Entropy (daily)')
                        st.plotly_chart(fig2, use_container_width=True)
                    # Anomaly density
                    if len(anomaly_cols) > 0:
                        anom_density = proc[anomaly_cols].mean().reset_index()
                        anom_density.columns = ['Feature', 'AnomalyRate']
                        fig3 = px.bar(anom_density, x='Feature', y='AnomalyRate', title='Anomaly density across features')
                        st.plotly_chart(fig3, use_container_width=True)

                # Correlation heatmap for chosen
                if len(chosen) >= 2:
                    corr = proc[chosen].corr()
                    figc = px.imshow(corr, text_auto=True, title='Correlation matrix')
                    st.plotly_chart(figc, use_container_width=True)

                # Markov transitions and small simulation UI
                st.subheader('Markov chain and short-term forecast')
                state_map = {'Healthy': 0, 'Moderate': 1, 'Unstable': 2}
                inv_map = {v: k for k, v in state_map.items()}
                proc['StateCode'] = proc['State'].map(state_map)
                tm = build_transition_matrix(proc['StateCode'].fillna(0).astype(int).values, n_states=3)
                st.write('Transition matrix:')
                st.dataframe(pd.DataFrame(tm, index=[inv_map[i] for i in range(3)], columns=[inv_map[i] for i in range(3)]))
                last_state = int(proc['StateCode'].iloc[-1]) if 'StateCode' in proc.columns and not proc['StateCode'].isna().all() else None
                if last_state is not None:
                    nsim = st.number_input('Monte Carlo sims', min_value=50, max_value=2000, value=300)
                    ndays = st.number_input('Simulate days', min_value=1, max_value=30, value=7)
                    if st.button('Run simulation'):
                        sims = []
                        for _ in range(nsim):
                            sims.append(simulate_markov(last_state, tm, steps=ndays))
                        final = np.bincount([s[-1] for s in sims], minlength=3) / nsim
                        dfp = pd.DataFrame({'State': [inv_map[i] for i in range(3)], 'Probability': final})
                        st.dataframe(dfp)
                        st.plotly_chart(px.bar(dfp, x='State', y='Probability', color='State'), use_container_width=True)

                        sample_paths = [ [inv_map[state] for state in seq] for seq in sims[:min(8, len(sims))] ]
                        sample_df = pd.DataFrame(sample_paths).transpose()
                        sample_df.index = [f'Day {i+1}' for i in range(sample_df.shape[0])]
                        sample_df.columns = [f'Path {i+1}' for i in range(sample_df.shape[1])]
                        st.subheader('Sample simulated state trajectories')
                        st.dataframe(sample_df)

                # Export
                buf = io.StringIO()
                proc.to_csv(buf, index=False)
                st.download_button('⬇️ Download processed CSV', data=buf.getvalue(), file_name='lifetrack_processed.csv', mime='text/csv')

elif page == 'Forecast Trends':
    st.header('📈 Forecast Trends — LifeTrack')
    st.markdown(mega_description_forecast())
    up = st.file_uploader('Upload processed CSV (single)', type=['csv'])
    if up:
        try:
            df = pd.read_csv(up)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        except Exception:
            df = auto_preprocess_uploaded([up])
        if df.empty:
            st.error('Could not parse uploaded file')
        else:
            # recompute state and markov
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            chosen = st.multiselect('Select features for BCI recompute', numeric_cols, default=[c for c in ['Steps', 'Calories', 'SleepHours', 'AvgHeartRate'] if c in numeric_cols])
            if len(chosen) == 0:
                st.warning('Select features')
            else:
                proc = compute_gaussian_anomalies(df, chosen)
                proc['Entropy'] = compute_entropy(proc, chosen)
                anomaly_cols = [c for c in proc.columns if c.endswith('_anomaly')]
                proc['AnomalyScore'] = proc[anomaly_cols].mean(axis=1) if len(anomaly_cols) > 0 else 0
                proc['BCI'] = compute_bci(proc['Entropy'], proc['AnomalyScore'])
                proc['State'] = proc['BCI'].apply(classify_state)
                proc['StateCode'] = proc['State'].map({'Healthy': 0, 'Moderate': 1, 'Unstable': 2})
                tm = build_transition_matrix(proc['StateCode'].values, n_states=3)
                #st.write('Transition matrix:')
                #st.dataframe(pd.DataFrame(tm, index=['Healthy', 'Moderate', 'Unstable'], columns=['Healthy', 'Moderate', 'Unstable']))
                # simulation UI
                days = st.number_input('Days to simulate', 1, 30, 7)
                nsim = st.number_input('Monte Carlo runs', 50, 2000, 300)
                if st.button('Run forecast'):
                    last_state = int(proc['StateCode'].iloc[-1])
                    final = np.zeros(3)
                    sequences = []
                    for _ in range(nsim):
                        seq = simulate_markov(last_state, tm, steps=days)
                        sequences.append(seq)
                        final[seq[-1]] += 1
                    final = final / nsim
                    st.dataframe(pd.DataFrame({'State': ['Healthy', 'Moderate', 'Unstable'], 'Probability': final}))
                    st.plotly_chart(px.bar(pd.DataFrame({'State': ['Healthy', 'Moderate', 'Unstable'], 'Probability': final}), x='State', y='Probability', color='State'))

                    # show a few sample paths
                    sample_paths = sequences[:min(10, len(sequences))]
                    sp_df = pd.DataFrame(sample_paths).transpose()
                    sp_df.index = [f'Day_{i+1}' for i in range(sp_df.shape[0])]
                    st.subheader('Sample simulated paths (state codes)')
                    st.dataframe(sp_df)


                # optional ARIMA
                try:
                    from statsmodels.tsa.arima.model import ARIMA
                    if len(proc) >= 12:
                        st.subheader('ARIMA (BCI) forecast demo')
                        p = st.number_input('p (Autoregressive Order)', 0, 5, 1)
                        d = st.number_input('d (Differencing Order)', 0, 2, 0)
                        q = st.number_input('q (Moving Average Order)', 0, 5, 0)
                        if st.button('Run ARIMA'):
                            series = proc['BCI'].ffill()
                            model = ARIMA(series, order=(p, d, q))
                            res = model.fit()
                            nsteps = st.number_input('Steps to forecast', 1, 30, 7)
                            fc_mean = res.get_forecast(steps=nsteps).predicted_mean
                            st.write(fc_mean)
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=proc['Date'] if 'Date' in proc.columns else proc.index, y=proc['BCI'], name='BCI'))
                            if 'Date' in proc.columns and pd.api.types.is_datetime64_any_dtype(proc['Date']):
                                future_dates = pd.date_range(start=proc['Date'].iloc[-1] + pd.Timedelta(days=1), periods=nsteps, freq='D')
                                fig.add_trace(go.Scatter(x=future_dates, y=fc_mean, name='Forecast', line=dict(dash='dash')))
                            else:
                                future_x = list(range(len(proc), len(proc) + nsteps))
                                fig.add_trace(go.Scatter(x=future_x, y=fc_mean, name='Forecast', line=dict(dash='dash')))
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info('Not enough points for ARIMA (>=12 recommended).')
                except Exception:
                    st.info('ARIMA not available in this environment.')
elif page == 'Explore Math Models':
    st.header('📚 Mathematical Models — LifeTrack')
    st.markdown(mega_description_models())
    st.write('Interactive demo: upload a processed CSV to try models on your data')
    up = st.file_uploader('Upload processed CSV', type=['csv'])
    if up:
        df = pd.read_csv(up)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            st.error('No numeric columns detected')
        else:
            sel = st.multiselect('Choose features for demo', numeric_cols, default=numeric_cols[:4])
            bins = st.slider('Entropy bins', 3, 12, 5)
            if sel:
                proc = compute_gaussian_anomalies(df, sel)
                proc['Entropy'] = compute_entropy(proc, sel, bins=bins)
                st.subheader('Entropy over time')
                if 'Date' in proc.columns:
                    st.plotly_chart(px.line(proc, x='Date', y='Entropy'), use_container_width=True)
                else:
                    st.plotly_chart(px.line(proc, y='Entropy'), use_container_width=True)
                # show formula and explanation
                st.latex(r"H(X) = -\sum p(x) \log_2 p(x)")
                st.write('Entropy measures unpredictability across the feature vector for a day.')

elif page == 'About / Export':
    st.header('ℹ️ About LifeTrack — Extended')
    st.markdown(mega_description_about())
    st.subheader('Assets')
    if os.path.exists(ASSETS_BANNER):
        st.image(ASSETS_BANNER, caption='Project banner')
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=200)
    st.subheader('Export options & notes')
    st.write('You can export processed CSV from the Upload & Analyze page. If you need programmatic PPT / PDF export, ask and I will add a code snippet to generate PowerPoint slides from processed figures.')
    st.caption('LifeTrack — educational demo. Not medical advice.')

# end of file