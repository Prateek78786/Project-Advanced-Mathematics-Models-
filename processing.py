import io
import re
import numpy as np
import pandas as pd
from scipy.stats import zscore


def safe_read_csv(file_storage):
    try:
        file_storage.stream.seek(0)
    except Exception:
        pass

    content = file_storage.read()
    if isinstance(content, bytes):
        try:
            content = content.decode('utf-8')
        except UnicodeDecodeError:
            content = content.decode('latin1', errors='replace')
    text = str(content)

    for sep in [',', ';', '\t']:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue

    try:
        return pd.read_csv(io.StringIO(text))
    except Exception:
        return None


def find_date_column(df):
    date_candidates = [c for c in df.columns if re.search(r'date|day|time', str(c), re.IGNORECASE)]
    for c in date_candidates:
        column = df[c]
        dtype = column.dtype
        is_datetime = pd.api.types.is_datetime64_any_dtype(dtype)
        is_string = pd.api.types.is_string_dtype(dtype) or dtype == object
        first_value = str(column.dropna().iloc[0]) if not column.dropna().empty else ''
        if is_datetime or is_string and re.search(r'\d', first_value):
            return c
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c].dtype) and df[c].astype(str).str.contains(r'\d{4}[-/]\d{2}[-/]\d{2}|\d{1,2}/\d{1,2}/\d{2,4}').any():
            return c
    return None


def coerce_date(df):
    df = df.copy()
    date_col = find_date_column(df)
    if date_col is None:
        return df, None
    try:
        df[date_col] = pd.to_datetime(df[date_col].astype(str), errors='coerce', infer_datetime_format=True)
    except Exception:
        df[date_col] = pd.to_datetime(df[date_col].astype(str), errors='coerce')

    df = df.rename(columns={date_col: 'Date'})
    if df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_convert(None)
    df['Date'] = pd.to_datetime(df['Date'].dt.date)
    return df, 'Date'


def clean_numeric(df):
    df = df.copy()
    for col in df.columns:
        if col == 'Date':
            continue
        if df[col].dtype == object:
            cleaned = df[col].astype(str).str.replace(r'[,$]', '', regex=True).str.strip()
            df[col] = pd.to_numeric(cleaned, errors='coerce')
    return df


def aggregate_daily(df):
    df = df.copy()
    if 'Date' not in df.columns:
        return df
    df = df.dropna(subset=['Date'])
    if df.shape[0] == 0:
        return df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if df['Date'].duplicated().any():
        agg = {}
        for col in numeric_cols:
            name = col.lower()
            if any(keyword in name for keyword in ['step', 'calori', 'distance', 'active', 'minute', 'speed']):
                agg[col] = 'sum'
            else:
                agg[col] = 'mean'
        df = df.groupby('Date').agg(agg).reset_index()
    return df


CANONICAL_COLUMNS = {
    'Steps': ['step', 'totalsteps', 'steps', 'stepstotal'],
    'Calories': ['calori', 'energy'],
    'SleepHours': ['sleep', 'minutesasleep', 'sleepminutes', 'sleepduration'],
    'AvgHeartRate': ['heartrate', 'bpm', 'avgheartrate', 'averageheartrate'],
    'SedentaryMinutes': ['sedentary', 'inactive', 'sedentaryminutes'],
    'WeightKg': ['weight', 'weightkg']
}


def coalesce_metric(df, target_name, alternatives):
    matches = [c for c in df.columns if any(p in c.lower() for p in alternatives)]
    if target_name in df.columns:
        return df
    if not matches:
        return df
    source = df[matches]
    if any('step' in c.lower() or 'calori' in c.lower() for c in matches):
        df[target_name] = source.mean(axis=1, skipna=True)
    else:
        df[target_name] = source.bfill(axis=1).iloc[:, 0]
    return df


def preprocess_files(files):
    daily_frames = []
    errors = []
    for uploaded in files:
        df = safe_read_csv(uploaded)
        if df is None or df.empty:
            errors.append(f'Could not read {uploaded.filename}')
            continue
        df, date_col = coerce_date(df)
        if date_col is None:
            errors.append(f'No date column found in {uploaded.filename}')
            continue
        df = clean_numeric(df)
        df = aggregate_daily(df)
        if 'Date' not in df.columns or df['Date'].isna().all():
            errors.append(f'No valid dates after parsing {uploaded.filename}')
            continue
        daily_frames.append(df)

    if not daily_frames:
        return pd.DataFrame(), errors

    merged = daily_frames[0]
    for frame in daily_frames[1:]:
        merged = pd.merge(merged, frame, on='Date', how='outer', suffixes=('', '_r'))

    for name, patterns in CANONICAL_COLUMNS.items():
        merged = coalesce_metric(merged, name, patterns)

    merged = merged.sort_values('Date').reset_index(drop=True)
    merged['Date'] = pd.to_datetime(merged['Date'], errors='coerce')
    merged = merged.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if merged[col].isna().all():
            continue
        merged[col] = merged[col].fillna(merged[col].median())

    return merged, errors


def compute_gaussian_anomalies(df, columns, threshold=2.0):
    df = df.copy()
    for c in columns:
        if c not in df.columns:
            continue
        z = np.nan_to_num(zscore(df[c].astype(float), nan_policy='omit'))
        df[f'{c}_z'] = z
        df[f'{c}_anomaly'] = np.where(np.abs(z) > threshold, 1, 0)
    return df


def shannon_entropy(row, bins=5):
    values = np.asarray(row).astype(float)
    counts, _ = np.histogram(values[~np.isnan(values)], bins=bins)
    probs = counts.astype(float) / counts.sum() if counts.sum() > 0 else counts
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs))) if len(probs) > 0 else 0.0


def compute_entropy(df, columns, bins=5):
    return df[columns].apply(lambda row: shannon_entropy(row, bins=bins), axis=1)


def compute_bci(entropy, anomaly_score, weights=(0.55, 0.45)):
    e = entropy.copy()
    if e.max() == e.min():
        e_norm = np.zeros_like(e)
    else:
        e_norm = (e - e.min()) / (e.max() - e.min())
    a = anomaly_score.copy().fillna(0).clip(0, 1)
    return (1 - weights[0] * e_norm) * (1 - weights[1] * a)


def classify_state(bci_value):
    if bci_value >= 0.78:
        return 'Healthy'
    if bci_value >= 0.55:
        return 'Moderate'
    return 'Unstable'


def build_transition_matrix(states, n_states=3):
    transitions = np.zeros((n_states, n_states), dtype=float)
    for source, target in zip(states[:-1], states[1:]):
        transitions[int(source), int(target)] += 1
    row_sums = transitions.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        probabilities = np.divide(transitions, row_sums, where=row_sums != 0)
    return np.nan_to_num(probabilities)


def simulate_markov(initial_state, transition_matrix, steps=10):
    state = int(initial_state)
    sequence = []
    for _ in range(steps):
        probs = transition_matrix[state]
        if probs.sum() == 0:
            sequence.append(state)
            continue
        state = np.random.choice(len(probs), p=probs)
        sequence.append(int(state))
    return sequence


def arima_forecast(series, steps=7, order=(1, 0, 1)):
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(series.fillna(method='ffill').astype(float), order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=steps)
        return forecast.tolist()
    except Exception:
        return []


def predict_sleep_and_fatigue(df):
    df = df.copy()
    if 'SleepHours' in df.columns:
        df['SleepRegularity'] = df['SleepHours'].rolling(window=7, min_periods=1).std().fillna(0)
        sleep_norm = 1 - (df['SleepHours'] - df['SleepHours'].min()) / (df['SleepHours'].max() - df['SleepHours'].min() + 1e-9)
    else:
        df['SleepRegularity'] = 0
        sleep_norm = 0

    sed_norm = 0
    if 'SedentaryMinutes' in df.columns:
        sed_norm = (df['SedentaryMinutes'] - df['SedentaryMinutes'].min()) / (df['SedentaryMinutes'].max() - df['SedentaryMinutes'].min() + 1e-9)
    hr_norm = 0
    if 'AvgHeartRate' in df.columns:
        hr_norm = (df['AvgHeartRate'] - df['AvgHeartRate'].min()) / (df['AvgHeartRate'].max() - df['AvgHeartRate'].min() + 1e-9)

    anomaly_norm = df['AnomalyScore'] if 'AnomalyScore' in df.columns else 0
    fatigue = 0.4 * sleep_norm + 0.25 * sed_norm + 0.2 * anomaly_norm + 0.15 * hr_norm
    df['FatigueScore'] = np.clip(fatigue, 0, 1)
    return df


def build_prescription_cards(summary):
    cards = []
    if summary['last_state'] == 'Unstable' or summary['bci_mean'] < 0.55:
        cards.append({
            'title': 'Stabilize daily routine',
            'description': 'Establish consistent wake and sleep times, and limit late-night screen exposure to improve overall lifestyle stability.',
            'type': 'warning'
        })
    else:
        cards.append({
            'title': 'Maintain strong habits',
            'description': 'Your routine is stable. Keep your current consistency, and monitor emerging trends weekly.',
            'type': 'success'
        })

    if summary['sleep_average'] and summary['sleep_average'] < 7.0:
        cards.append({
            'title': 'Prioritize restorative sleep',
            'description': 'Aim for 7–8 hours nightly and keep your bedtime consistent. Sleep regularity is essential for recovery and cognitive resilience.',
            'type': 'alert'
        })

    if summary['fatigue_average'] and summary['fatigue_average'] >= 0.6:
        cards.append({
            'title': 'Reduce fatigue load',
            'description': 'Add gentle movement breaks, hydrate regularly, and schedule at least one active recovery day to lower stress and fatigue risk.',
            'type': 'alert'
        })

    if summary['anomaly_average'] and summary['anomaly_average'] >= 0.15:
        cards.append({
            'title': 'Investigate outlier days',
            'description': 'Several days show unusual patterns. Review your recent schedule, sleep, and stressors to understand anomalies.',
            'type': 'info'
        })

    if not cards:
        cards.append({
            'title': 'Strong consistency detected',
            'description': 'Your metrics are within a balanced zone. Keep tracking to catch early deviations and maintain performance.',
            'type': 'success'
        })
    return cards


def prepare_chart_data(df):
    return {
        'dates': df['Date'].dt.strftime('%Y-%m-%d').tolist(),
        'bci': df['BCI'].round(3).tolist(),
        'entropy': df['Entropy'].round(3).tolist(),
        'anomaly': df['AnomalyScore'].round(3).tolist(),
        'fatigue': df['FatigueScore'].round(3).tolist(),
        'sleep': df['SleepHours'].fillna(0).round(2).tolist(),
        'states': df['State'].tolist()
    }


def analyze_upload(files):
    data, errors = preprocess_files(files)
    if data.empty:
        return {'errors': errors}

    numeric_cols = [c for c in data.select_dtypes(include=[np.number]).columns if c != 'Date']
    columns_to_analyze = [c for c in ['Steps', 'Calories', 'SleepHours', 'AvgHeartRate', 'SedentaryMinutes', 'WeightKg'] if c in numeric_cols]
    if not columns_to_analyze:
        columns_to_analyze = numeric_cols[:4]

    data = compute_gaussian_anomalies(data, columns_to_analyze)
    data['Entropy'] = compute_entropy(data, columns_to_analyze)
    anomaly_columns = [col for col in data.columns if col.endswith('_anomaly')]
    data['AnomalyScore'] = data[anomaly_columns].mean(axis=1) if anomaly_columns else 0
    data['BCI'] = compute_bci(data['Entropy'], data['AnomalyScore'])
    data['State'] = data['BCI'].apply(classify_state)
    data['StateCode'] = data['State'].map({'Healthy': 0, 'Moderate': 1, 'Unstable': 2}).fillna(1).astype(int)
    data = predict_sleep_and_fatigue(data)

    states = data['StateCode'].tolist()
    transition_matrix = build_transition_matrix(states)
    last_state = int(states[-1]) if states else 1
    simulations = [simulate_markov(last_state, transition_matrix, steps=10) for _ in range(300)]
    distribution = np.bincount([seq[-1] for seq in simulations], minlength=3) / 300.0

    forecast_sequence = simulate_markov(last_state, transition_matrix, steps=14)
    state_lookup = ['Healthy', 'Moderate', 'Unstable']
    forecast_labels = [state_lookup[s] for s in forecast_sequence]

    arima_values = []
    if len(data) >= 12 and 'statsmodels' in globals():
        arima_values = arima_forecast(data['BCI'], steps=14, order=(1, 0, 1))
    else:
        arima_values = []

    summary = {
        'days': int(data.shape[0]),
        'start_date': data['Date'].dt.strftime('%Y-%m-%d').iloc[0],
        'end_date': data['Date'].dt.strftime('%Y-%m-%d').iloc[-1],
        'bci_mean': float(data['BCI'].mean()),
        'anomaly_average': float(data['AnomalyScore'].mean()),
        'sleep_average': float(data['SleepHours'].mean()) if 'SleepHours' in data.columns else None,
        'fatigue_average': float(data['FatigueScore'].mean()),
        'last_state': data['State'].iloc[-1],
        'state_counts': data['State'].value_counts().to_dict(),
        'transition_matrix': transition_matrix.tolist(),
        'forecast': {
            'state_probability': {
                'Healthy': float(distribution[0]),
                'Moderate': float(distribution[1]),
                'Unstable': float(distribution[2])
            },
            'simulated_path': forecast_labels,
            'arima_bci': [float(v) for v in arima_values] if arima_values else []
        }
    }

    prescriptions = build_prescription_cards(summary)
    daily_preview = data[['Date', 'BCI', 'Entropy', 'AnomalyScore', 'State', 'SleepHours', 'FatigueScore']].head(16).copy()
    daily_preview['Date'] = daily_preview['Date'].dt.strftime('%Y-%m-%d')

    return {
        'errors': errors,
        'summary': summary,
        'charts': prepare_chart_data(data),
        'prescriptions': prescriptions,
        'daily_preview': daily_preview.to_dict(orient='records'),
        'anomaly_heatmap': {
            'labels': [col.replace('_anomaly', '') for col in anomaly_columns],
            'values': [float(data[col].mean()) for col in anomaly_columns]
        }
    }
