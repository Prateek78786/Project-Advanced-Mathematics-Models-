import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="LifeTrack Prototype", layout="wide")

st.title("📈 LifeTrack: Lifestyle Consistency Analyzer")

# Upload dataset
uploaded = st.file_uploader("📤 Upload your lifestyle CSV file", type=['csv'])

if uploaded:
    data = pd.read_csv(uploaded)
    st.subheader("📋 Uploaded Data")
    st.dataframe(data.head())

    
    # Gaussian anomaly detection
    
    numeric_cols = data.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        data[col + '_z'] = zscore(data[col])
        data[col + '_anomaly'] = data[col + '_z'].apply(lambda x: 1 if abs(x) > 2 else 0)

    
    # Shannon Entropy Calculation
    
    def shannon_entropy(row):
        probs = np.histogram(row, bins=5, density=True)[0]
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    data['Entropy'] = data[numeric_cols].apply(shannon_entropy, axis=1)

    
    # Anomaly Score + BCI

    anomaly_cols = [c for c in data.columns if '_anomaly' in c]
    data['AnomalyScore'] = data[anomaly_cols].mean(axis=1)

    def compute_bci(entropy, anomaly):
        e_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min())
        bci = (1 - 0.5 * e_norm) * (1 - 0.5 * anomaly)
        return bci

    data['BCI'] = compute_bci(data['Entropy'], data['AnomalyScore'])

    
    # Classify Lifestyle State
    
    def classify_state(bci):
        if bci > 0.75: return "Healthy"
        elif bci > 0.5: return "Moderate"
        else: return "Unstable"
    data['State'] = data['BCI'].apply(classify_state)

    st.subheader("📊 Analyzed Data")
    st.dataframe(data[['Day', 'Entropy', 'AnomalyScore', 'BCI', 'State']])

    
    # Markov Chain Modeling
    
    st.subheader("🔁 Markov Chain State Transitions")

    # Encode states numerically
    state_mapping = {"Healthy": 0, "Moderate": 1, "Unstable": 2}
    data['StateCode'] = data['State'].map(state_mapping)

    # Build transition matrix
    transitions = np.zeros((3, 3))
    states = data['StateCode'].values

    for i in range(len(states) - 1):
        transitions[states[i], states[i + 1]] += 1

    transition_matrix = (transitions.T / transitions.sum(axis=1)).T
    transition_matrix = np.nan_to_num(transition_matrix)  # replace NaN with 0

    transition_df = pd.DataFrame(
        transition_matrix,
        index=["Healthy", "Moderate", "Unstable"],
        columns=["Healthy", "Moderate", "Unstable"]
    )

    st.write("Transition Probability Matrix:")
    st.dataframe(transition_df.style.background_gradient(cmap="YlGnBu"))

    # Predict next state
    last_state = data['StateCode'].iloc[-1]
    predicted_next_state = np.argmax(transition_matrix[last_state])
    reverse_map = {v: k for k, v in state_mapping.items()}
    predicted_label = reverse_map[predicted_next_state]

    st.info(f"🧠 Predicted next lifestyle state: **{predicted_label}**")

    
    # Visualization
    
    st.subheader("📉 Balanced Consistency Index (BCI) Trend")
    fig_bci = px.line(data, x='Day', y='BCI', markers=True, color='State',
                      color_discrete_map={"Healthy": "green", "Moderate": "orange", "Unstable": "red"})
    st.plotly_chart(fig_bci, use_container_width=True)

    st.subheader("🔥 Entropy Distribution")
    fig_entropy = px.bar(data, x='Day', y='Entropy', color='State',
                         color_discrete_map={"Healthy": "green", "Moderate": "orange", "Unstable": "red"})
    st.plotly_chart(fig_entropy, use_container_width=True)

    st.subheader("⚠️ Anomaly Density per Feature")
    anomaly_density = data[anomaly_cols].mean().reset_index()
    anomaly_density.columns = ['Feature', 'AnomalyRate']
    fig_anom = px.bar(anomaly_density, x='Feature', y='AnomalyRate',
                      color='AnomalyRate', color_continuous_scale='RdYlGn_r')
    st.plotly_chart(fig_anom, use_container_width=True)

    
    # Summary and Insights
    
    avg_bci = data['BCI'].mean()
    avg_entropy = data['Entropy'].mean()

    col1, col2 = st.columns(2)
    col1.metric(label="Average BCI", value=f"{avg_bci:.3f}")
    col2.metric(label="Average Entropy", value=f"{avg_entropy:.3f}")

    st.success("✅ Analysis complete. Prototype executed successfully.")

    st.caption("Developed as part of LifeTrack — A Mathematical Framework for Lifestyle Consistency and Prediction.")

else:
    st.info("Please upload a CSV file (with at least 10 numerical lifestyle columns) to begin analysis.")
