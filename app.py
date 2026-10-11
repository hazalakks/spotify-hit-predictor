"""
Spotify Hit Predictor & Audio Engagement Web Application
Built with Streamlit and Explainable Machine Learning.
Allows users, curators, and music producers to test acoustic signatures in real-time.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Spotify Hit Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Spotify Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1DB954;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #B3B3B3;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #181818;
        border-radius: 12px;
        padding: 20px;
        border-left: 5px solid #1DB954;
        margin-bottom: 15px;
    }
    .hit-badge {
        background-color: #1DB954;
        color: #000000;
        font-weight: 800;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.1rem;
    }
    .skip-badge {
        background-color: #E53935;
        color: #FFFFFF;
        font-weight: 800;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("models/preprocessor.joblib")
    model = joblib.load("models/lightgbm.joblib")
    return preprocessor, model

def plot_radar_chart(features_dict):
    categories = ["Danceability", "Energy", "Valence", "Acousticness", "Speechiness", "Liveness"]
    values = [
        features_dict["danceability"],
        features_dict["energy"],
        features_dict["valence"],
        features_dict["acousticness"],
        features_dict["speechiness"],
        features_dict["liveness"]
    ]
    
    # Repeat first value to close polygon
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#121212")
    ax.set_facecolor("#181818")
    
    ax.plot(angles, values, color="#1DB954", linewidth=2.5)
    ax.fill(angles, values, color="#1DB954", alpha=0.35)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="#FFFFFF", fontsize=10, fontweight="bold")
    ax.set_yticklabels([])
    ax.spines["polar"].set_color("#535353")
    ax.grid(color="#404040", linestyle="--")
    
    return fig

def main():
    st.markdown('<div class="main-header">🎵 Spotify Hit Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Hit Song Science & Audio Engagement Diagnostic Pipeline</div>', unsafe_allow_html=True)
    
    # Sidebar: Track Input Controls
    st.sidebar.header("🎚️ Track Audio Features")
    
    track_title = st.sidebar.text_input("Track Title", value="Midnight Resonance")
    artist_name = st.sidebar.text_input("Artist Name", value="Nova Echo")
    genre = st.sidebar.selectbox("Genre", ["pop", "rap", "rock", "edm", "latin", "r&b"], index=0)
    
    st.sidebar.subheader("Acoustic Sliders")
    danceability = st.sidebar.slider("Danceability", 0.0, 1.0, 0.74, 0.01)
    energy = st.sidebar.slider("Energy", 0.0, 1.0, 0.82, 0.01)
    valence = st.sidebar.slider("Valence (Musical Positiveness)", 0.0, 1.0, 0.65, 0.01)
    loudness = st.sidebar.slider("Loudness (dB)", -30.0, 0.0, -5.4, 0.1)
    speechiness = st.sidebar.slider("Speechiness", 0.0, 1.0, 0.08, 0.01)
    acousticness = st.sidebar.slider("Acousticness", 0.0, 1.0, 0.12, 0.01)
    instrumentalness = st.sidebar.slider("Instrumentalness", 0.0, 1.0, 0.00, 0.01)
    liveness = st.sidebar.slider("Liveness", 0.0, 1.0, 0.14, 0.01)
    tempo = st.sidebar.slider("Tempo (BPM)", 50.0, 220.0, 124.0, 1.0)
    duration_min = st.sidebar.slider("Duration (Minutes)", 1.0, 7.0, 3.2, 0.1)
    release_year = st.sidebar.slider("Release Year", 1970, 2025, 2023, 1)
    
    key_val = st.sidebar.selectbox("Key", list(range(12)), format_func=lambda x: ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][x])
    mode_val = st.sidebar.radio("Modality", [1, 0], format_func=lambda x: "Major" if x == 1 else "Minor", horizontal=True)
    
    # Feature calculations
    raw_dict = {
        "danceability": danceability,
        "energy": energy,
        "loudness": loudness,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "tempo": tempo,
        "duration_min": duration_min,
        "mood_index": valence * energy,
        "party_index": valence * danceability,
        "dance_energy_ratio": danceability / (energy + 1e-5),
        "speech_to_dance": speechiness / (danceability + 1e-5),
        "song_age": 2020 - release_year,
        "key_sin": np.sin(2 * np.pi * key_val / 12.0),
        "key_cos": np.cos(2 * np.pi * key_val / 12.0),
        "playlist_genre": genre,
        "mode": mode_val
    }
    
    input_df = pd.DataFrame([raw_dict])
    
    # Load model
    try:
        preprocessor, model = load_artifacts()
        transformed_input = preprocessor.transform(input_df)
        prob = model.predict_proba(transformed_input)[0, 1]
    except Exception as e:
        st.warning(f"Models are currently training or not yet found: {e}. Displaying demo simulation.")
        prob = 0.72
        
    threshold = 0.50
    is_hit = prob >= threshold
    
    # Main Dashboard Layout
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown(f"### Diagnostic Report: *{track_title}* by **{artist_name}**")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Predicted Hit Probability", f"{prob * 100:.1f}%")
        with col_m2:
            st.metric("Curator Decision Threshold", f"{threshold * 100:.0f}%")
            
        if is_hit:
            st.markdown(f'<div class="hit-badge">🔥 COMMERCIAL HIT POTENTIAL ({prob*100:.1f}%)</div>', unsafe_allow_html=True)
            st.success("High streaming retention predicted. Recommended for editorial placement on Top Hits playlists.")
        else:
            st.markdown(f'<div class="skip-badge">⏭️ LOW HIT POTENTIAL ({prob*100:.1f}%)</div>', unsafe_allow_html=True)
            st.info("Audio profile resembles tracks that stayed below popularity 60. Better suited to niche or genre-specific playlists.")
            
        st.markdown("#### Russell's Affective Quadrant:")
        if valence >= 0.5 and energy >= 0.5:
            mood_label = "🌟 **Euphoric & Energetic** (Prime Club / High-Energy Party Hit)"
        elif valence < 0.5 and energy >= 0.5:
            mood_label = "⚡ **Tense & Intense** (Aggressive Trap / Workout Anthem)"
        elif valence < 0.5 and energy < 0.5:
            mood_label = "🌧️ **Melancholic & Reflective** (Lo-Fi / Late Night Sad Hour)"
        else:
            mood_label = "☕ **Peaceful & Acoustic** (Chill Study / Sunday Morning Relaxation)"
        st.write(mood_label)
        
        st.markdown("#### Audio Feature Ratios:")
        st.write(f"- **Mood Index (Valence × Energy):** `{raw_dict['mood_index']:.2f}`")
        st.write(f"- **Party Index (Valence × Danceability):** `{raw_dict['party_index']:.2f}`")
        st.write(f"- **Dance / Energy Balance:** `{raw_dict['dance_energy_ratio']:.2f}`")
        st.write(f"- **Speech Density:** `{raw_dict['speech_to_dance']:.2f}`")
        
    with col2:
        st.markdown("### 🕸️ Acoustic Radar Fingerprint")
        radar_fig = plot_radar_chart(raw_dict)
        st.pyplot(radar_fig)
        
    st.markdown("---")
    st.markdown("### 💡 Why this prediction?")
    st.write(
        "SHAP analysis of the LightGBM model shows that vocal presence (low instrumentalness), louder masters, "
        "pop/rock genre and shorter duration increase hit probability, while very high energy and long runtimes lower it. "
        "Note: audio explains only part of popularity - artist fame and marketing are not in the data (ROC-AUC ~0.75)."
    )

if __name__ == "__main__":
    main()
