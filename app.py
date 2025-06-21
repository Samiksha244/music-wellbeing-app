import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from datetime import datetime
import os

# Sample dataset
X = np.array([
    [25, 2], [32, 1], [45, 0], [60, 2], [29, 1],
    [52, 0], [41, 2], [35, 1], [22, 2], [48, 0]
])
y = np.array([90, 75, 50, 88, 70, 45, 85, 72, 92, 40])

# Train models
reg = LinearRegression()
reg.fit(X, y)

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

cluster_labels = {
    0: "High engagement → strong cognitive/emotional resilience",
    1: "Moderate engagement → stable benefits",
    2: "Low engagement → higher stress, lower memory performance"
}

playlist_links = {
    "happy": "https://open.spotify.com/embed/playlist/37i9dQZF1DXdPec7aLTmlC",  # Happy Hits
    "sad": "https://open.spotify.com/embed/playlist/37i9dQZF1DX3YSRoSdA634",    # Life Sucks
    "stressed": "https://open.spotify.com/embed/playlist/37i9dQZF1DWXe9gFZP0gtP",  # Chill Out
    "in pain": "https://open.spotify.com/embed/playlist/37i9dQZF1DX3PIPIT6lEg5"   # Deep Focus
}

# Streamlit UI
st.set_page_config(page_title="Music & Well-being Predictor", layout="centered")
st.title("🎵 Music & Well-being Predictor")
st.markdown("This tool shows how music engagement affects well-being using ML and gives personalized music suggestions.")

age = st.slider("Your Age", 10, 80, 25)
engagement_level = st.radio("Musical Engagement Level", [0, 1, 2], format_func=lambda x: {
    0: "Low (rarely listen)", 1: "Moderate (casual listening/singing)", 2: "High (play instrument or therapy)"
}[x])
mood = st.selectbox("Current Mood or Physical State", ["happy", "sad", "stressed", "in pain"])

if st.button("🎯 Predict and Recommend"):
    input_data = np.array([[age, engagement_level]])
    score = reg.predict(input_data)[0]
    cluster = kmeans.predict(input_data)[0]

    st.success(f"Predicted Well-being Score: **{score:.1f}/100**")
    st.info(f"Cluster Group: {cluster} — {cluster_labels[cluster]}")
    
    # Spotify Embed
    st.markdown("### 🎧 Music Suggestion for You:")
    st.components.v1.iframe(playlist_links[mood], height=80)

    # Save to log
    log = pd.DataFrame([{
        "timestamp": datetime.now(),
        "age": age,
        "engagement_level": engagement_level,
        "mood": mood,
        "score": round(score, 1)
    }])
    if not os.path.exists("user_log.csv"):
        log.to_csv("user_log.csv", index=False)
    else:
        log.to_csv("user_log.csv", mode='a', index=False, header=False)

    # Chart
    st.markdown("---")
    st.subheader("📈 Regression Fit: Music Engagement vs Well-being")
    fig, ax = plt.subplots()
    ax.scatter(X[:, 1], y, color='blue', label='Data')
    ax.plot(X[:, 1], reg.predict(X), color='red', label='Regression Line')
    ax.set_xlabel("Musical Engagement Level")
    ax.set_ylabel("Well-being Score")
    ax.set_title("Impact of Musical Engagement on Well-being")
    ax.legend()
    st.pyplot(fig)

    # Show log history
    st.markdown("---")
    if st.checkbox("📊 Show My Mood & Score History"):
        df = pd.read_csv("user_log.csv")
        st.dataframe(df.tail(10))
        st.line_chart(df["score"])
