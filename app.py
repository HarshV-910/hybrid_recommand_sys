import streamlit as st
import pandas as pd
import numpy as np
from numpy import load
import joblib
from scipy.sparse import load_npz
from src.features.content_sys import content_based_recommand
from src.features.collaborative_sys import collaborative_recommand
from src.features.hybrid_sys import HybridRecommender

# Load data
transformed_data = load_npz('data/processed/df_transformed.npz')
# transformer = joblib.load('models/transformer.joblib')
df_song = pd.read_csv('data/raw/Music_Info.csv')
df_user = pd.read_csv('data/raw/User_Listening_History.csv')
filtered_song_df = pd.read_csv('data/processed/collab_filtered.csv')
track_ids = np.load('models/track_ids.npy', allow_pickle=True)
interaction_matrix = load_npz('data/processed/interaction_matrix.npz')

# song_names = joblib.load('models/song_names.joblib')

st.title("Music Recommendation System")
st.write("#### Welcome to the Music Recommendation System! Please enter your preferences to get personalized music recommendations.")
st.write("Here we have some like 50683 songs")

song_artist = df_song['name'].astype(str) + " by " + df_song['artist'].astype(str)
song_artist = song_artist.tolist()

song_name = st.selectbox('Select a song', song_artist)
song_name, artist_name = song_name.split(" by ")
k = st.selectbox('Select number of recommendations', [5, 10, 15, 20], index=1)

recommandation_type = st.selectbox('Select recommendation type', ['Content-Based', 'Collaborative-Based', 'Hybrid'])

# Get recommendations
if st.button('Get Recommendations'):
    try:
        if recommandation_type == 'Collaborative-Based':
            recommendations = collaborative_recommand(song_name,artist_name,track_ids,df_song, interaction_matrix, k)
        elif recommandation_type == 'Hybrid':
            # Use full song corpus for content similarity and align with track_ids for collaborative mixing
            hybrid_recommender = HybridRecommender(song_name, artist_name, df_song, transformed_data, track_ids, interaction_matrix, k)
            recommendations = hybrid_recommender.get_recommendations()
        else:
            recommendations = content_based_recommand(song_name, df_song, transformed_data, k)
        # st.write(recommendations)
        st.write("#### Recommended Songs:")
        
        for idx, rec in recommendations.iterrows():
            song_name = rec['name'].title()
            artist_name = rec['artist'].title()
            url = rec['spotify_preview_url']
            
            if idx == 0:
                st.markdown(f"**{song_name}** by **{artist_name}** (Your Selection)")
                st.audio(url)
                st.write("---")
            else:
                st.markdown(f"**{song_name}** by **{artist_name}**")
                st.audio(url)
                st.write("---")
                
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")

# Background styling
# yt_api_key = AIzaSyD9_E7Lekg7HjneleI287bpWWxMdTgf5Gw
# st.set_page_config(layout="wide")
b1 = "https://images.unsplash.com/photo-1478760329108-5c3ed9d495a0?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
b2 = "https://images.unsplash.com/photo-1563832528262-15e2bca87584?q=80&w=2019&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{b1}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: -1;
    }}
    header {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True
)