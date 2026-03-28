import os
import logging
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

logger = logging.getLogger('content_based_recommender_logger')
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Ensure log directory exists
os.makedirs('reports/logs', exist_ok=True)

# File handler
file_handler = logging.FileHandler('reports/logs/content_sys.log')
file_handler.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Adding handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# -----------------------------------------------------------------------------
# Recommandation Functions
# -----------------------------------------------------------------------------

def content_based_recommand(song_name, df_song, df_transformed, k=10):
    logger.info("Starting content-based recommendation process.")
    song_row = df_song[df_song['name'] == song_name]
    if song_row.empty:
        print(f"Song '{song_name}' not found in the dataset.")
    else:
        song_index = song_row.index[0]
        song_input = df_transformed[song_index].reshape(1, -1)
        similarity_score = cosine_similarity(song_input,df_transformed)
        top_k_indices = np.argsort(similarity_score.ravel())[-k-1:][::-1]
        top_k_songs = df_song.iloc[top_k_indices]
        top_k_list = top_k_songs[['name','artist','spotify_preview_url']].reset_index(drop=True)
        logger.info("Recommendations generated successfully.")
        return top_k_list
        
# recommand('Why Wait',df_song=df_song,df_transformed=df_transformed,k=10) example of recommandation function usage

from pathlib import Path
os.makedirs('.dvc_markers', exist_ok=True)
Path(".dvc_markers/stage_content.done").touch()