import os
import logging
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

logger = logging.getLogger('collaborative_based_recommender_logger')
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Ensure log directory exists
os.makedirs('reports/logs', exist_ok=True)

# File handler
file_handler = logging.FileHandler('reports/logs/collaborative_based_recommender.log')
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

def collaborative_recommand(song_name, artist_name, track_ids, df_song, interaction_matrix, k=10):
  song_row = df_song.loc[(df_song['name'] == song_name) & (df_song['artist'] == artist_name)]
  if song_row.empty:
    logger.warning(f"Song not found for collaborative recommendation: {song_name} by {artist_name}")
    return pd.DataFrame(columns=df_song.columns)

  input_track_id = song_row['track_id'].values.item()
  idx_matches = np.where(track_ids == input_track_id)[0]

  ind = idx_matches[0]
  input_array = interaction_matrix[ind]

  similarity_score = cosine_similarity(input_array, interaction_matrix).ravel()

  # exclude the query track itself and get top k
  top_idx = np.argsort(similarity_score)[::-1] # reverse sort
  top_idx = top_idx[:k+1]

  recommandation = track_ids[top_idx]
  top = similarity_score[top_idx]

  temp_df = pd.DataFrame({'track_id': recommandation.tolist(), 'score': top})
  top_k_song = (
      df_song
      .loc[df_song['track_id'].isin(recommandation)]
      .merge(temp_df, on='track_id')
      .sort_values(by='score', ascending=False)
      .drop(columns=['track_id', 'score'])
      .reset_index(drop=True)
  )

  return top_k_song

# collaborative_recommand(song_name='Mr. Brightside', user_data=df_user, df_song=df_song, interaction_matrix=sparse_matrix)