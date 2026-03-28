import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# from .collaborative_sys import collaborative_recommand
# from .content_sys import content_based_recommand
# from scipy.sparse import load_npz


class HybridRecommender:
    def __init__(self,
                 song_name: str,
                 artist_name: str,
                 df_songs: pd.DataFrame,
                 transformed_matrix: np.ndarray,
                 track_ids: np.ndarray,
                 interaction_matrix: np.ndarray,
                 no_of_recommendations: int = 10,
                 weight_collaborative: float = 0.3, # type: ignore
                 weight_content: float = 0.8):
    
        self.song_name = song_name
        self.artist_name = artist_name
        self.no_of_recommendations = no_of_recommendations
        self.weight_collaborative = weight_collaborative
        self.weight_content = weight_content
        self.df_songs = df_songs
        self.transformed_matrix = transformed_matrix
        self.track_ids = track_ids
        self.interaction_matrix = interaction_matrix
        
    
    def content_based_similarity(self,song_name, artist_name, df_songs, transformed_matrix):
        song_row = df_songs.loc[(df_songs['name'] == song_name) & (df_songs['artist'] == artist_name)]
        song_index = song_row.index[0]
        input_vector = transformed_matrix[song_index].reshape(1, -1)
        content_similarity_score = cosine_similarity(input_vector, transformed_matrix).ravel()
        return content_similarity_score
    
    def collaborative_based_similarity(self,song_name, artist_name, track_ids, df_songs, interaction_matrix):

        song_row = df_songs.loc[(df_songs['name'] == song_name) & (df_songs['artist'] == artist_name)]
        input_track_id = song_row['track_id'].values.item()
        idx = np.where(track_ids == input_track_id)[0].item()
        input_array = interaction_matrix[idx]
        collab_similarity_score = cosine_similarity(input_array, interaction_matrix).ravel()
        return collab_similarity_score
    
    def normalize_scores(self, scores):
        if np.max(scores) - np.min(scores) == 0:
            return np.zeros_like(scores)
        normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        return normalized_scores
    
    def weighted_combination(self, content_scores, collab_scores):
        weighted_scores = (self.weight_content * content_scores) + (self.weight_collaborative * collab_scores)
        return weighted_scores
    
    def get_recommendations(self):
        # content scores over all songs passed in df_songs / transformed_matrix
        content_scores_full = self.content_based_similarity(self.song_name, self.artist_name, self.df_songs, self.transformed_matrix)

        # collaborative scores over track_ids + interaction_matrix (subspace)
        collab_scores = self.collaborative_based_similarity(self.song_name, self.artist_name, self.track_ids, self.df_songs, self.interaction_matrix)

        if len(content_scores_full) != len(self.df_songs):
            raise ValueError(f"df_songs length ({len(self.df_songs)}) and transformed_matrix length ({len(content_scores_full)}) mismatch")

        # align content scores to the track_ids ordering used by interaction_matrix
        if self.df_songs['track_id'].nunique() != len(self.df_songs):
            raise ValueError("df_songs contains duplicate track_id. Hybrid strategy requires unique track ids for alignment.")

        track_index_map = pd.Series(self.df_songs.index.values, index=self.df_songs['track_id']).reindex(self.track_ids)
        if track_index_map.isna().any():
            missing = self.track_ids[track_index_map.isna().values]
            raise ValueError(f"The following track_ids are missing from df_songs: {missing[:10]}")

        content_scores = content_scores_full[track_index_map.astype(int).values]

        normalized_content_scores = self.normalize_scores(content_scores)
        normalized_collab_scores = self.normalize_scores(collab_scores)

        final_scores = self.weighted_combination(normalized_content_scores, normalized_collab_scores)

        top_indices = np.argsort(final_scores)[::-1][:self.no_of_recommendations]
        recommended_track_ids = self.track_ids[top_indices]

        # preserve ranking order in result
        recommended_songs = self.df_songs[self.df_songs['track_id'].isin(recommended_track_ids)]
        recommended_songs = recommended_songs.set_index('track_id').loc[recommended_track_ids].reset_index()

        return recommended_songs
    
        