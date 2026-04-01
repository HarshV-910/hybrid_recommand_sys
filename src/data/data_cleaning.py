import os
import logging
import pandas as pd
import numpy as np
from category_encoders.count import CountEncoder
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
import yaml
from scipy.sparse import save_npz
import joblib
import dask.dataframe as dd
from scipy.sparse import csr_matrix

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

logger = logging.getLogger('data_cleaning_logger')
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Ensure log directory exists
os.makedirs('reports/logs', exist_ok=True)

# File handler
file_handler = logging.FileHandler('reports/logs/data_cleaning.log')
file_handler.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Adding handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# -----------------------------------------------------------------------------
# Parameter Loading
# -----------------------------------------------------------------------------

def load_params(param_path: str) -> int:
    """Load 'max_features' for vectorizer from YAML config."""
    try:
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)
            max_features = params['data_cleaning']['max_features']
            logger.info(f"Loaded max_features: {max_features}")
            return max_features
    except FileNotFoundError:
        logger.error(f"Parameter file not found at: {param_path}")
        raise
    except (yaml.YAMLError, KeyError) as e:
        logger.error(f"Error parsing parameters from {param_path}: {e}")
        raise


# -----------------------------------------------------------------------------
# Data Processing Functions
# -----------------------------------------------------------------------------

def clean(df,col):
    try:
        logger.info("Starting data cleaning.")
        df.reset_index(drop=True,inplace=True)
        # droping some column which have unique values
        df.drop(columns=col,errors='ignore', inplace=True)
        logger.info(f"Columns {col} dropped successfully.")
        # replacing null values in tags column with no tag
        df['tags'] = df['tags'].fillna('no_tag')
        # converting some columns in lower case
        df['artist'] = df['artist'].str.lower()
        # convert year to category for frequency encoding
        df['year'] = df['year'].astype('category')
        logger.info("Data cleaning completed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        raise

# transform the data
def transform_data(df,freq_encode_col,ohe_cols,tfidf_col,standard_encode_col,min_max_scale_col,mf):
    try:
        scaler1 = MinMaxScaler()
        scaler2 = StandardScaler()
        ohe = OneHotEncoder(handle_unknown='ignore')
        tfidf = TfidfVectorizer(stop_words='english',max_features=mf)

        logger.info("Starting data transformation.")
        transformer = ColumnTransformer([
            ('freq_encode',CountEncoder(normalize=True,return_df=True),freq_encode_col),
            ('ohe',ohe,ohe_cols),
            ('tfidf',tfidf, tfidf_col),
            ('standard_encode',scaler2,standard_encode_col),
            ('min_max_scale',scaler1,min_max_scale_col)
        ],remainder='passthrough',n_jobs=-1,force_int_remainder_cols=False)
        
        transformed_data = transformer.fit_transform(df)
        logger.info("Data transformation completed successfully.")
        
        os.makedirs('models', exist_ok=True)
        joblib.dump(transformer, 'models/transformer.joblib')

        return transformed_data
    except Exception as e:
        logger.error(f"Data transformation failed: {e}")
        raise

# -----------------------------------------------------------------------------
# interaction matrix create Function
# -----------------------------------------------------------------------------

def create_interaction_matrix(df:dd.DataFrame, track_ids) -> csr_matrix:
    logger.info("Starting interaction matrix creation.")

    # if Dask DF, convert to pandas to avoid partial aggregation/categorial matrix issues
    if isinstance(df, dd.DataFrame):
        df = df.compute()

    # ensure playcount is numeric
    # df['playcount'] = df['playcount'].astype(np.float64)
    df['playcount'] = df['playcount'].astype(np.float32)

    # limit to non-zero playcount rows (if any)
    df = df[df['playcount'] > 0]

    # categories for track and user codes
    df['user_id'] = df['user_id'].astype('category')
    df['track_id'] = df['track_id'].astype('category')

    user_ids = df['user_id'].cat.categories.values
    track_ids = df['track_id'].cat.categories.values
    np.save('models/track_ids.npy', track_ids, allow_pickle=True)

    df['user_idx'] = df['user_id'].cat.codes
    df['track_idx'] = df['track_id'].cat.codes

    interaction_array = df.groupby(['track_idx', 'user_idx'])['playcount'].sum().reset_index()

    # row_indices = interaction_array['track_idx'].astype(np.int64).values
    # col_indices = interaction_array['user_idx'].astype(np.int64).values
    # values = interaction_array['playcount'].astype(np.float64).values
    row_indices = interaction_array['track_idx'].astype(np.int32).values
    col_indices = interaction_array['user_idx'].astype(np.int32).values
    values = interaction_array['playcount'].astype(np.float32).values

    # sparse_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(len(track_ids), len(user_ids)))
    sparse_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(len(track_ids), len(user_ids)), dtype=np.float32)
    save_npz('data/processed/interaction_matrix.npz', sparse_matrix)

    logger.info("Interaction matrix created successfully.")
    return sparse_matrix

# -----------------------------------------------------------------------------
# song data filter Function
# -----------------------------------------------------------------------------

def filter_song_data(df_song: pd.DataFrame, track_ids:list) -> pd.DataFrame:
   # filtering the song data to only include songs that are present in the interaction matrix
   logger.info("Starting song data filtering.")
   filtered_df = df_song[df_song['track_id'].isin(track_ids)]
   filtered_df.reset_index(drop=True,inplace=True)
   filtered_df.to_csv('data/processed/collab_filtered.csv', index=False)
   logger.info("Filtered song data saved.")
   return filtered_df

# -----------------------------------------------------------------------------
# Main Execution Function
# -----------------------------------------------------------------------------

def main():
    try:

        param_path = 'params.yaml'
        data_path = os.path.join("data", "raw")
        processed_path = os.path.join("data", "processed")
        os.makedirs(processed_path, exist_ok=True)
        os.makedirs('models', exist_ok=True)

        # Load datasets
        songdata_file = os.path.join(data_path, "Music_Info.csv")
        userdata_file = os.path.join(data_path, "User_Listening_History.csv")
        user_df = dd.read_csv(userdata_file)
        df_song = pd.read_csv(songdata_file)
        # song_df = pd.read_csv(songdata_file, usecols=['track_id','name','artist','spotify_preview_url'])
        # user_df = pd.read_csv(userdata_file)
        logger.info("Dataset loaded successfully.")

# -------- FOR COLLABORATIVE BASED RECOMMENDER SYSTEM ---------

        # getting unique track ids
        unique_track_ids = user_df.loc[:,'track_id'].unique().compute()
        unique_track_ids = unique_track_ids.tolist()

        filter_song_data(df_song, unique_track_ids)
        create_interaction_matrix(user_df, unique_track_ids)

# -------- FOR CONTENT BASED RECOMMENDER SYSTEM --------

        song_names = df_song['name'].unique().tolist()

        col_to_clean = ['spotify_id','track_id','genre','name','spotify_preview_url']
        # Clean a copy so app metadata columns remain available in df_song.
        df_cleaned = clean(df_song.copy(), col_to_clean)

        freq_encode_col = ['year'] # range : 0 to 1
        ohe_cols = ['artist','time_signature','key']
        tfidf_col = 'tags'
        standard_encode_col = ['duration_ms','loudness','tempo']
        min_max_scale_col = ['danceability','energy','speechiness','acousticness','instrumentalness','liveness','valence']
        mf = load_params(param_path)
        df_transformed = transform_data(df_cleaned,freq_encode_col,ohe_cols,tfidf_col,standard_encode_col,min_max_scale_col,mf)

        # Save transformed data

        save_npz(os.path.join(processed_path, "df_transformed.npz"), df_transformed) # to save sparse matrix
        df_cleaned.to_csv(os.path.join(processed_path, "df_cleaned.csv"), index=False)
        # Keep app-facing metadata file aligned with transformed matrix row order.
        df_song.to_csv(os.path.join(processed_path, "Music_Info_app.csv"), index=False)
        joblib.dump(song_names, "models/song_names.joblib")
        logger.info(f"Cleaned & Transformed data saved to {processed_path}")

    except Exception as e:
        logger.critical(f"Data preprocessing pipeline failed: {e}")
        raise

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
