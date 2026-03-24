import os
import logging
import pandas as pd
import numpy as np
from category_encoders.count import CountEncoder
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
import yaml
from scipy.sparse import save_npz
import joblib

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
# Recommandation Functions
# -----------------------------------------------------------------------------

def recommand(song_name, df_song, df_transformed, k=10):
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
    return top_k_list
# recommand('Why Wait',df_song=df_song,df_transformed=df_transformed,k=10) example of recommandation function usage

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
# Main Execution Function
# -----------------------------------------------------------------------------

def main():
    try:

        param_path = 'params.yaml'
        data_path = os.path.join("data", "raw")
        processed_path = os.path.join("data", "processed")
        os.makedirs(processed_path, exist_ok=True)

        # Load datasets
        data_file = os.path.join(data_path, "Music_Info.csv")
        df_song = pd.read_csv(data_file)
        logger.info("Dataset loaded successfully.")
        song_names = df_song['name'].unique().tolist()

        col_to_clean = ['spotify_id','track_id','genre','name','spotify_preview_url']
        df_cleaned = clean(df_song,col_to_clean)


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
        joblib.dump(song_names, "models/song_names.joblib")
        logger.info(f"Cleaned & Transformed data saved to {processed_path}")





        logger.info(f"Transformed data saved to {processed_path}")

    except Exception as e:
        logger.critical(f"Data preprocessing pipeline failed: {e}")
        raise

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
