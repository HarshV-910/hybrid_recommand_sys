# Use official Python base image
FROM python:3.10

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app/

# Copy requirements and install dependencies
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --default-timeout=300 -r requirements.txt

COPY ./data/processed/collab_filtered.csv \
     ./data/processed/interaction_matrix.npz \
     ./data/processed/df_cleaned.csv \
     ./data/processed/df_transformed.npz \
     ./data/processed/

COPY ./models/track_ids.npy \
     ./models/song_names.joblib \
     ./models/transformer.joblib \
     ./models/

COPY ./data/raw/Music_Info.csv ./data/raw/

COPY app.py .
COPY ./src/ /app/src/

EXPOSE 8000

CMD ["streamlit", "run", "app.py", "--server.port", "8000"]

# file structure will looks like:
# /app
# │
# ├── requirements.txt
# ├── app.py
# │
# ├── src/
# │   ├── ... (your full source code)
# │
# ├── data/
# │   ├── raw/
# │   │   └── Music_Info.csv
# │   │
# │   └── processed/
# │       ├── collab_filtered.csv
# │       ├── interaction_matrix.npz
# │       ├── df_cleaned.csv
# │       └── df_transformed.npz
# │
# ├── models/
# │   ├── track_ids.npy
# │   ├── song_names.joblib
# │   └── transformer.joblib
# │
# └── (Python packages installed globally)