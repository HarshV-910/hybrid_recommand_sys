# 🎵 Enhanced Music Recommendation System - Setup Guide

## What's New ✨

Your music recommendation app has been upgraded with:

✅ **Beautiful Recommendation Cards** - Songs displayed in attractive cards with thumbnails
✅ **YouTube Integration** - Automatically fetches song videos and thumbnails
✅ **Dual Playback Options** - Listen via Spotify preview or watch on YouTube
✅ **Two-Column Layout** - Modern card-based UI for better browsing
✅ **Smooth Hover Effects** - Enhanced user experience with interactive elements

## Features

### 1. **Recommendation Cards**
- Shows song thumbnail (from YouTube)
- Song title and artist name
- Dual playback tabs (Audio & Video)
- Hover effects for better interactivity

### 2. **Audio Player (Spotify)**
- Plays 30-second preview clips
- Falls back gracefully if preview unavailable

### 3. **Video Player (YouTube)**
- Embedded YouTube player
- Click any card to watch the full music video
- Shows video directly in the app

## Setup Instructions

### Step 1: Install Dependencies ✓ (Already Done)
```bash
pip install streamlit google-api-python-client google-auth-oauthlib python-dotenv
```

### Step 2: Get YouTube API Key
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project
3. Enable YouTube Data API v3
4. Create an API key (in Credentials section)
5. Copy your API key

### Step 3: Configure .env File ✓ (Already Done)
The `.env` file already has your API key:
```env
YOUTUBE_API_KEY=Your_API_Key_Here
```

## Running the App

```bash
# Navigate to project directory
cd /home/harsh/MLops/hybrid_recommand_sys

# Activate hrs_env (your working environment)
source hrs_env/bin/activate

# Run Streamlit app
streamlit run app.py
```

The app will open at `http://localhost:8501`

## How to Use

1. **Select a Song** from the sidebar dropdown
2. **Choose Number of Recommendations** (5, 10, 15, or 20)
3. **Click "Get Recommendations"** button
4. **Browse Cards** - Each card shows:
   - Song thumbnail from YouTube
   - Song title and artist
   - Two tabs:
     - 🎵 **Audio**: 30-second Spotify preview
     - 🎬 **Video**: Full YouTube music video

## Troubleshooting

### Issue: "Could not fetch YouTube video"
- **Solution**: Check if YouTube API key is valid and quota not exceeded
- Visit [Google Cloud Console](https://console.cloud.google.com) to check usage

### Issue: No preview available
- **Solution**: Some songs don't have Spotify previews. Use YouTube video tab instead.

### Issue: Slow loading
- **Solution**: API calls take time. YouTube search happens for each song.
  First load might take 30-60 seconds depending on number of recommendations.

## Customization

### Change Card Layout
Edit line in `app.py` to change columns:
```python
cols = st.columns(3)  # Change 2 to 3 for 3-column layout
```

### Modify Colors
The Spotify green (`#1DB954`) is used for branding. Change all occurrences to your preferred color.

### Add More Tabs
Add more playback options by duplicating the tab structure:
```python
with tab3:
    # Your custom player here
    pass
```

## File Structure
```
hybrid_recommand_sys/
├── app.py                 # Main Streamlit app (UPDATED)
├── .env                   # API keys (UPDATED)
├── requirements.txt       # Dependencies (UPDATED)
├── SETUP_GUIDE.md        # This file
├── data/                  # Data directory
├── models/                # ML models
└── src/                   # Source code
```

## Performance Tips

- **Reduce recommendations** to 5-10 for faster loading
- **First app load** takes longer due to YouTube API calls
- **Subsequent loads** are faster due to Streamlit caching

Enjoy your enhanced music recommendation system! 🎶
