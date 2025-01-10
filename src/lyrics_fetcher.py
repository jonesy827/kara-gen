"""Module for fetching lyrics from external APIs."""

import requests

def get_lyrics(artist_name, track_name):
    """Fetch lyrics from lrclib.net API.
    
    Args:
        artist_name (str): Name of the artist
        track_name (str): Name of the track
        
    Returns:
        str or None: Lyrics if found, None otherwise
    """
    query = f"{artist_name} {track_name}".replace(' ', '+')
    api_url = f"https://lrclib.net/api/search?q={query}"
    print(f"Fetching lyrics from: {api_url}")
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        if data:
            lyrics = data[0]['plainLyrics']
            print(f"Lyrics retrieved successfully for {artist_name} - {track_name}")
            return lyrics
        else:
            print(f"No lyrics found for {artist_name} - {track_name}")
            return None
    except Exception as e:
        print(f"An error occurred while fetching lyrics: {e}")
        return None 