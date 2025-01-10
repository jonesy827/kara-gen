# transcribe.py

import sys
import argparse
import os
import json
import requests
from Levenshtein import distance
import whisper
import torch

def clean_word(word):
    """Clean a word by removing punctuation and converting to lowercase."""
    return ''.join(c.lower() for c in word if c.isalnum())

def clean_line(line):
    """Clean a line of text for comparison."""
    return ' '.join(clean_word(word) for word in line.split())

def format_timestamp(seconds):
    """Convert seconds to LRC timestamp format [mm:ss.xx]"""
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:05.2f}"

def line_similarity(line1, line2):
    """Calculate similarity between two lines using word overlap and Levenshtein."""
    # Clean and split into words
    words1 = set(clean_line(line1).split())
    words2 = set(clean_line(line2).split())
    
    # Calculate word overlap
    common_words = words1 & words2
    overlap_score = len(common_words) / max(len(words1), len(words2))
    
    # Calculate Levenshtein similarity for the full lines
    clean1 = clean_line(line1)
    clean2 = clean_line(line2)
    max_len = max(len(clean1), len(clean2))
    if max_len == 0:
        return 0
    lev_score = 1 - (distance(clean1, clean2) / max_len)
    
    # Combine scores with more weight on word overlap
    return (0.7 * overlap_score) + (0.3 * lev_score)

def get_lyrics(artist_name, track_name):
    query = f"{artist_name} {track_name}".replace(' ', '+')
    api_url = f"https://lrclib.net/api/search?q={query}"
    print(f"Fetching lyrics from: {api_url}")
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        if data:
            lyrics = data[0]['plainLyrics']
            print(f"Lyrics retrieved successfully for {artist_name} - {track_name}:\n{lyrics}")
            return lyrics
        else:
            print(f"No lyrics found for {artist_name} - {track_name}")
            return None
    except Exception as e:
        print(f"An error occurred while fetching lyrics: {e}")
        return None

def match_words_to_lyrics(transcribed_words, lyrics):
    """Match transcribed words to lyrics using Levenshtein distance."""
    if not lyrics:
        return transcribed_words
    
    # Split lyrics into words and clean them
    lyrics_words = [clean_word(word) for word in lyrics.replace('\n', ' ').split() if clean_word(word)]
    
    # Process each transcribed word
    matched_words = []
    lyrics_idx = 0
    
    for word_info in transcribed_words:
        transcribed_word = clean_word(word_info['word'])
        if not transcribed_word:
            continue
            
        # Look for the best match in the next few lyrics words
        best_match = None
        best_distance = float('inf')
        search_range = min(10, len(lyrics_words) - lyrics_idx)
        
        for i in range(search_range):
            current_lyrics_word = lyrics_words[lyrics_idx + i]
            current_distance = distance(transcribed_word, current_lyrics_word)
            
            # If we find an exact match or a very close match
            if current_distance < best_distance and current_distance <= 2:
                best_distance = current_distance
                best_match = current_lyrics_word
                best_idx = i
        
        # If we found a good match, update the word and advance the lyrics index
        if best_match:
            word_info['word'] = best_match
            word_info['original_word'] = transcribed_word
            word_info['confidence'] = 1 - (best_distance / max(len(best_match), len(transcribed_word)))
            lyrics_idx += best_idx + 1
        else:
            word_info['confidence'] = 0.0
            
        matched_words.append(word_info)
    
    return matched_words

def create_output_directory(artist, track):
    """Create and return path to output directory based on artist and track name."""
    # Clean names to be filesystem friendly
    artist = ''.join(c for c in artist if c.isalnum() or c in (' -_'))
    track = ''.join(c for c in track if c.isalnum() or c in (' -_'))
    
    # Create directory path
    dir_path = os.path.join('output', f"{artist} - {track}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def generate_lrc_file(words_data, output_path):
    """Generate an enhanced LRC file with word-level timing."""
    lrc_lines = []
    
    # Add metadata
    lrc_lines.append("[ar:{}]".format(words_data['metadata']['artist']))
    lrc_lines.append("[ti:{}]".format(words_data['metadata']['track']))
    lrc_lines.append("[length:{}]".format(words_data['words'][-1]['end']))
    
    # Split original lyrics into lines and clean them
    original_lyrics = words_data['metadata'].get('original_lyrics', '')
    if original_lyrics:
        lyrics_lines = [line.strip() for line in original_lyrics.split('\n') if line.strip()]
        # Clean and normalize each line for comparison
        clean_lyrics_lines = [clean_line(line) for line in lyrics_lines]
    else:
        lyrics_lines = []
        clean_lyrics_lines = []
    
    # Process transcribed words
    words = words_data['words']
    current_line = []
    line_start = words[0]['start'] if words else 0
    word_idx = 0
    
    while word_idx < len(words):
        word = words[word_idx]
        current_line.append(word)
        
        # Get the text of the current line
        current_text = ' '.join(w['word'] for w in current_line)
        current_clean = clean_line(current_text)
        
        # Decide if we should end the line
        end_line = False
        
        # First check if we match a lyrics line
        if clean_lyrics_lines:
            best_match = None
            best_score = 0
            
            for i, lyrics_line in enumerate(clean_lyrics_lines):
                if not lyrics_line:  # Skip used lines
                    continue
                
                # Calculate similarity score
                score = line_similarity(current_clean, lyrics_line)
                if score > best_score and score > 0.5:  # Require 50% similarity
                    best_score = score
                    best_match = i
            
            # If we found a good match
            if best_match is not None:
                end_line = True
                clean_lyrics_lines[best_match] = ''  # Mark as used
        
        # If no lyrics match and we have a significant pause, also end the line
        elif word_idx < len(words) - 1:
            next_word = words[word_idx + 1]
            if next_word['start'] - word['end'] > 1.5:
                end_line = True
        
        # Write line if needed
        if end_line or word_idx == len(words) - 1:
            if current_line:
                timestamp = format_timestamp(line_start)
                line_text = ' '.join(f"<{format_timestamp(w['start'])}>{w['word']}" for w in current_line)
                lrc_lines.append(f"[{timestamp}]{line_text}")
                current_line = []
                if word_idx < len(words) - 1:
                    line_start = words[word_idx + 1]['start']
        
        word_idx += 1
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lrc_lines))

def validate_json_file(file_path):
    """Validate that the JSON file exists and has the required structure."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Check required fields
        if not isinstance(data, dict):
            return False, "JSON root must be an object"
        if 'metadata' not in data:
            return False, "Missing 'metadata' field"
        if 'words' not in data:
            return False, "Missing 'words' field"
        if not isinstance(data['words'], list):
            return False, "Words must be an array"
        if not data['words']:
            return False, "Words array is empty"
            
        # Check word structure
        for word in data['words']:
            if not all(k in word for k in ('word', 'start', 'end')):
                return False, "Words must have 'word', 'start', and 'end' fields"
        
        return True, data
    except FileNotFoundError:
        return False, f"File not found: {file_path}"
    except json.JSONDecodeError:
        return False, f"Invalid JSON in file: {file_path}"
    except Exception as e:
        return False, f"Error reading file: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Transcribe audio files with word-level timestamps using OpenAI Whisper.')
    parser.add_argument('audio_file', help='Path to the audio file to transcribe.')
    parser.add_argument('--output', default=None, help='Path to the output file. If not specified, will use the input filename with .json extension.')
    parser.add_argument('--model', default='medium', choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='The size of the Whisper model to use. Larger models are more accurate but require more resources.')
    parser.add_argument('--artist', help='Artist name for lyrics lookup')
    parser.add_argument('--track', help='Track name for lyrics lookup')
    parser.add_argument('--skip-transcription', action='store_true', help='Skip transcription and use existing JSON file to generate LRC')
    args = parser.parse_args()

    # Determine output file path
    output_file = args.output
    if output_file is None:
        output_file = os.path.splitext(args.audio_file)[0] + '.json'

    # Create output directory if artist and track are provided
    if args.artist and args.track:
        output_dir = create_output_directory(args.artist, args.track)
        json_file = os.path.join(output_dir, 'transcription.json')
        lrc_file = os.path.join(output_dir, 'lyrics.lrc')
    else:
        json_file = output_file
        lrc_file = os.path.splitext(output_file)[0] + '.lrc'

    try:
        if args.skip_transcription:
            # Validate and load existing JSON
            print(f"\nValidating existing JSON file: {json_file}")
            is_valid, result = validate_json_file(json_file)
            if not is_valid:
                print(f"Error: {result}")
                sys.exit(1)
            output_data = result
            print("JSON file validated successfully")
        else:
            # Set up device
            device = "cpu"
            try:
                if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                    device = "mps"
                    print("Using MPS (Metal) acceleration")
                else:
                    print("Using CPU (MPS acceleration not available)")
            except:
                print("Using CPU (Error checking MPS availability)")

            # Load the Whisper model
            print(f"\nLoading Whisper model '{args.model}'...")
            model = whisper.load_model(args.model)
            
            try:
                if device == "mps":
                    model = model.to(device)
                    print("Successfully moved model to MPS device")
            except Exception as e:
                print("Failed to use MPS, falling back to CPU")
                device = "cpu"
                model = model.to("cpu")

            print(f"\nTranscribing '{args.audio_file}'...")
            print(f"Output will be saved to '{json_file}'")

            # Get lyrics if artist and track are provided
            lyrics = None
            original_lyrics = None
            if args.artist and args.track:
                lyrics = get_lyrics(args.artist, args.track)
                original_lyrics = lyrics

            # Transcribe with word-level timestamps
            result = model.transcribe(args.audio_file, word_timestamps=True)

            # Extract the segments and create word list
            segments = result.get('segments', [])
            words = []
            
            for segment in segments:
                for word_info in segment.get('words', []):
                    words.append({
                        'word': word_info['word'].strip(),
                        'start': word_info['start'],
                        'end': word_info['end']
                    })

            # Match words to lyrics if available
            if lyrics:
                words = match_words_to_lyrics(words, lyrics)

            # Create output data structure
            output_data = {
                'metadata': {
                    'artist': args.artist,
                    'track': args.track,
                    'model': args.model,
                    'audio_file': args.audio_file,
                    'original_lyrics': original_lyrics
                },
                'words': words
            }

            # Save to output file
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

        # Generate LRC file
        generate_lrc_file(output_data, lrc_file)
        print(f"LRC file saved to: {lrc_file}")

        print("\nTranscription completed successfully!")
        print(f"Results saved to: {json_file}")

    except Exception as e:
        print(f"\nError during transcription: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
