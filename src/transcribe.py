"""Main script for transcribing audio files and generating LRC files."""

import sys
import argparse
import os
import json
import torch
import whisper

from .lyrics_fetcher import get_lyrics
from .word_matcher import match_words_to_lyrics
from .lrc_generator import create_output_directory, generate_lrc_file

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