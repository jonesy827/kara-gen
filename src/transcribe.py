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
from .audio_processor import AudioProcessor

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
    parser.add_argument('--timestamp', action='store_true', help='Add timestamp to output filenames')
    parser.add_argument('--break-threshold', type=float, default=5.0,
                        help='Time in seconds to consider as an instrumental break (default: 5.0)')
    parser.add_argument('--process-audio', action='store_true', help='Generate karaoke audio tracks (lead vocals, backing vocals, no vocals)')
    parser.add_argument('--audio-only', action='store_true', help='Only process audio separation without transcription')
    parser.add_argument('--lyrics-only', action='store_true', help='Only fetch and save lyrics without audio processing or transcription')
    args = parser.parse_args()

    # If lyrics-only mode, just fetch and save lyrics then exit
    if args.lyrics_only:
        if not args.artist or not args.track:
            print("Error: --artist and --track are required when using --lyrics-only")
            sys.exit(1)
            
        print("\nFetching lyrics...")
        lyrics = get_lyrics(args.artist, args.track)
        if not lyrics:
            print("\nFailed to fetch lyrics")
            sys.exit(1)

        # Load the Whisper model for transcription
        print(f"\nLoading Whisper model '{args.model}'...")
        model = whisper.load_model(args.model).cpu()  # Force CPU mode
        print("Using CPU for transcription")

        # Transcribe with word-level timestamps
        print(f"\nTranscribing '{args.audio_file}'...")
        result = model.transcribe(args.audio_file, word_timestamps=True)

        # Extract words with timestamps
        segments = result.get('segments', [])
        words = []
        for segment in segments:
            for word_info in segment.get('words', []):
                words.append({
                    'word': word_info['word'].strip(),
                    'start': word_info['start'],
                    'end': word_info['end']
                })

        # Match words to lyrics
        words = match_words_to_lyrics(words, lyrics)

        # Create output data structure
        output_data = {
            'metadata': {
                'artist': args.artist,
                'track': args.track,
                'model': args.model,
                'audio_file': args.audio_file,
                'original_lyrics': lyrics
            },
            'words': words
        }

        # Create output directory and save files
        output_dir = create_output_directory(args.artist, args.track)
        
        # Save raw lyrics
        lyrics_file = os.path.join(output_dir, "lyrics.txt")
        with open(lyrics_file, 'w', encoding='utf-8') as f:
            f.write(lyrics)
        print(f"\nLyrics saved to: {lyrics_file}")

        # Generate and save LRC file
        lrc_file = os.path.join(output_dir, "lyrics.lrc")
        generate_lrc_file(output_data, lrc_file, break_threshold=args.break_threshold)
        print(f"LRC file saved to: {lrc_file}")

        # Save transcription data
        json_file = os.path.join(output_dir, "transcription.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Transcription data saved to: {json_file}")
        
        sys.exit(0)

    # If audio-only mode, just do the audio processing and exit
    if args.audio_only:
        if not args.artist or not args.track:
            print("Error: --artist and --track are required when using --audio-only")
            sys.exit(1)
        
        # Convert audio file path to absolute path and handle spaces
        audio_file = os.path.abspath(args.audio_file)
        if not os.path.exists(audio_file):
            print(f"Error: Audio file not found: {audio_file}")
            sys.exit(1)
            
        output_dir = create_output_directory(args.artist, args.track)
        print("\nProcessing audio tracks...")
        try:
            processor = AudioProcessor(output_dir)
            output_files = processor.process_audio(str(audio_file), str(args.artist), str(args.track))
            print("\nGenerated audio files:")
            for track_type, file_path in output_files.items():
                print(f"- {track_type}: {os.path.basename(file_path)}")
            sys.exit(0)
        except Exception as e:
            print(f"\nError processing audio: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Determine output file path
    output_file = args.output
    if output_file is None:
        output_file = os.path.splitext(args.audio_file)[0] + '.json'

    # Add timestamp to filenames if requested
    timestamp = ''
    if args.timestamp:
        from datetime import datetime
        timestamp = datetime.now().strftime('_%Y%m%d_%H%M%S')

    # Create output directory if artist and track are provided
    if args.artist and args.track:
        output_dir = create_output_directory(args.artist, args.track)
        if args.skip_transcription:
            # When skipping transcription, use original JSON as input but create new timestamped output
            input_json = os.path.join(output_dir, 'transcription.json')
            json_file = os.path.join(output_dir, f'transcription{timestamp}.json')
            lrc_file = os.path.join(output_dir, f'lyrics{timestamp}.lrc')
        else:
            json_file = os.path.join(output_dir, f'transcription{timestamp}.json')
            lrc_file = os.path.join(output_dir, f'lyrics{timestamp}.lrc')

        # Process audio if requested
        if args.process_audio:
            print("\nProcessing audio tracks...")
            try:
                processor = AudioProcessor(output_dir)
                output_files = processor.process_audio(args.audio_file, args.artist, args.track)
                print("\nGenerated audio files:")
                for track_type, file_path in output_files.items():
                    print(f"- {track_type}: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"\nError processing audio: {str(e)}")
                if not args.skip_transcription:
                    print("Continuing with transcription...")
    else:
        if args.skip_transcription:
            # Use input file as is, but create new timestamped output
            input_json = output_file
            base, ext = os.path.splitext(output_file)
            json_file = f"{base}{timestamp}.json"
            lrc_file = f"{base}{timestamp}.lrc"
        else:
            base, ext = os.path.splitext(output_file)
            json_file = f"{base}{timestamp}.json"
            lrc_file = f"{base}{timestamp}.lrc"

    try:
        if args.skip_transcription:
            # Validate and load existing JSON
            print(f"\nValidating existing JSON file: {input_json}")
            is_valid, result = validate_json_file(input_json)
            if not is_valid:
                print(f"Error: {result}")
                sys.exit(1)
            output_data = result
            print("JSON file validated successfully")
            
            # Copy the validated JSON to the new timestamped location
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
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
            model = whisper.load_model(args.model).cpu()  # Force CPU mode
            print("Using CPU for transcription")
            
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
            result = model.transcribe(args.audio_file, word_timestamps=True, fp32=True)

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
        generate_lrc_file(output_data, lrc_file, break_threshold=args.break_threshold)
        print(f"LRC file saved to: {lrc_file}")

        print("\nTranscription completed successfully!")
        print(f"Results saved to: {json_file}")

    except Exception as e:
        print(f"\nError during transcription: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 