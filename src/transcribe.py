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
    parser.add_argument('--timestamp', action='store_true', help='Add timestamp to output filenames')
    parser.add_argument('--break-threshold', type=float, default=5.0,
                        help='Time in seconds to consider as an instrumental break (default: 5.0)')
    parser.add_argument('--audio-only', action='store_true', help='Only process audio separation without transcription')
    parser.add_argument('--transcribe-only', action='store_true', help='Only transcribe audio and generate JSON, no LRC file')
    parser.add_argument('--lyrics-only', action='store_true', help='Only generate LRC file from existing transcription.json')
    args = parser.parse_args()

    # Validate artist and track are provided when needed
    if args.lyrics_only or args.audio_only:
        if not args.artist or not args.track:
            print("Error: --artist and --track are required when using --lyrics-only or --audio-only")
            sys.exit(1)

    # Create output directory if artist and track are provided
    output_dir = None
    if args.artist and args.track:
        output_dir = create_output_directory(args.artist, args.track)

    # If lyrics-only mode, just generate LRC from existing JSON
    if args.lyrics_only:
        if not output_dir:
            print("Error: --artist and --track are required for --lyrics-only")
            sys.exit(1)

        # Load existing transcription JSON
        json_file = os.path.join(output_dir, 'transcription.json')
        if not os.path.exists(json_file):
            print(f"Error: No transcription.json found in {output_dir}")
            sys.exit(1)

        print(f"\nLoading existing transcription from: {json_file}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                output_data = json.load(f)

            # Load timing info if it exists
            timing_file = os.path.join(output_dir, "timing_info.json")
            if os.path.exists(timing_file):
                with open(timing_file, 'r') as f:
                    timing_info = json.load(f)
                print(f"\nLoaded timing info: {timing_info}")
                start_offset = timing_info.get('start_offset', 0.0)
                if start_offset > 0:
                    print(f"Will adjust timestamps by +{start_offset:.2f} seconds")
                # Ensure timing info is in metadata
                if 'metadata' not in output_data:
                    output_data['metadata'] = {}
                output_data['metadata']['timing_info'] = timing_info

            # Generate LRC file
            lrc_file = os.path.join(output_dir, "lyrics.lrc")
            generate_lrc_file(output_data, lrc_file, break_threshold=args.break_threshold)
            print(f"LRC file saved to: {lrc_file}")
            sys.exit(0)
        except Exception as e:
            print(f"Error processing transcription: {str(e)}")
            sys.exit(1)

    # If audio-only mode, just do the audio processing and exit
    if args.audio_only:
        if not output_dir:
            print("Error: --artist and --track are required for --audio-only")
            sys.exit(1)
        
        # Convert audio file path to absolute path and handle spaces
        audio_file = os.path.abspath(args.audio_file)
        if not os.path.exists(audio_file):
            print(f"Error: Audio file not found: {audio_file}")
            sys.exit(1)
            
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

    # Set up file paths
    if output_dir:
        json_file = os.path.join(output_dir, f'transcription{timestamp}.json')
        lrc_file = os.path.join(output_dir, f'lyrics{timestamp}.lrc')
    else:
        base, ext = os.path.splitext(output_file)
        json_file = f"{base}{timestamp}.json"
        lrc_file = f"{base}{timestamp}.lrc"

    try:
        # Process audio if we have artist and track info
        audio_to_transcribe = args.audio_file
        processed_audio_files = None
        timing_info = {}
        
        if output_dir:
            print("\nProcessing audio tracks...")
            try:
                processor = AudioProcessor(output_dir)
                processed_audio_files = processor.process_audio(args.audio_file, args.artist, args.track)
                print("\nGenerated audio files:")
                for track_type, file_path in processed_audio_files.items():
                    print(f"- {track_type}: {os.path.basename(file_path)}")
                # Use the lead vocals file for transcription
                audio_to_transcribe = processed_audio_files['lead_vocals']
                
                # Load timing info
                timing_file = os.path.join(output_dir, "timing_info.json")
                if os.path.exists(timing_file):
                    with open(timing_file, 'r') as f:
                        timing_info = json.load(f)
                    print(f"\nLoaded timing info: {timing_info}")
                    start_offset = timing_info.get('start_offset', 0.0)
                    if start_offset > 0:
                        print(f"Will adjust timestamps by +{start_offset:.2f} seconds")
            except Exception as e:
                print(f"\nError processing audio: {str(e)}")
                print("Continuing with transcription using original audio...")
                import traceback
                traceback.print_exc()

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

        print(f"\nTranscribing '{audio_to_transcribe}'...")
        print(f"Output will be saved to '{json_file}'")

        # Get lyrics if artist and track are provided and we're not in transcribe-only mode
        lyrics = None
        original_lyrics = None
        if args.artist and args.track and not args.transcribe_only:
            lyrics = get_lyrics(args.artist, args.track)
            original_lyrics = lyrics

        # Transcribe with word-level timestamps
        print(f"\nTranscribing '{audio_to_transcribe}'...")
        try:
            result = model.transcribe(
                audio_to_transcribe, 
                word_timestamps=True,
                condition_on_previous_text=False,
                initial_prompt="This is a song with vocals and instrumental sections. The vocals may start after a long instrumental intro.",
                no_speech_threshold=0.1,  # Much more lenient
                compression_ratio_threshold=2.4,  # Help with music/speech distinction
                logprob_threshold=None  # Don't filter based on confidence
            )
            
            if not result:
                print("Error: Transcription returned no result")
                sys.exit(1)
            
            print("Raw transcription result:", result.keys())
            
            # Save raw transcription for debugging
            if output_dir:
                debug_raw = os.path.join(output_dir, "debug_raw_transcription.json")
                with open(debug_raw, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
                print(f"Saved raw transcription to: {debug_raw}")
            
            segments = result.get('segments', [])
            if not segments:
                print("Error: No segments found in transcription")
                sys.exit(1)
                
            print(f"Found {len(segments)} segments")
            
            words = []
            for i, segment in enumerate(segments):
                print(f"Processing segment {i+1}/{len(segments)}")
                segment_words = segment.get('words', [])
                if not segment_words:
                    print(f"Warning: No words in segment {i+1}")
                    continue
                    
                for word_info in segment_words:
                    word = word_info.get('word', '').strip()
                    # Only skip if it's JUST a period, allow words with periods
                    if not word or word in ['.', ' .', '. ']:
                        print(f"Warning: Skipping empty word or lone period in segment {i+1}")
                        continue
                        
                    # Adjust timestamps if we trimmed silence
                    start_time = word_info['start']
                    end_time = word_info['end']
                    if timing_info.get('start_offset', 0.0) > 0:
                        start_time += timing_info['start_offset']
                        end_time += timing_info['start_offset']
                        
                    words.append({
                        'word': word,
                        'start': start_time,
                        'end': end_time
                    })

            if not words:
                print("Error: No words extracted from transcription")
                sys.exit(1)
            
            print(f"Extracted {len(words)} words from transcription")

            # Match words to lyrics if available and not in transcribe-only mode
            if lyrics and not args.transcribe_only:
                print("\nMatching words to lyrics...")
                try:
                    words = match_words_to_lyrics(words, lyrics)
                    print(f"Successfully matched words to lyrics")
                except Exception as e:
                    print(f"Error during lyrics matching: {str(e)}")
                    print("Continuing with unmatched transcription")
                    import traceback
                    traceback.print_exc()

            # Create output data structure
            output_data = {
                'metadata': {
                    'artist': args.artist,
                    'track': args.track,
                    'model': args.model,
                    'audio_file': args.audio_file,
                    'transcribed_audio': audio_to_transcribe,
                    'original_lyrics': original_lyrics,
                    'processed_audio_files': processed_audio_files,
                    'word_count': len(words),
                    'timing_info': timing_info
                },
                'words': words
            }

            # Save to output file
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\nTranscription saved to: {json_file}")
                
        except Exception as e:
            print(f"\nError during transcription or word processing:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Generate LRC file if not in transcribe-only mode
        if not args.transcribe_only:
            generate_lrc_file(output_data, lrc_file, break_threshold=args.break_threshold)
            print(f"LRC file saved to: {lrc_file}")

        print("\nTranscription completed successfully!")

    except Exception as e:
        print(f"\nError during transcription: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 