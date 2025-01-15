# Current Status: Implementing Audio Separation in kara-gen

We're adding audio separation functionality to kara-gen using audio-separator. The code structure is in place, but we're encountering an API issue with the Separator class.

## What's Working
- Basic project structure is set up
- Dependencies are installed (including audio-separator and onnxruntime)
- Command line interface with --audio-only flag is implemented
- File path handling and directory creation is working

## Current Issue
We're getting an error because we're using `separate_into_files` but the API seems to be different:
```python
Error processing audio: 'Separator' object has no attribute 'separate_into_files'
```

## Next Steps Needed
1. Check the correct API for audio-separator's Separator class
2. Update the `audio_processor.py` code to use the correct method for separation
3. Test the audio separation with a sample file

## Key Files Modified
1. `src/audio_processor.py` - Main audio separation logic
2. `src/transcribe.py` - CLI interface and audio-only mode
3. `requirements.txt` - Added audio-separator and dependencies

## Test Command Being Used
```bash
python3 -m src.transcribe "Billy Strings - Leaning on a Travelin' Song.flac" --artist "Billy Strings" --track "Leaning on a Travelin' Song" --audio-only
```

The next step is to determine the correct API for the Separator class and update our code accordingly. 