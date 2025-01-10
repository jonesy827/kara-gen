# Karaoke LRC Generator Debugging

## Current State

We have a Python script that generates karaoke LRC files with word-level timing from audio transcriptions. The main components are:

### Key Files
- `src/lrc_generator.py`: Main generator using sliding window approach
- `src/word_matcher.py`: Word matching utilities
- `src/utils.py`: Helper functions
- `output/Billy Strings - Gild the Lily/transcription.json`: Example transcription data
- `output/Billy Strings - Gild the Lily/lyrics.lrc`: Generated LRC file

### Example Data

From transcription.json:
```json
{
  "metadata": {
    "artist": "Billy Strings",
    "track": "Gild the Lily",
    "original_lyrics": "Sitting under the cypress tree I saw a miracle flying high\nI tuned into the song that she was singing\n..."
  },
  "words": [
    {
      "word": "it",
      "start": 41.76,
      "end": 42.98,
      "original_word": "i",
      "confidence": 0.5
    },
    {
      "word": "tuned",
      "start": 42.98,
      "end": 43.24,
      "confidence": 0.0
    },
    {
      "word": "it",
      "start": 43.24,
      "end": 43.68,
      "original_word": "into",
      "confidence": 0.5
    }
    // ... more words
  ]
}
```

Current LRC output:
```
[00:36.10]<00:36.10>Sitting <00:36.86>under <00:37.38>the <00:37.76>cypress <00:38.32>tree <00:38.72>I <00:38.90>saw <00:39.26>a <00:39.50>miracle <00:39.96>flying <00:40.64>high
[00:41.76]I tuned into the song that she was singing
```

## Current Issue

Our sliding window approach is failing to match some lines properly. For example, in the second line above, we have the correct words in our transcription.json but they're not getting matched and timed in the output LRC.

### Current Approach
1. Split original lyrics into lines
2. For each line:
   - Try different window sizes (line_length ± 2)
   - Score each window position using Levenshtein distance
   - Accept matches with score > 0.6
   - Fall back to raw line with basic timing if no match

### Debugging Task

Need to investigate why our sliding window matching is failing to find matches that clearly exist in the transcription data. Specific areas to look at:

1. Word matching logic in sliding window
2. How we're using confidence scores
3. Window size and position calculations
4. Scoring thresholds

## Test Case

Use the Billy Strings example:
```bash
source venv/bin/activate
python3 -m src.transcribe output/Billy\ Strings\ -\ Gild\ the\ Lily/transcription.json --artist "Billy Strings" --track "Gild the Lily" --skip-transcription
``` 