# kara-gen: Karaoke LRC Generator

A sophisticated tool for generating word-level timed LRC (lyrics) files from audio transcriptions, using a hybrid approach of precise word matching and intelligent timing interpolation.

## Overview

This tool generates high-quality karaoke LRC files by matching transcribed audio with original lyrics, ensuring every line and word has accurate timing information. It uses a combination of:

- Sliding window word matching with adaptive sizing
- Levenshtein distance-based word similarity scoring
- Confidence-weighted scoring from transcription data
- Intelligent timing interpolation for unmatched sections

## Key Features

- Word-level timing accuracy for matched sections
- No missing lyrics - every line is timed
- Preserves original lyrics structure and formatting
- Handles parenthetical sections and line variations
- Smart fallback for sections with low confidence matches

## Technical Approach

### 1. Word Matching Algorithm

The core matching uses a sophisticated sliding window approach:

```python
# Example of how matching works:
Original:    "Sitting under the cypress tree"
Transcribed: "i sitting under the cypress the"
Result:      [0.8 match] -> preserves original words with transcribed timing
```

Key components:
- Adaptive window sizing (base size ±4 words)
- Position-weighted scoring favoring centered matches
- Confidence score integration from transcription
- Bonus scoring for exact word matches

### 2. Two-Pass Processing

#### First Pass: Find Confident Matches
- Processes each line looking for high-confidence matches
- Uses Levenshtein distance for word similarity
- Incorporates transcription confidence scores
- Requires 0.4 minimum match score

#### Second Pass: Timing Interpolation
- Fills gaps between confident matches
- Evenly distributes timing for unmatched lines
- Maintains word-level timing even in interpolated sections
- Preserves small gaps between lines for readability

### 3. Scoring System

Word matches are scored using multiple factors:
- Word similarity (Levenshtein distance)
- Position weight in window
- Transcription confidence
- Exact match bonus (2x)
- Full line match bonus (1.2x)

## Usage

1. Prepare your audio transcription:
```json
{
  "metadata": {
    "artist": "Artist Name",
    "track": "Track Name",
    "original_lyrics": "Full lyrics here..."
  },
  "words": [
    {
      "word": "transcribed",
      "start": 1.23,
      "end": 1.45,
      "confidence": 0.8
    },
    ...
  ]
}
```

2. Run the generator:
```bash
python3 -m src.transcribe path/to/transcription.json --artist "Artist Name" --track "Track Name"
```

3. Get your LRC file:
```
[ar:Artist Name]
[ti:Track Name]
[00:01.23]<00:01.23>First<00:01.45>word<00:01.78>with<00:02.10>timing
```

## Implementation Details

### Window Matching Logic

```python
# Scoring example:
word_score = base_similarity * position_weight * confidence_factor
final_score = (total_word_scores / total_weights) * bonuses
```

Key parameters:
- Window size range: -2 to +4 from base size
- Position weight curve: 1 - (abs(pos - center) / (length * 1.5))
- Minimum match score: 0.4
- Exact word match bonus: 2x
- Full line match bonus: 1.2x

### Timing Interpolation

For sections between confident matches:
1. Calculate time span between known good timings
2. Count lines needing interpolation
3. Distribute time evenly, reserving 10% for gaps
4. Generate word-level timing within each line

## Results

The generator produces LRC files with:
- Accurate word-level timing for matched sections
- Natural timing flow through interpolated sections
- Complete coverage of all lyrics
- Preserved line structure and formatting
- Clean, karaoke-ready output

## Future Improvements

Potential areas for enhancement:
- Machine learning for improved word matching
- Rhythm analysis for better timing interpolation
- Support for multiple languages
- Handling of repeated sections
- Integration with more audio transcription services 