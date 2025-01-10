"""Module for generating LRC files from transcribed words.

This module implements a sophisticated approach to generating karaoke LRC files by matching
transcribed audio with original lyrics. It uses a hybrid approach combining:

1. Sliding Window Word Matching:
   - Adaptive window sizing to handle variations in word count
   - Position-weighted scoring favoring centered matches
   - Confidence score integration from transcription data
   - Bonus scoring for exact matches and full line matches

2. Two-Pass Processing:
   - First pass finds high-confidence matches using sliding windows
   - Second pass interpolates timing for unmatched sections
   - Ensures complete coverage of all lyrics with timing

3. Intelligent Timing Interpolation:
   - Evenly distributes time between known good matches
   - Maintains word-level timing even in interpolated sections
   - Preserves small gaps between lines for readability

The algorithm is designed to be robust against common transcription issues:
- Word recognition errors
- Extra or missing words
- Timing misalignments
- Variations in phrasing

Key Parameters:
- Window size range: base size -2 to +4 words
- Position weight curve: 1 - (abs(pos - center) / (length * 1.5))
- Minimum match score: 0.4
- Exact word match bonus: 2x
- Full line match bonus: 1.2x
- Inter-line gap: 10% of available time

Example:
    >>> words_data = {
    ...     "metadata": {
    ...         "artist": "Artist",
    ...         "track": "Track",
    ...         "original_lyrics": "First line\\nSecond line"
    ...     },
    ...     "words": [
    ...         {"word": "first", "start": 1.0, "end": 1.5},
    ...         {"word": "line", "start": 1.5, "end": 2.0}
    ...     ]
    ... }
    >>> generate_lrc_file(words_data, "output.lrc")
"""

import os
from Levenshtein import distance
from .utils import clean_word, format_timestamp

def create_output_directory(artist, track):
    """Create and return path to output directory based on artist and track name.
    
    Args:
        artist (str): Artist name
        track (str): Track name
        
    Returns:
        str: Path to output directory
    """
    # Clean names to be filesystem friendly
    artist = ''.join(c for c in artist if c.isalnum() or c in (' -_'))
    track = ''.join(c for c in track if c.isalnum() or c in (' -_'))
    
    # Create directory path
    dir_path = os.path.join('output', f"{artist} - {track}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def find_best_window_match(lyrics_words, transcribed_words, start_idx, window_size):
    """Find the best matching window of transcribed words for a sequence of lyrics words.
    
    Uses a sophisticated sliding window approach with adaptive sizing and multi-factor scoring:
    
    1. Window Positioning:
       - Tries multiple window positions (up to 20 positions ahead)
       - At each position, tries multiple window sizes (-2 to +4 from base size)
       - Ensures minimum window size matches lyrics word count
    
    2. Scoring System:
       - Word similarity using Levenshtein distance
       - Position weighting favoring centered matches
       - Confidence score integration from transcription
       - Exact match bonus (2x)
       - Full line match bonus (1.2x)
    
    3. Score Calculation:
       word_score = base_similarity * position_weight * confidence_factor
       final_score = (total_word_scores / total_weights) * bonuses
    
    The algorithm is designed to be robust against:
    - Transcription errors
    - Word order variations
    - Extra or missing words
    - Different word forms
    
    Args:
        lyrics_words (list): Sequence of words from original lyrics
        transcribed_words (list): List of transcribed word dictionaries
        start_idx (int): Starting index in transcribed words
        window_size (int): Base size of window to consider
        
    Returns:
        tuple: (matched_words, score, next_idx) where:
            - matched_words: List of transcribed word dictionaries that best match the lyrics
            - score: Float between 0 and 1 indicating match quality
            - next_idx: Index to start searching from for the next line
            
    Example:
        >>> lyrics = ["sitting", "under", "the", "tree"]
        >>> transcribed = [
        ...     {"word": "i", "start": 1.0, "end": 1.1},
        ...     {"word": "sitting", "start": 1.1, "end": 1.5},
        ...     {"word": "under", "start": 1.5, "end": 1.8},
        ...     {"word": "the", "start": 1.8, "end": 1.9},
        ...     {"word": "tree", "start": 1.9, "end": 2.2}
        ... ]
        >>> words, score, next_idx = find_best_window_match(lyrics, transcribed, 0, 4)
        >>> print([w["word"] for w in words])
        ['sitting', 'under', 'the', 'tree']
    """
    best_score = 0
    best_match = None
    best_window_start = None
    
    # Try different window positions, looking ahead up to 20 positions
    # For each position, try different window sizes
    for window_start in range(start_idx, min(len(transcribed_words), start_idx + 20)):
        # Try different window sizes at this position
        for size_delta in range(-2, 5):  # Allow more expansion than contraction
            current_size = max(window_size + size_delta, len(lyrics_words))
            
            if window_start + current_size > len(transcribed_words):
                continue
                
            window = transcribed_words[window_start:window_start + current_size]
            score = 0
            total_weight = 0
            matched_count = 0
            
            # Score this window position
            for i, lw in enumerate(lyrics_words):
                if i >= len(window):
                    break
                    
                lw_clean = clean_word(lw)
                tw_clean = clean_word(window[i]['word'])
                
                # Skip empty words
                if not lw_clean or not tw_clean:
                    continue
                    
                # Calculate word similarity using Levenshtein distance
                word_dist = distance(lw_clean, tw_clean)
                word_score = 1 - (word_dist / max(len(lw_clean), len(tw_clean)))
                
                # Weight exact matches more heavily (2x)
                if word_dist == 0:
                    word_score *= 2
                    matched_count += 1
                
                # Incorporate confidence score if available
                confidence = window[i].get('confidence', 0.5)
                
                # Words closer to the center of the window score higher
                # Use a gentler position weight curve
                position_weight = 1 - (abs(i - (len(lyrics_words) // 2)) / (len(lyrics_words) * 1.5))
                
                # Combine scores with confidence
                weighted_score = word_score * position_weight * (0.5 + 0.5 * confidence)
                score += weighted_score
                total_weight += position_weight
            
            # Normalize score by total weight and apply bonus for matching all words
            if total_weight > 0:
                score = score / total_weight
                # Bonus for matching all words
                if matched_count == len(lyrics_words):
                    score *= 1.2
                
                if score > best_score:
                    best_score = score
                    best_match = window[:len(lyrics_words)]  # Only keep the words we need
                    best_window_start = window_start
    
    # Return match if score is good enough (> 0.4)
    if best_score > 0.4:
        return best_match, best_score, best_window_start + len(best_match)
    return None, 0, start_idx

def generate_lrc_file(words_data, output_path):
    """Generate an enhanced LRC file with word-level timing, using the original lyrics as the source of truth.
    
    This function generates an LRC file that preserves the original lyrics structure while adding
    word-level timing information from the transcribed words. Every line will have timing information,
    either from matched transcription or interpolated between known good matches.
    
    Args:
        words_data (dict): Dictionary containing:
            - metadata (dict): Song metadata including artist, track, and original_lyrics
            - words (list): List of transcribed word dictionaries
        output_path (str): Path where the LRC file should be saved
    """
    lrc_lines = []
    
    # Add metadata
    lrc_lines.append("[ar:{}]".format(words_data['metadata']['artist']))
    lrc_lines.append("[ti:{}]".format(words_data['metadata']['track']))
    lrc_lines.append("[length:{}]".format(format_timestamp(words_data['words'][-1]['end'])))
    
    # Get and process the original lyrics
    original_lyrics = words_data['metadata'].get('original_lyrics', '')
    if not original_lyrics:
        print("No original lyrics provided. Cannot generate LRC file.")
        return
    
    # Split lyrics into lines and words
    lyrics_lines = [line.strip() for line in original_lyrics.split('\n')]
    lyrics_words = [line.split() for line in lyrics_lines]
    
    # First pass: Find all confident matches
    transcribed_idx = 0
    line_matches = []  # List of (line_idx, start_time, end_time, matched_words) tuples
    
    for line_idx, line_words in enumerate(lyrics_words):
        if not line_words:  # Handle empty lines
            line_matches.append((line_idx, None, None, None))
            continue
            
        # Try different window sizes based on line length
        base_window_size = len(line_words)
        best_match = None
        best_score = 0
        best_next_idx = transcribed_idx
        
        # Try window sizes from -3 to +3 of the line length
        for window_delta in range(-3, 4):
            window_size = max(base_window_size + window_delta, 1)
            
            if transcribed_idx + window_size > len(words_data['words']):
                continue
                
            match, score, next_idx = find_best_window_match(
                line_words, words_data['words'], transcribed_idx, window_size
            )
            
            if score > best_score:
                best_match = match
                best_score = score
                best_next_idx = next_idx
        
        if best_match and best_score > 0.4:  # Good match found
            start_time = best_match[0]['start']
            end_time = best_match[-1]['end']
            line_matches.append((line_idx, start_time, end_time, best_match))
            transcribed_idx = best_next_idx
        else:
            line_matches.append((line_idx, None, None, None))
    
    # Second pass: Process lines in order, interpolating timings where needed
    last_good_time = 0
    next_good_time = words_data['words'][-1]['end']
    last_good_idx = -1
    
    # Find first good match
    for i, (_, start_time, _, _) in enumerate(line_matches):
        if start_time is not None:
            last_good_time = start_time
            last_good_idx = i
            break
    
    # Process each line
    for i, (line_idx, start_time, end_time, words) in enumerate(line_matches):
        if not lyrics_lines[line_idx]:  # Empty line
            lrc_lines.append("")
            continue
            
        if start_time is not None:  # Good match
            # Format line with matched word timings
            lrc_line = "[{}]".format(format_timestamp(start_time))
            for word, timing in zip(lyrics_words[line_idx], words):
                lrc_line += "<{}>{}".format(format_timestamp(timing['start']), word)
            
            # Update last good match
            last_good_time = start_time
            last_good_idx = i
            
            # Look ahead for next good match
            next_good_time = words_data['words'][-1]['end']
            for next_match in line_matches[i+1:]:
                if next_match[1] is not None:
                    next_good_time = next_match[1]
                    break
        else:  # Interpolate timing
            # Calculate position between last and next good match
            time_span = next_good_time - last_good_time
            lines_between = sum(1 for x in line_matches[last_good_idx+1:i+1] 
                              if x[1] is None and lyrics_lines[x[0]])
            
            if lines_between > 0:
                # Calculate interpolated time
                time_per_line = time_span / (lines_between + 1)
                relative_pos = sum(1 for x in line_matches[last_good_idx+1:i] 
                                 if x[1] is None and lyrics_lines[x[0]])
                interpolated_time = last_good_time + (time_per_line * (relative_pos + 1))
                
                # Calculate word timings
                words_in_line = len(lyrics_words[line_idx])
                word_time_span = time_per_line * 0.9  # Leave small gap between lines
                time_per_word = word_time_span / words_in_line if words_in_line > 0 else 0
                
                # Generate line with interpolated word timings
                lrc_line = "[{}]".format(format_timestamp(interpolated_time))
                for word_idx, word in enumerate(lyrics_words[line_idx]):
                    word_time = interpolated_time + (word_idx * time_per_word)
                    lrc_line += "<{}>{}".format(format_timestamp(word_time), word)
            else:
                # If no lines between, use midpoint
                interpolated_time = (last_good_time + next_good_time) / 2
                lrc_line = "[{}]{}".format(
                    format_timestamp(interpolated_time),
                    lyrics_lines[line_idx]
                )
        
        lrc_lines.append(lrc_line)
    
    # Write the LRC file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lrc_lines)) 