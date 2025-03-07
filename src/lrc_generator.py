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

4. Enhanced Repeated Line Handling:
   - Pattern detection for repeated lyrics sections
   - Context-aware matching for disambiguating repeated lines
   - Extended search range for finding the best match
   - Global optimization to ensure consistent matching

The algorithm is designed to be robust against common transcription issues:
- Word recognition errors
- Extra or missing words
- Timing misalignments
- Variations in phrasing
- Repeated lines and sections

Key Parameters:
- Window size range: base size -2 to +4 words
- Position weight curve: 1 - (abs(pos - center) / (length * 1.5))
- Minimum match score: 0.4
- Exact word match bonus: 2x
- Full line match bonus: 1.2x
- Inter-line gap: 10% of available time
- Search range: 100 positions (extended from 20)
- Context bonus: Up to 1.5x for consistent sequential matches

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
from collections import defaultdict
from .utils import clean_word, clean_line, format_timestamp

def create_output_directory(artist, track):
    """Create and return path to output directory.
    
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

def detect_repeated_lines(lyrics_lines):
    """Detect repeated lines in lyrics and create a map of repetitions.
    
    Args:
        lyrics_lines (list): List of lyrics lines
        
    Returns:
        dict: Dictionary mapping line indices to lists of indices of identical lines
    """
    # Clean lines for comparison
    clean_lines = [clean_line(line) for line in lyrics_lines]
    
    # Find repeated lines
    line_occurrences = defaultdict(list)
    for i, line in enumerate(clean_lines):
        if line.strip():  # Skip empty lines
            line_occurrences[line].append(i)
    
    # Create a map of repetitions (only include lines that repeat)
    repetition_map = {}
    for line, occurrences in line_occurrences.items():
        if len(occurrences) > 1:
            for idx in occurrences:
                repetition_map[idx] = occurrences
    
    return repetition_map

def find_best_window_match(lyrics_words, transcribed_words, start_idx, window_size, 
                          context=None, repetition_info=None, previous_matches=None):
    """Find the best matching window of transcribed words for a sequence of lyrics words.
    
    Uses a sophisticated sliding window approach with adaptive sizing and multi-factor scoring:
    
    1. Window Positioning:
       - Tries multiple window positions (up to 100 positions ahead)
       - At each position, tries multiple window sizes (-2 to +4 from base size)
       - Ensures minimum window size matches lyrics word count
    
    2. Scoring System:
       - Word similarity using Levenshtein distance
       - Position weighting favoring centered matches
       - Confidence score integration from transcription
       - Exact match bonus (2x)
       - Full line match bonus (1.2x)
       - Context bonus for consistent sequential matches (up to 1.5x)
       - Repetition handling for disambiguating repeated lines
    
    3. Score Calculation:
       word_score = base_similarity * position_weight * confidence_factor
       final_score = (total_word_scores / total_weights) * bonuses * context_bonus
    
    The algorithm is designed to be robust against:
    - Transcription errors
    - Word order variations
    - Extra or missing words
    - Different word forms
    - Repeated lines and sections
    
    Args:
        lyrics_words (list): Sequence of words from original lyrics
        transcribed_words (list): List of transcribed word dictionaries
        start_idx (int): Starting index in transcribed words
        window_size (int): Base size of window to consider
        context (tuple, optional): Tuple of (line_idx, prev_match_end_time) for context scoring
        repetition_info (dict, optional): Information about repeated lines
        previous_matches (list, optional): List of previous matches for context scoring
        
    Returns:
        tuple: (matched_words, score, next_idx) where:
            - matched_words: List of transcribed word dictionaries that best match the lyrics
            - score: Float between 0 and 1 indicating match quality
            - next_idx: Index to start searching from for the next line
    """
    best_score = 0
    best_match = None
    best_window_start = None
    
    # Determine search range - use larger range for repeated lines
    search_range = 100  # Default extended search range
    if repetition_info and repetition_info.get('is_repeated', False):
        search_range = 150  # Even larger range for repeated lines
    
    # Try different window positions, looking ahead up to search_range positions
    for window_start in range(start_idx, min(len(transcribed_words), start_idx + search_range)):
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
            
            # Normalize score by total weight and apply bonuses
            if total_weight > 0:
                score = score / total_weight
                
                # Bonus for matching all words
                if matched_count == len(lyrics_words):
                    score *= 1.2
                
                # Apply context bonus if we have context information
                if context and previous_matches:
                    line_idx, prev_match_end_time = context
                    
                    # Check if this match follows logically from previous matches
                    if window[0]['start'] > prev_match_end_time:
                        # Bonus for matches that follow previous matches in time
                        time_gap = window[0]['start'] - prev_match_end_time
                        if time_gap < 5.0:  # Reasonable gap between lines
                            # Higher bonus for smaller gaps
                            context_bonus = 1.0 + max(0, (5.0 - time_gap) / 10.0)
                            score *= min(1.5, context_bonus)  # Cap at 1.5x
                
                # Apply repetition handling if this is a repeated line
                if repetition_info and repetition_info.get('is_repeated', False):
                    occurrence_idx = repetition_info.get('occurrence_idx', 0)
                    
                    # For repeated lines, prefer matches that occur later in the audio
                    # for later occurrences in the lyrics
                    if occurrence_idx > 0:
                        # Check if this match occurs after previous occurrences
                        # by comparing with expected time position
                        expected_pos = repetition_info.get('expected_position', 0)
                        if window[0]['start'] >= expected_pos:
                            # Bonus for matches that occur where we expect them
                            score *= 1.1
                
                if score > best_score:
                    best_score = score
                    best_match = window[:len(lyrics_words)]  # Only keep the words we need
                    best_window_start = window_start
    
    # Adjust threshold for repeated lines - be more lenient with later occurrences
    threshold = 0.4  # Default threshold
    if repetition_info and repetition_info.get('is_repeated', False):
        occurrence_idx = repetition_info.get('occurrence_idx', 0)
        if occurrence_idx > 0:
            threshold = max(0.35, threshold - (0.01 * occurrence_idx))  # Gradually reduce threshold
    
    # Return match if score is good enough
    if best_score > threshold:
        return best_match, best_score, best_window_start + len(best_match)
    return None, 0, start_idx

def optimize_matches(line_matches, lyrics_lines, words_data):
    """Perform global optimization on matches to ensure consistency.
    
    This function looks for inconsistencies in the matched lines and attempts to fix them:
    1. Identifies lines that should have matches but don't
    2. Detects and resolves overlapping matches
    3. Ensures repeated sections have consistent timing
    
    Args:
        line_matches (list): List of (line_idx, start_time, end_time, matched_words) tuples
        lyrics_lines (list): List of lyrics lines
        words_data (dict): Dictionary containing transcribed words
        
    Returns:
        list: Optimized list of line matches
    """
    # Create a copy of line_matches to avoid modifying the original
    optimized_matches = line_matches.copy()
    
    # Check for overlapping matches
    for i in range(len(optimized_matches) - 1):
        current = optimized_matches[i]
        next_match = optimized_matches[i + 1]
        
        # Skip if either match is not a real line or doesn't have timing
        if (current[0] < 0 or next_match[0] < 0 or 
            current[1] is None or next_match[1] is None):
            continue
        
        # Check for overlap
        if current[2] > next_match[1]:
            # Overlapping matches - adjust end time of current match
            print(f"Fixing overlap between lines {current[0]} and {next_match[0]}")
            
            # Set end time to halfway between the two start times
            new_end_time = (current[1] + next_match[1]) / 2
            optimized_matches[i] = (current[0], current[1], new_end_time, current[3])
    
    # Check for repeated lines with inconsistent timing
    repetition_map = detect_repeated_lines(lyrics_lines)
    for line_idx, occurrences in repetition_map.items():
        if len(occurrences) <= 1:
            continue
            
        # Get all matches for this repeated line
        matches = [(i, match) for i, match in enumerate(optimized_matches) 
                  if match[0] in occurrences and match[1] is not None]
        
        if len(matches) <= 1:
            continue
            
        # Calculate average duration for this line
        durations = [(match[2] - match[1]) for _, match in matches]
        avg_duration = sum(durations) / len(durations)
        
        # Adjust matches to have consistent duration
        for match_idx, match in matches:
            current_duration = match[2] - match[1]
            
            # If duration differs significantly from average, adjust it
            if abs(current_duration - avg_duration) > 0.5:
                print(f"Adjusting duration for repeated line {match[0]} " +
                      f"from {current_duration:.2f}s to {avg_duration:.2f}s")
                
                new_end_time = match[1] + avg_duration
                optimized_matches[match_idx] = (match[0], match[1], new_end_time, match[3])
    
    return optimized_matches

def generate_lrc_file(words_data, output_path=None, break_threshold=5.0):
    """Generate an enhanced LRC file with word-level timing, using the original lyrics as the source of truth.
    
    This function generates an LRC file that preserves the original lyrics structure while adding
    word-level timing information from the transcribed words. Every line will have timing information,
    either from matched transcription or interpolated between known good matches.
    
    The enhanced algorithm includes:
    1. Pattern detection for repeated lyrics sections
    2. Context-aware matching for disambiguating repeated lines
    3. Extended search range for finding the best match
    4. Global optimization to ensure consistent matching
    
    Args:
        words_data (dict): Dictionary containing:
            - metadata (dict): Song metadata including artist, track, and original_lyrics
            - words (list): List of transcribed word dictionaries
        output_path (str, optional): Path where the LRC file should be saved. If not provided,
            will use artist and track name to generate path.
        break_threshold (float): Time in seconds to consider as an instrumental break (default: 5.0)
    """
    if not words_data.get('words'):
        print("No words data provided. Cannot generate LRC file.")
        return
        
    lrc_lines = []
    
    # Add metadata
    artist = words_data['metadata']['artist']
    track = words_data['metadata']['track']
    
    lrc_lines.append("[ar:{}]".format(artist))
    lrc_lines.append("[ti:{}]".format(track))
    
    # Get the end time from the last word, or use a default if no words
    if words_data['words']:
        end_time = words_data['words'][-1]['end']
    else:
        print("Warning: No words found, using default length of 5:00")
        end_time = 300.0  # Default to 5 minutes
    lrc_lines.append("[length:{}]".format(format_timestamp(end_time)))
    
    # Get and process the original lyrics
    original_lyrics = words_data['metadata'].get('original_lyrics', '')
    if not original_lyrics:
        print("No original lyrics provided. Cannot generate LRC file.")
        return
        
    # If no output path provided, create one based on artist and track
    if output_path is None:
        dir_path = create_output_directory(artist, track)
    
    # Get timing offset if available
    timing_info = words_data['metadata'].get('timing_info', {})
    start_offset = timing_info.get('start_offset', 0.0)
    
    # Function to adjust timestamps
    def adjust_time(t):
        return t + start_offset if t is not None else None
    
    # Split lyrics into lines and words
    lyrics_lines = [line.strip() for line in original_lyrics.split('\n')]
    lyrics_words = [line.split() for line in lyrics_lines]
    
    # Detect repeated lines in lyrics
    repetition_map = detect_repeated_lines(lyrics_lines)
    print(f"Detected {len(repetition_map)} lines with repetitions")
    
    # First pass: Find all confident matches and detect breaks
    transcribed_idx = 0
    line_matches = []  # List of (line_idx, start_time, end_time, matched_words) tuples
    last_word_end = 0
    break_times = set()  # Track break timestamps to prevent duplicates
    previous_matches = []  # Track previous matches for context scoring
    
    # Add initial instrumental break if we have timing info
    timing_info = words_data['metadata'].get('timing_info', {})
    if timing_info and timing_info.get('start_offset', 0.0) > 0:
        start_offset = timing_info['start_offset']
        if start_offset >= break_threshold:
            line_matches.append((-1, 0.0, start_offset, None))  # Add initial instrumental break
            last_word_end = start_offset
    
    for line_idx, line_words in enumerate(lyrics_lines):
        # Check for instrumental break before this line
        if transcribed_idx > 0 and transcribed_idx < len(words_data['words']):
            current_word_start = words_data['words'][transcribed_idx]['start']
            gap_duration = current_word_start - last_word_end
            if gap_duration >= break_threshold:
                # Only add break if we haven't seen this timestamp before
                break_key = (last_word_end, current_word_start)
                if break_key not in break_times:
                    break_times.add(break_key)
                    line_matches.append((-1, last_word_end, current_word_start, None))  # -1 indicates break
                    
        if not line_words:  # Handle empty lines
            line_matches.append((line_idx, None, None, None))
            continue
            
        # Prepare repetition information if this is a repeated line
        repetition_info = None
        if line_idx in repetition_map:
            occurrences = repetition_map[line_idx]
            occurrence_idx = occurrences.index(line_idx)
            
            # For repeated lines, calculate expected position based on previous occurrences
            expected_position = 0
            if occurrence_idx > 0:
                # Find previous occurrences that have been matched
                prev_occurrences = [idx for idx in occurrences[:occurrence_idx] 
                                   if any(match[0] == idx and match[1] is not None 
                                         for match in line_matches)]
                
                if prev_occurrences:
                    # Use the timing of the last matched occurrence as a reference
                    last_matched = max(prev_occurrences)
                    last_match_info = next((match for match in line_matches if match[0] == last_matched), None)
                    
                    if last_match_info and last_match_info[1] is not None:
                        # Estimate position based on time since last occurrence
                        last_match_time = last_match_info[1]
                        lines_between = line_idx - last_matched
                        
                        # Rough estimate: assume each line takes about 3-5 seconds
                        expected_position = last_match_time + (lines_between * 4.0)
            
            repetition_info = {
                'is_repeated': True,
                'occurrence_idx': occurrence_idx,
                'total_occurrences': len(occurrences),
                'expected_position': expected_position
            }
            
            print(f"Line {line_idx} is repeated (occurrence {occurrence_idx+1}/{len(occurrences)})")
            
        # Try different window sizes based on line length
        base_window_size = len(line_words)
        best_match = None
        best_score = 0
        best_next_idx = transcribed_idx
        
        # Prepare context information for scoring
        context = None
        if line_matches:
            # Find the last non-empty match
            last_match = next((m for m in reversed(line_matches) if m[0] >= 0 and m[1] is not None), None)
            if last_match:
                context = (last_match[0], last_match[2])  # (line_idx, end_time)
        
        # Try window sizes from -3 to +3 of the line length
        for window_delta in range(-3, 4):
            window_size = max(base_window_size + window_delta, 1)
            
            if transcribed_idx + window_size > len(words_data['words']):
                continue
                
            match, score, next_idx = find_best_window_match(
                line_words, words_data['words'], transcribed_idx, window_size,
                context=context, repetition_info=repetition_info, previous_matches=previous_matches
            )
            
            if score > best_score:
                best_match = match
                best_score = score
                best_next_idx = next_idx
        
        if best_match and best_score > 0.4:  # Good match found
            # For repeated lines, verify this match doesn't overlap with previous matches
            is_valid = True
            if repetition_info and repetition_info.get('is_repeated', True):
                for prev_match in line_matches:
                    if prev_match[1] is not None and prev_match[2] is not None:
                        # Check for significant overlap
                        if (best_match[0]['start'] < prev_match[2] and 
                            best_match[-1]['end'] > prev_match[1]):
                            overlap_ratio = (min(best_match[-1]['end'], prev_match[2]) - 
                                            max(best_match[0]['start'], prev_match[1])) / \
                                           (best_match[-1]['end'] - best_match[0]['start'])
                            
                            if overlap_ratio > 0.5:  # More than 50% overlap
                                print(f"Rejecting match for line {line_idx} due to overlap with line {prev_match[0]}")
                                is_valid = False
                                break
            
            if is_valid:
                start_time = best_match[0]['start']
                end_time = best_match[-1]['end']
                line_matches.append((line_idx, start_time, end_time, best_match))
                transcribed_idx = best_next_idx
                last_word_end = end_time
                previous_matches.append((line_idx, start_time, end_time))
            else:
                line_matches.append((line_idx, None, None, None))
        else:
            line_matches.append((line_idx, None, None, None))
    
    # Global optimization pass: Check for inconsistencies and fix them
    line_matches = optimize_matches(line_matches, lyrics_lines, words_data)
    
    # Second pass: Process lines in order, interpolating timings where needed
    last_good_time = 0
    next_good_time = words_data['words'][-1]['end']
    last_good_idx = -1
    
    # Find first good match
    for i, (line_idx, start_time, _, _) in enumerate(line_matches):
        if line_idx >= 0 and start_time is not None:  # Skip break markers
            last_good_time = start_time
            last_good_idx = i
            break
    
    # Process each line
    last_was_break = False
    for i, (line_idx, start_time, end_time, words) in enumerate(line_matches):
        if line_idx == -1:  # Instrumental break
            if not last_was_break:  # Only add break if we haven't just added one
                duration = end_time - start_time
                if lrc_lines and lrc_lines[-1] != "":  # Add blank line before break if needed
                    lrc_lines.append("")
                lrc_lines.append(f"[{format_timestamp(adjust_time(start_time))}]♪ ═══════ INSTRUMENTAL [{format_timestamp(duration)}] ═══════ ♪")
                lrc_lines.append("")
            last_was_break = True
            continue
            
        last_was_break = False
        if not lyrics_lines[line_idx]:  # Empty line
            if lrc_lines and lrc_lines[-1] != "":  # Only add blank line if previous line wasn't blank
                lrc_lines.append("")
            continue
            
        if start_time is not None:  # Good match
            # Format line with matched word timings
            lrc_line = "[{}]".format(format_timestamp(adjust_time(start_time)))
            for word, timing in zip(lyrics_words[line_idx], words):
                lrc_line += "<{}>{}".format(format_timestamp(adjust_time(timing['start'])), word)
            
            # Update last good match
            last_good_time = start_time
            last_good_idx = i
            
            # Look ahead for next good match
            next_good_time = words_data['words'][-1]['end']
            for next_match in line_matches[i+1:]:
                if next_match[1] is not None and next_match[0] >= 0:  # Skip breaks
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
                lrc_line = "[{}]".format(format_timestamp(adjust_time(interpolated_time)))
                for word_idx, word in enumerate(lyrics_words[line_idx]):
                    word_time = interpolated_time + (word_idx * time_per_word)
                    lrc_line += "<{}>{}".format(format_timestamp(adjust_time(word_time)), word)
            else:
                # If no lines between, use midpoint
                interpolated_time = (last_good_time + next_good_time) / 2
                lrc_line = "[{}]{}".format(
                    format_timestamp(adjust_time(interpolated_time)),
                    lyrics_lines[line_idx]
                )
        
        lrc_lines.append(lrc_line)
    
    # Write the LRC file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lrc_lines))
