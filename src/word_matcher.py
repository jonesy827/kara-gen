"""Module for matching and aligning transcribed words with lyrics."""

from Levenshtein import distance
from collections import defaultdict
from .utils import clean_word, clean_line

def line_similarity(line1, line2):
    """Calculate similarity between two lines using word overlap and Levenshtein.
    
    Args:
        line1 (str): First line to compare
        line2 (str): Second line to compare
        
    Returns:
        float: Similarity score between 0 and 1
    """
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

def detect_repeated_words(lyrics_lines):
    """Detect repeated words in lyrics and create a map of repetitions.
    
    Args:
        lyrics_lines (list): List of lyrics lines
        
    Returns:
        dict: Dictionary mapping word indices to frequency counts
    """
    # Clean lines for comparison
    clean_lines = [clean_line(line) for line in lyrics_lines]
    
    # Find repeated words
    word_occurrences = defaultdict(int)
    for line in clean_lines:
        for word in line.split():
            if word.strip():  # Skip empty words
                word_occurrences[word] += 1
    
    return word_occurrences

def match_words_to_lyrics(transcribed_words, lyrics):
    """Match transcribed words to lyrics using Levenshtein distance.
    
    Enhanced version with improved handling of repeated words and lines:
    - Increased search range for better matching of repeated content
    - Adaptive search based on word frequency in lyrics
    - Improved confidence scoring for repeated words
    
    Args:
        transcribed_words (list): List of word dictionaries with timing info
        lyrics (str): Original lyrics text
        
    Returns:
        list: Updated word dictionaries with matched words
    """
    if not lyrics or not transcribed_words:
        print("No lyrics or transcribed words to match")
        return transcribed_words
    
    print(f"Starting word matching with {len(transcribed_words)} transcribed words")
    
    # Split lyrics into lines and words, preserving line structure
    lyrics_lines = lyrics.split('\n')
    lyrics_words = []
    word_to_line = {}  # Maps word index to line number
    
    for line_num, line in enumerate(lyrics_lines):
        line_words = line.split()
        for word in line_words:
            clean = clean_word(word)
            if clean:
                lyrics_words.append((word, clean))  # Keep both original and clean forms
                word_to_line[len(lyrics_words) - 1] = line_num
    
    if not lyrics_words:
        print("No lyrics words to match against")
        return transcribed_words
        
    print(f"Found {len(lyrics_words)} words in lyrics")
    
    # Detect repeated words to adjust search strategy
    word_occurrences = detect_repeated_words(lyrics_lines)
    repeated_words = {word for word, count in word_occurrences.items() if count > 1}
    print(f"Detected {len(repeated_words)} repeated words in lyrics")
        
    # Process each transcribed word
    matched_words = []
    lyrics_idx = 0
    last_line_num = -1
    
    for word_info in transcribed_words:
        transcribed_word = clean_word(word_info['word'])
        if not transcribed_word:
            print(f"Skipping empty transcribed word: {word_info['word']}")
            continue
            
        # Look for the best match in the lyrics words
        best_match = None
        best_distance = float('inf')
        best_idx = None
        
        # Make sure we don't go past the end of lyrics_words
        if lyrics_idx >= len(lyrics_words):
            # We've run out of lyrics to match against
            print(f"Reached end of lyrics at word: {transcribed_word}")
            word_info['confidence'] = 0.0
            word_info['line_number'] = last_line_num
            matched_words.append(word_info)
            continue
        
        # Determine search range - use larger range for repeated words
        base_search_range = 30  # Increased from 10
        
        # Check if this word is likely to be repeated
        is_common_word = transcribed_word in repeated_words
        
        # Adjust search range based on word frequency
        if is_common_word:
            search_range = min(50, len(lyrics_words) - lyrics_idx)  # Much larger range for repeated words
            print(f"Using extended search range ({search_range}) for repeated word: '{transcribed_word}'")
        else:
            search_range = min(base_search_range, len(lyrics_words) - lyrics_idx)
        
        # First, try to find an exact match
        exact_match_found = False
        for i in range(search_range):
            current_idx = lyrics_idx + i
            if current_idx >= len(lyrics_words):
                break
                
            _, current_clean = lyrics_words[current_idx]
            
            # Prioritize exact matches
            if current_clean == transcribed_word:
                best_distance = 0
                best_match = lyrics_words[current_idx][0]  # Use original form
                best_idx = i
                exact_match_found = True
                break
        
        # If no exact match, look for close matches
        if not exact_match_found:
            for i in range(search_range):
                current_idx = lyrics_idx + i
                if current_idx >= len(lyrics_words):
                    break
                    
                _, current_clean = lyrics_words[current_idx]
                current_distance = distance(transcribed_word, current_clean)
                
                # If we find a very close match
                if current_distance < best_distance and current_distance <= 2:
                    best_distance = current_distance
                    best_match = lyrics_words[current_idx][0]  # Use original form
                    best_idx = i
        
        # If we found a good match, update the word and advance the lyrics index
        if best_match:
            print(f"Matched '{transcribed_word}' to '{best_match}' (distance: {best_distance})")
            word_info['word'] = best_match
            word_info['original_word'] = transcribed_word
            word_info['confidence'] = 1 - (best_distance / max(len(best_match), len(transcribed_word)))
            word_info['line_number'] = word_to_line[lyrics_idx + best_idx]
            
            # Add line break marker if we've moved to a new line
            current_line_num = word_to_line[lyrics_idx + best_idx]
            if last_line_num != -1 and current_line_num > last_line_num:
                word_info['line_break'] = True
            last_line_num = current_line_num
            
            lyrics_idx += best_idx + 1
        else:
            print(f"No match found for '{transcribed_word}'")
            word_info['confidence'] = 0.0
            word_info['line_number'] = last_line_num

        matched_words.append(word_info)
    
    print(f"Matched {len(matched_words)} words total")
    return matched_words
