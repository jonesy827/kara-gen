"""Utility functions for text processing and formatting."""

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