"""Karaoke lyrics generation package."""

from .transcribe import main
from .lyrics_fetcher import get_lyrics
from .word_matcher import match_words_to_lyrics, line_similarity
from .lrc_generator import generate_lrc_file, create_output_directory
from .utils import clean_word, clean_line, format_timestamp

__all__ = [
    'main',
    'get_lyrics',
    'match_words_to_lyrics',
    'line_similarity',
    'generate_lrc_file',
    'create_output_directory',
    'clean_word',
    'clean_line',
    'format_timestamp',
] 