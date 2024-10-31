"""Utility modules for Doc2Podcast"""

from .audio_utils import (
    combine_audio_segments,
    adjust_speed,
    normalize_audio,
    add_silence,
    remove_silence,
    save_audio,
    load_audio
)
from .text_utils import (
    extract_sections,
    format_dialogue,
    add_speech_patterns,
    format_for_tts,
    clean_transcript,
    extract_key_points
)
from .cache_manager import cache_manager

__all__ = [
    'combine_audio_segments',
    'adjust_speed',
    'normalize_audio',
    'add_silence',
    'remove_silence',
    'save_audio',
    'load_audio',
    'extract_sections',
    'format_dialogue',
    'add_speech_patterns',
    'format_for_tts',
    'clean_transcript',
    'extract_key_points',
    'cache_manager'
]
