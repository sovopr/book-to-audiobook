"""Emotion → royalty-free background music asset mapper."""
import os
from typing import Optional

# Absolute path to the assets directory (relative to this file)
_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

EMOTION_MAP = {
    "tense":       "tense.mp3",
    "scary":       "tense.mp3",
    "angry":       "tense.mp3",
    "dramatic":    "tense.mp3",
    "romantic":    "romantic.mp3",
    "happy":       "upbeat.mp3",
    "upbeat":      "upbeat.mp3",
    "sad":         "melancholic.mp3",
    "melancholic": "melancholic.mp3",
    "scared":      "melancholic.mp3",
    "neutral":     "neutral.mp3",
}


def get_music_path(emotion: str) -> Optional[str]:
    """
    Return absolute path to the background music file for a given emotion label.
    Falls back to neutral.mp3 for unknown emotions.
    Returns None if the asset file doesn't exist.
    """
    filename = EMOTION_MAP.get(emotion.lower(), "neutral.mp3")
    path = os.path.join(_ASSETS_DIR, filename)
    return path if os.path.exists(path) else None
