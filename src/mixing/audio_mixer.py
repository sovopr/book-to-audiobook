"""Audio mixer: stitches voice clips + background music into a final MP3 using pydub."""
import os
from typing import Dict, List, Optional

from pydub import AudioSegment
from pydub.effects import normalize

from src.sound.emotion_mapper import get_music_path

# Timing constants (milliseconds)
LINE_PAUSE_MS = 400       # silence between lines
SCENE_PAUSE_MS = 1200     # silence between scene/emotion changes
CROSSFADE_MS = 500        # crossfade between scene blocks
MUSIC_VOLUME_DB = -18     # background music ducked below voice


def _load_or_silence(path: Optional[str], duration_ms: int = 0) -> AudioSegment:
    """Load an audio file, or return silence if path is None/missing."""
    if path and os.path.exists(path):
        return AudioSegment.from_file(path)
    return AudioSegment.silent(duration=duration_ms)


def _duck_music(music: AudioSegment, voice_duration_ms: int) -> AudioSegment:
    """Loop or trim music to match voice duration and duck it."""
    if len(music) == 0:
        return AudioSegment.silent(duration=voice_duration_ms)
    # Loop music ONLY if it's too short
    if len(music) >= voice_duration_ms:
        music = music[:voice_duration_ms]
    else:
        loops = (voice_duration_ms // len(music)) + 2
        music = (music * loops)[:voice_duration_ms]
    
    return music + MUSIC_VOLUME_DB


def mix_audiobook(
    lines: List[Dict],
    audio_paths: List[Optional[str]],
    output_path: str,
    progress_callback=None,
    enable_music: bool = True,
) -> str:
    """
    Stitch all voice clips with emotion-matched background music into one MP3.

    Args:
        lines: Structured line dicts (type, character, text, emotion).
        audio_paths: Parallel list of per-line MP3 paths (None = skip).
        output_path: Final MP3 output path.
        progress_callback: Optional callable(current, total).
        enable_music: If False, uses a lightning fast ffmpeg direct disk concat.

    Returns:
        Absolute path to the final mixed MP3.
    """
    assert len(lines) == len(audio_paths), "Lines and audio_paths must be the same length"

    total = len(lines)
    
    # ─── FAST PATH: Direct Disk Concat ─────────────────────────────────────────
    if not enable_music:
        import subprocess
        # Ffmpeg concat demuxer requires a list file
        list_file = output_path + ".list.txt"
        valid_paths = [p for p in audio_paths if p and os.path.exists(p)]
        
        with open(list_file, "w", encoding="utf-8") as f:
            for p in valid_paths:
                # Ffmpeg requires absolute paths or relative paths in singles quotes
                abs_p = os.path.abspath(p)
                f.write(f"file '{abs_p}'\n")
                
        if progress_callback:
            progress_callback(total // 2, total)  # Halfway visual update
            
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
            "-i", list_file, "-c:a", "libmp3lame", "-b:a", "192k", output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        os.remove(list_file)
        
        if progress_callback:
            progress_callback(total, total)
            
        return output_path

    # ─── SLOW PATH: PyDub Mixing with Music ────────────────────────────────────
    final = AudioSegment.empty()
    prev_emotion = None

    # Pre-load background music segments per emotion to avoid repeated disk I/O
    music_cache: Dict[str, AudioSegment] = {}

    for i, (line, path) in enumerate(zip(lines, audio_paths)):
        emotion = line.get("emotion", "neutral").lower()
        voice_clip = _load_or_silence(path, LINE_PAUSE_MS)

        # Get (or cache) background music for this emotion
        if emotion not in music_cache:
            music_path = get_music_path(emotion)
            if music_path:
                music_cache[emotion] = AudioSegment.from_file(music_path)
            else:
                music_cache[emotion] = AudioSegment.empty()

        # Duck music under voice
        bg = _duck_music(music_cache[emotion], len(voice_clip))

        # Overlay voice on music
        if len(bg) > 0:
            mixed = bg.overlay(voice_clip)
        else:
            mixed = voice_clip

        # Add pause between lines
        pause_ms = SCENE_PAUSE_MS if (prev_emotion and emotion != prev_emotion) else LINE_PAUSE_MS
        silence = AudioSegment.silent(duration=pause_ms)

        if prev_emotion and emotion != prev_emotion and len(final) > 0:
            # Crossfade at scene transitions
            final = final.append(mixed, crossfade=min(CROSSFADE_MS, len(final), len(mixed)))
        else:
            final = final + mixed

        final = final + silence
        prev_emotion = emotion

        if progress_callback:
            progress_callback(i + 1, total)

    # Normalize and export
    final = normalize(final)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final.export(output_path, format="mp3", bitrate="128k")
    return output_path
