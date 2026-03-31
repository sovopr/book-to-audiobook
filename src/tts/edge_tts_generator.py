"""Edge-TTS generator: free Microsoft neural voices, Streamlit-safe async handling."""
import asyncio
import os
import threading
from typing import Dict, List

import edge_tts


def list_voices() -> List[str]:
    """
    Return a sorted list of available English edge-tts voice names.
    Runs in a dedicated thread to avoid Streamlit event-loop conflicts.
    """
    result = []
    error_holder = []

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            voices = loop.run_until_complete(edge_tts.list_voices())
            # Filter to English voices for brevity; remove filter for all languages
            result.extend(
                sorted(v["ShortName"] for v in voices if v["Locale"].startswith("en-"))
            )
        except Exception as e:
            error_holder.append(e)
        finally:
            loop.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join()

    if error_holder:
        raise error_holder[0]
    return result


async def _synthesize_one(text: str, voice_name: str, output_path: str, sem: asyncio.Semaphore, emotion: str = "neutral") -> str:
    """Coroutine: synthesize one text segment and save as MP3."""
    # Instantly skip if previously synthesized
    if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
        return output_path
        
    rate = "+0%"
    pitch = "+0Hz"
    
    # Emotional prosody mapping using SSML traits
    e = emotion.lower()
    if any(x in e for x in ["angry", "furious", "yell", "shout", "mad"]):
        rate = "+15%"
        pitch = "+10Hz"
    elif any(x in e for x in ["sad", "cry", "depress", "heartbreak"]):
        rate = "-15%"
        pitch = "-10Hz"
    elif any(x in e for x in ["happy", "excite", "joy", "cheer", "laugh"]):
        rate = "+10%"
        pitch = "+5Hz"
    elif any(x in e for x in ["scare", "terrify", "anxious", "panic", "fear"]):
        rate = "+20%"
        pitch = "+15Hz"
    elif any(x in e for x in ["whisper", "quiet", "secret"]):
        rate = "-10%"
        pitch = "-15Hz"
    elif any(x in e for x in ["surprise", "shock", "gasp"]):
        rate = "+10%"
        pitch = "+20Hz"
        
    async with sem:
        communicator = edge_tts.Communicate(text, voice_name, rate=rate, pitch=pitch)
        await communicator.save(output_path)
        return output_path


def synthesize_all(
    lines: List[Dict],
    voice_map: Dict[str, str],
    output_dir: str,
    progress_callback=None,
) -> List[str]:
    """
    Streamlit-safe batch TTS generation.

    Streamlit already runs an event loop, so asyncio.run() raises
    'event loop is already running'. Instead, we spawn a dedicated
    background thread with its own fresh event loop, run all
    coroutines concurrently via asyncio.gather(), then block until
    every MP3 is on disk before returning.

    Args:
        lines: Structured dialogue list from extract_dialogue().
        voice_map: {character_name: edge_tts_voice_name}
        output_dir: Directory to save individual MP3 clips.
        progress_callback: Optional callable(current, total).

    Returns:
        Ordered list of absolute MP3 file paths (one per line).
    """
    os.makedirs(output_dir, exist_ok=True)
    default_voice = "en-US-AriaNeural"  # fallback narrator voice
    results: List[str] = []
    errors: List[Exception] = []
    completed_count = [0]
    total_tasks = 0

    async def _run_all():
        tasks = []
        paths = []
        sem = asyncio.Semaphore(20)  # Rate limit concurrent Edge-TTS connections
        
        # Build tasks
        for idx, line in enumerate(lines):
            character = line.get("character", "Narrator")
            emotion = line.get("emotion", "neutral")
            text = line.get("text", "").strip()
            if not text:
                paths.append(None)
                continue
            voice = voice_map.get(character, default_voice)
            out_path = os.path.join(output_dir, f"line_{idx:05d}.mp3")
            
            async def _task_wrapper(t_text, t_voice, t_path, t_emotion):
                res = await _synthesize_one(t_text, t_voice, t_path, sem, t_emotion)
                completed_count[0] += 1
                return res
                
            tasks.append(_task_wrapper(text, voice, out_path, emotion))
            paths.append(out_path)

        completed = await asyncio.gather(*tasks, return_exceptions=True)
        # Map results back — exceptions become None (skip silently)
        real_paths = []
        task_idx = 0
        for path in paths:
            if path is None:
                real_paths.append(None)
            else:
                outcome = completed[task_idx]
                task_idx += 1
                if isinstance(outcome, Exception):
                    print(f"TTS skipped for line due to error: {outcome}")
                    real_paths.append(None)
                else:
                    real_paths.append(path)
        return real_paths

    def _thread_target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            paths = loop.run_until_complete(_run_all())
            results.extend(paths)
        except Exception as e:
            errors.append(e)
        finally:
            loop.close()

    # Count valid lines for progress display
    total_tasks = sum(1 for line in lines if line.get("text", "").strip())
    
    t = threading.Thread(target=_thread_target, daemon=True)
    t.start()

    # Poll safely on main thread
    while t.is_alive():
        t.join(timeout=0.5)
        if progress_callback and total_tasks > 0:
            progress_callback(completed_count[0], total_tasks)

    if errors:
        raise errors[0]

    if progress_callback and total_tasks > 0:
        progress_callback(total_tasks, total_tasks)

    return results
