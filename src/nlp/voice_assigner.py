"""AI-powered automatic voice assignment for detected characters."""
import json
import re
from typing import Dict, List

from google import genai
from google.genai import types


ASSIGN_PROMPT = """You are an expert audio engineer casing voices for an audiobook.
You have a list of characters and a list of available AI voices.
Your job is to assign exactly ONE voice to EVERY character based on their likely gender, age, or personality.

Rules:
1. ONLY use voices from the provided "AVAILABLE VOICES" list.
2. Output ONLY a valid JSON object mapping "Character Name" -> "voice-name". no markdown fringes.
3. Every character MUST be assigned a voice. Do not skip any.
4. Try to give distinct characters distinct voices if possible.
5. "Narrator" should generally get a professional, clear voice (e.g., Aria or Guy).
"""


def auto_assign_voices(
    characters: List[str],
    available_voices: List[str],
    api_key: str,
    model_name: str = "gemini-1.5-flash",
) -> Dict[str, str]:
    """
    Ask Gemini to map characters to TTS voices.
    Returns: {"Character Name": "en-US-AriaNeural", ...}
    """
    if not characters:
        return {}
        
    client = genai.Client(api_key=api_key)
    
    voices_str = "\n".join(f"- {v}" for v in available_voices)
    chars_str = "\n".join(f"- {c}" for c in characters)
    
    prompt = f"{ASSIGN_PROMPT}\n\n[AVAILABLE VOICES]:\n{voices_str}\n\n[CHARACTERS TO ASSIGN]:\n{chars_str}"
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
            )
        )
        
        text = response.text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
        text = text.strip()
        
        mapping = json.loads(text)
        
        # Verify and sanitize the output
        final_map = {}
        for char in characters:
            assigned = mapping.get(char)
            # Fallback if hallucinated or skipped
            if not assigned or assigned not in available_voices:
                assigned = "en-US-AriaNeural" if "en-US-AriaNeural" in available_voices else available_voices[0]
            final_map[char] = assigned
            
        return final_map
        
    except Exception as e:
        raise RuntimeError(f"Gemini API Error during auto-assign: {e}")
