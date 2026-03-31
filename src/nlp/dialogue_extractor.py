"""LLM-based dialogue extraction with speaker attribution and emotion tagging."""
import asyncio
import json
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types


SYSTEM_PROMPT = """You are an expert literary analyst and audiobook producer.
Your job is to read a passage of a book and extract every utterance — 
both dialogue and narration — as structured JSON.

Rules:
1. Output ONLY a valid JSON array. No markdown, no extra text.
2. Each element must have exactly these keys:
   - "type": "dialogue" or "narration"
   - "character": the speaker's name, or "Narrator" for narration
   - "text": the exact text spoken/narrated (no surrounding quotes)
   - "emotion": one of: neutral, happy, sad, tense, dramatic, romantic, melancholic, upbeat, angry, scared
3. Do NOT skip any text. Every sentence must appear in the output.
4. For attribution, use context clues (speech-act verbs, prior turns) to identify the speaker.
5. Split long narrator blocks at natural sentence boundaries.
"""


def _build_prompt(chunk: str, overlap_prefix: str) -> str:
    context_parts = []

    if overlap_prefix:
        context_parts.append(
            f"[CONTEXT — end of previous passage, do NOT re-extract this]:\n{overlap_prefix}"
        )

    context_block = "\n\n".join(context_parts)
    if context_block:
        context_block = context_block + "\n\n---\n\n"

    return f"{context_block}[PASSAGE TO EXTRACT]:\n{chunk}"


def _parse_json_response(text: Optional[str]) -> List[Dict]:
    """Extract JSON array from model output, stripping markdown fences if present."""
    if not text:
        raise ValueError("Model returned empty or None response (possibly blocked by safety filters).")
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("JSON response is not a list.")
        
    valid_items = [item for item in parsed if isinstance(item, dict)]
    return valid_items


def test_connection(api_key: str, model_name: str = "gemini-1.5-flash") -> str:
    """
    Quick sanity-check: send a tiny JSON prompt and return the raw model output.
    Used by the UI to debug extraction failures.
    """
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents='Return this exact JSON array and nothing else: [{"type":"narration","character":"Narrator","text":"Hello.","emotion":"neutral"}]',
    )
    return response.text


def extract_dialogue(
    chunks: List[Tuple[str, str]],
    api_key: str,
    model_name: str = "gemini-1.5-flash",
    progress_callback=None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Process chunks concurrently through Gemini and return structured lines + errors.
    Returns:
        (all_lines, errors)
    """
    # 5-minute timeout (API expects milliseconds)
    client = genai.Client(
        api_key=api_key,
        http_options={"timeout": 300000},
    )

    results_array: List[List[Dict]] = [[] for _ in chunks]
    errors_array: List[str] = ["" for _ in chunks]
    completed_count = [0]
    total = len(chunks)

    async def _process_chunk(idx: int, chunk: str, overlap: str):
        prompt = _build_prompt(chunk, overlap)
        
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.2,
                        safety_settings=[
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                        ],
                    ),
                )
                raw = response.text
                parsed = _parse_json_response(raw)
                results_array[idx] = parsed
                break
            except json.JSONDecodeError as je:
                if attempt == 2:
                    bad_output = response.text if response and hasattr(response, "text") else ""
                    fix_prompt = (
                        f"The following text is supposed to be a valid JSON array but it isn't. "
                        f"Fix it and return ONLY the valid JSON array:\n\n{bad_output}"
                    )
                    try:
                        fix_response = await client.aio.models.generate_content(
                            model=model_name,
                            contents=fix_prompt,
                        )
                        parsed = _parse_json_response(fix_response.text)
                        results_array[idx] = parsed
                        break
                    except Exception as fe:
                        raw_preview = (response.text[:200] if response and hasattr(response, "text") else "N/A")
                        errors_array[idx] = f"Chunk {idx+1}: JSON repair failed. Raw output preview: {raw_preview!r}. Error: {fe}"
                else:
                    await asyncio.sleep(2)
            except Exception as e:
                if attempt == 2:
                    errors_array[idx] = f"Chunk {idx+1}: API error after 3 attempts: {type(e).__name__}: {e}"
                    # Inject a placeholder so the audiobook doesn't skip 6000 tokens silently
                    results_array[idx] = [{
                        "type": "narration",
                        "character": "Narrator",
                        "text": "[A section of the text was skipped here due to API content restrictions.]",
                        "emotion": "neutral"
                    }]
                else:
                    await asyncio.sleep(2)
        
        completed_count[0] += 1

    async def _run_all():
        tasks = []
        for i, (chunk, overlap) in enumerate(chunks):
            tasks.append(_process_chunk(i, chunk, overlap))
        await asyncio.gather(*tasks)

    # Run in a dedicated thread to avoid Streamlit event-loop clashes
    def _thread_target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_run_all())
        finally:
            loop.close()

    t = threading.Thread(target=_thread_target, daemon=True)
    t.start()

    # Poll the thread to update Streamlit progress safely on the main thread
    while t.is_alive():
        t.join(timeout=0.5)
        if progress_callback:
            progress_callback(completed_count[0], total)
            
    if progress_callback:
        progress_callback(total, total)

    # Flatten the ordered results
    all_lines: List[Dict] = []
    for lines in results_array:
        all_lines.extend(lines)
        
    errors = [e for e in errors_array if e]
    return all_lines, errors
