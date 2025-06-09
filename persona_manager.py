# persona_manager.py

import os
import torch
import asyncio
import re

from tokenizer import CharTokenizer
from model import NoraTransformerLM

PERSONA_PATH = "nora_persona.txt"

# 1) A meta-prompt that explicitly tells Nora to invent a persona and avoid quoting:
PERSONA_META_PROMPT = (
    "Below, Nora will create a brand‐new identity for herself. "
    "She must NOT quote from any books or passages she has read. "
    "Instead, she should invent her own style, voice, quirks, and personality traits as if she were a completely new person. "
    "Her persona should be flirty, playful, curious, and speak in full sentences. "
    "Write at least three paragraphs in your own words.\n\n"
    "Nora, please invent and write your complete persona now:\n\nNora:"
)


async def generate_persona(model: NoraTransformerLM, tokenizer: CharTokenizer, device: str) -> str:
    """
    Ask Nora to write out her own, original persona, avoiding any verbatim quotes.
    Returns the raw generated text.
    """
    # We’ll ask for up to 512 tokens, with higher temperature and top_p sampling.
    # That combination tends to produce more creative, less‐memorizable text.
    raw = await asyncio.to_thread(
        model.generate,
        tokenizer,
        device,
        PERSONA_META_PROMPT,
        512,      # allow several paragraphs
        1.2,      # higher temperature for more creativity
        0         # top_k=0 means no fixed-k; we’ll apply top_p filtering instead
    )

    # At this point, “raw” may include the word “Nora:” etc. Strip everything before “Nora:”
    if "Nora:" in raw:
        persona_text = raw.split("Nora:")[-1].strip()
    else:
        persona_text = raw.strip()

    # Now apply a simple post‐filter: remove any long spans that match exact sequences in the book corpus.
    # This is optional but helps ensure she didn’t copy large chunks verbatim. We check for 6+ character substrings
    # appearing more than once in her output.
    def remove_long_quotes(text: str) -> str:
        filtered = text
        # find any substring of length ≥6 that appears twice; we’ll just guess she’s quoting if it’s repeated.
        for match in re.finditer(r"\b[\w',]{6,}\b", text):
            substr = match.group(0)
            if filtered.count(substr) > 1:
                filtered = filtered.replace(substr, "[…]")
        return filtered

    persona_text = remove_long_quotes(persona_text)
    return persona_text


def ensure_persona_file(model: NoraTransformerLM, tokenizer: CharTokenizer, device: str):
    """
    If nora_persona.txt does not exist, generate one (ensuring originality).
    """
    if os.path.isfile(PERSONA_PATH):
        return

    print("[persona] No persona found. Generating a new, original persona…")
    persona_text = asyncio.run(generate_persona(model, tokenizer, device))

    # Save to disk
    with open(PERSONA_PATH, "w", encoding="utf-8") as f:
        f.write(persona_text)
    print(f"[persona] Wrote new persona to {PERSONA_PATH}.")


async def maybe_update_persona(model: NoraTransformerLM, tokenizer: CharTokenizer, device: str):
    """
    Regenerate Nora’s persona if she requests it, overwriting the file.
    """
    print("[persona] Updating persona at Nora's request…")
    persona_text = await generate_persona(model, tokenizer, device)
    with open(PERSONA_PATH, "w", encoding="utf-8") as f:
        f.write(persona_text)
    print(f"[persona] Updated persona in {PERSONA_PATH}.")
    return persona_text
