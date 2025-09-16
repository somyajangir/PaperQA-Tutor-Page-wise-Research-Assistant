# agent/summarizer.py
import os
from functools import lru_cache
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Default location; you can override at runtime with env var:
#   set SUMMARIZER_MODEL_DIR="path/to/merged_full_model"
MODEL_DIR = os.environ.get("SUMMARIZER_MODEL_DIR", "merged_full_model")

@lru_cache(maxsize=1)
def _load():
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(
            f"[Summarizer] Local model not found at '{MODEL_DIR}'. "
            f"Place your 'merged_full_model/' there or set SUMMARIZER_MODEL_DIR."
        )
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    mdl.eval()
    return tok, mdl

def summarize_page(text: str, max_src_len: int = 512, max_tgt_len: int = 96) -> str:
    """
    Summarize a single page of text using your fine-tuned local FLAN-T5.
    Uses beam search for stronger summaries (matches your notebook inference).
    """
    tok, mdl = _load()
    prompt = "summarize: " + (text or "").strip()
    if not prompt.strip():
        return "No text on this page to summarize."

    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_src_len)
    with torch.no_grad():
        out = mdl.generate(
            **enc,
            num_beams=4,
            max_length=max_tgt_len,
            min_new_tokens=16,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            early_stopping=True,
        )
    return tok.decode(out[0], skip_special_tokens=True).strip()
