# agent/summarizer.py
import os
from functools import lru_cache
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Default location; override with:  export/set SUMMARIZER_MODEL_DIR=/path/to/merged_full_model
MODEL_DIR = os.environ.get("SUMMARIZER_MODEL_DIR", "merged_full_model")

@lru_cache(maxsize=1)
def _load():
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(
            f"[Summarizer] Local model not found at '{MODEL_DIR}'. "
            f"Place your 'merged_full_model/' there or set SUMMARIZER_MODEL_DIR."
        )
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    # Gemma is causal; make sure pad token exists
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device).eval()
    return tok, mdl, device

def summarize_page(text: str, max_src_len: int = 1024, max_tgt_len: int = 160) -> str:
    """
    Summarize a single page using your fine-tuned Gemma-3-270M (LoRA merged).
    Uses the same instruction + 'Summary:' continuation format as training,
    and beam search for stronger outputs.
    """
    tok, mdl, device = _load()
    if not text or not text.strip():
        return "No text on this page to summarize."

    instr = "Summarize the following content in 3–4 sentences."
    resp_prefix = "\n\nSummary: "
    prompt = f"{instr}\n\nArticle:\n{text.strip()}{resp_prefix}"

    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_src_len).to(device)
    with torch.no_grad():
        out = mdl.generate(
            **enc,
            num_beams=4,
            max_new_tokens=max_tgt_len,   # allow longer summaries
            min_new_tokens=32,            # encourage 3–4+ sentences
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            early_stopping=True,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    dec = tok.decode(out[0], skip_special_tokens=True)
    # Strip the prompt if the model echoes it
    return dec[len(prompt):].strip() if dec.startswith(prompt) else dec.strip()
