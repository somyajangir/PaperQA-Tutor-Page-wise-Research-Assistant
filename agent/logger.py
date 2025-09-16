# agent/logger.py
import json
import logging
import hashlib
from datetime import datetime

LOG_FILE = "logs/interactions.jsonl"

def setup_logger():
    logging.basicConfig(
        filename='logs/app.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

def hash_prompt(prompt):
    return hashlib.md5(prompt.encode()).hexdigest()

def log_interaction(paper_id, section, action, details, prompt="", context_len=0, model=""):
    log_entry = {
        "ts": datetime.utcnow().isoformat(),
        "paper_id": paper_id,
        "section": section,
        "action": action,
        "model": model,
        "details": details, 
        "prompt_hash": hash_prompt(prompt) if prompt else None,
        "context_len_chars": context_len
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logging.error(f"Failed to write to JSONL log: {e}")