# agent/embeddings.py
from functools import lru_cache
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

@lru_cache(maxsize=None) # Set maxsize=None for a singleton-like behavior
def get_embedding_model() -> SentenceTransformer:
    """
    Loads and caches the embedding model. All agents will call this
    function to get the exact same model instance in memory.
    """
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' into cache...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded.")
    return model