# agent/indexer.py
import faiss
import numpy as np
import pickle
import re
from typing import List, Dict
from .embeddings import get_embedding_model 

def word_chunker(text: str, max_words: int = 150, overlap_words: int = 30):
    """Custom sentence-aware word chunker adhering to the spec."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences: return []
    chunks, current_chunk_words = [], []
    
    for sentence in sentences:
        sentence_words = sentence.split()
        if not sentence_words: continue
        
        if len(current_chunk_words) + len(sentence_words) > max_words and current_chunk_words:
            chunks.append(" ".join(current_chunk_words))
            overlap_start_index = max(0, len(current_chunk_words) - overlap_words)
            current_chunk_words = current_chunk_words[overlap_start_index:]
            
            if len(current_chunk_words) + len(sentence_words) > max_words:
                 if current_chunk_words:
                    chunks.append(" ".join(current_chunk_words))
                 current_chunk_words = sentence_words[:max_words]
                 continue
        current_chunk_words.extend(sentence_words)

    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))
    return chunks

class RAGIndexer:
    def __init__(self):
        self.model = get_embedding_model() 
        self.index = None
        self.metadata_store = [] 

    def build_index(self, pages_data: List[Dict], paper_id: str):
        """Builds a FAISS index from the list of PAGE data."""
        all_chunk_metadata = []
        
        # --- NEW LOGIC: Iterate pages directly, not sections ---
        for page_dict in pages_data:
            page_text = page_dict['text']
            page_num = page_dict['page']
            chunks = word_chunker(page_text)
            
            for chunk_text in chunks:
                all_chunk_metadata.append({
                    "paper_id": paper_id,
                    "page": page_num, # Citation is now exact
                    "text": chunk_text
                })

        if not all_chunk_metadata: return
        self.metadata_store = all_chunk_metadata
        chunk_texts = [meta["text"] for meta in all_chunk_metadata]
        
        print(f"Embedding {len(chunk_texts)} chunks (word-based) for RAG index...")
        embeddings = self.model.encode(chunk_texts, show_progress_bar=True, batch_size=64)
        
        faiss.normalize_L2(embeddings) 
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d) # Cosine Similarity
        self.index.add(embeddings.astype(np.float32))
        print("Global RAG index built successfully with IndexFlatIP.")

    def save_index(self, path_prefix: str):
        faiss.write_index(self.index, f"{path_prefix}.index")
        with open(f"{path_prefix}.meta.pkl", "wb") as f:
            pickle.dump(self.metadata_store, f)

    def load_index(self, path_prefix: str):
        self.index = faiss.read_index(f"{path_prefix}.index")
        with open(f"{path_prefix}.meta.pkl", "rb") as f:
            self.metadata_store = pickle.load(f)

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        if self.index is None: raise Exception("Index not built.")
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        _, indices = self.index.search(query_embedding.astype(np.float32), k)
        return [self.metadata_store[i] for i in indices[0] if i != -1]