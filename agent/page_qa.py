# agent/page_qa.py 
# Renamed from section_qa.py
import faiss
import numpy as np
import google.generativeai as genai
import os
from .indexer import word_chunker 
from .embeddings import get_embedding_model
from typing import List, Dict

class PageQAAgent: # Renamed class
    """
    A stateful agent that builds a private index ONCE for the current PAGE.
    """
    
    def __init__(self, page_text: str, paper_id: str, page_num: int):
        self.model = get_embedding_model()
        self.paper_id = paper_id
        self.page = page_num
        
        self.chunks = word_chunker(page_text)
        self.chunk_metadata = []

        for chunk in self.chunks:
            self.chunk_metadata.append({
                "text": chunk,
                "page": self.page
            })
            
        if self.chunks:
            chunk_embeddings = self.model.encode(self.chunks)
            faiss.normalize_L2(chunk_embeddings)
            d = chunk_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(chunk_embeddings.astype(np.float32))
        else:
            self.index = None

    def answer_question(self, question: str) -> str:
        """Answers a question grounded *only* in the page's text."""
        if self.index is None:
            return "There is no text on this page to search."

        # 1. Retrieve (k=3)
        query_embedding = self.model.encode([question])
        faiss.normalize_L2(query_embedding)
        _, indices = self.index.search(query_embedding.astype(np.float32), 3)
        
        retrieved_snippets, citation_set = [], set()
        for i in indices[0]:
            if i != -1:
                meta = self.chunk_metadata[i]
                retrieved_snippets.append(meta["text"])
                citation_set.add(f"[{self.paper_id}, p. {meta['page']}]") # Citation will just be this page

        if not retrieved_snippets:
            return "I could not find an answer on this specific page."

        # 2. Synthesize Answer (Using Gemini)
        context_str = "\n\n".join(retrieved_snippets)
        citations_line = "; ".join(sorted(list(citation_set)))

        prompt = f"""
        Based ONLY on the following context snippets from page {self.page}, answer the user's question.
        Quote a short supporting phrase. Answer in 2-3 concise sentences.
        At the end, you MUST output exactly one line beginning with "Citations:":
        
        Context Snippets:
        {context_str}
        
        Question:
        {question}
        
        Concise Answer:
        [Your answer here]
        Citations: {citations_line}
        """
        
        try:
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            model = genai.GenerativeModel("gemini-2.5-flash-lite")
            response = model.generate_content(prompt)
            final_text = response.text
            
            if "Citations:" not in final_text:
                final_text += f"\n\nCitations: {citations_line}"
                
            return final_text
        except Exception as e:
            return f"Error synthesizing answer. Raw snippets: {context_str}"