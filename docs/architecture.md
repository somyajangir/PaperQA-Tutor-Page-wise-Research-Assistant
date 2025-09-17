# 🏗️ Architecture — PaperQA Section Tutor

## 🎯 Overview

The **PaperQA Section Tutor** is an intelligent agent designed to help students comprehensively study research papers **page-by-page**. The system employs a multi-stage approach:

1. 📄 **PDF Ingestion** → 2. 🔍 **RAG Index** → 3. 📚 **Page Explanation** → 4. ❓ **Page Q&A** → 5. 🧪 **Quiz & Grading**

Plus an **🤖 offline local summarizer** fine-tuned on Gemma-3-270M for enhanced learning experience.

---

## 🧩 System Components

| Component | 🎯 Role | 🛠️ Key Libraries/Models |
|-----------|---------|------------------------|
| `agent/ingestor.py` | Extract page text with layout-preserving newlines | PyMuPDF (`fitz`) |
| `agent/embeddings.py` | Singleton embedding model loader | `sentence-transformers/all-MiniLM-L6-v2` |
| `agent/indexer.py` | Word-chunker + FAISS IP index (cosine) over all page chunks | FAISS |
| `agent/page_qa.py` | Per-page QA with small FAISS + Gemini synthesis + forced citations | FAISS, Google Generative AI |
| `agent/explainer.py` | Page explainer (Conceptual Summary / Breakdown / Key Terms) | Google Generative AI |
| `agent/qg_generator.py` | Quiz generation (exact 5 Q/A JSON) | Google Generative AI |
| `agent/evaluator.py` | Quiz grading (Correct / Partial / Incorrect + feedback) | Google Generative AI |
| `agent/summarizer.py` | **🌟 Local summarization (fine-tuned Gemma-3-270M)** | Transformers (AutoModelForCausalLM) |
| `agent/logger.py` | JSONL logging helper | Python stdlib |
| `ui/app.py` | Streamlit UI (PDF viewer, Page Tutor, RAG tab) | Streamlit |

### 🌐 External Dependencies
- **🧠 External Services:** Google Gemini (2.5 Flash Lite) for explanation, QG, grading, and RAG synthesis
- **🤖 Local Models:** MiniLM embeddings + Gemma-3-270M (merged LoRA) for local page summarization

---

## 🏗️ System Architecture Overview

```mermaid
flowchart TD
    %% User Input
    USER[👤 Student] 
    PDF[📄 Research Paper PDF]
    
    %% Core Processing Pipeline
    UPLOAD[📤 Upload PDF]
    INGEST[🔧 PDF Processing<br/>Extract text per page]
    CHUNK[✂️ Text Chunking<br/>Split into segments]
    EMBED[🧠 Create Embeddings<br/>MiniLM-L6-v2]
    INDEX[🗂️ FAISS Vector Store<br/>Searchable Knowledge Base]
    
    %% Two Main Learning Paths
    TUTOR[🎓 PAGE TUTOR<br/>Focused Learning]
    RAG[🔍 RAG SYSTEM<br/>Global Q&A]
    
    %% Page Tutor Features
    EXPLAIN[📚 Page Explanation<br/>Concepts & Terms]
    SUMMARY[🤖 Local Summary<br/>Gemma-3-270M Offline]
    PAGEQA[❓ Page Questions<br/>Specific to current page]
    QUIZ[🧪 Generate Quiz<br/>5 Questions]
    GRADE[📊 Auto-Grading<br/>Correct/Partial/Wrong]
    
    %% RAG Features
    SEARCH[🔎 Semantic Search<br/>Find relevant chunks]
    ANSWER[💬 AI Answer<br/>With source citations]
    
    %% Connections - Input Flow
    USER --> PDF
    PDF --> UPLOAD
    UPLOAD --> INGEST
    INGEST --> CHUNK
    CHUNK --> EMBED
    EMBED --> INDEX
    
    %% Main System Split
    INDEX --> TUTOR
    INDEX --> RAG
    
    %% Page Tutor Flow
    TUTOR --> EXPLAIN
    TUTOR --> SUMMARY
    TUTOR --> PAGEQA
    TUTOR --> QUIZ
    QUIZ --> GRADE
    
    %% RAG Flow
    RAG --> SEARCH
    SEARCH --> ANSWER
    
    %% User Interaction
    USER -.-> TUTOR
    USER -.-> RAG
    GRADE -.-> USER
    ANSWER -.-> USER
    
    %% Styling for clarity
    classDef userClass fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    classDef processClass fill:#f1f8e9,stroke:#388e3c,stroke-width:2px
    classDef tutorClass fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef ragClass fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef aiClass fill:#fff8e1,stroke:#f57c00,stroke-width:2px
    
    class USER,PDF userClass
    class UPLOAD,INGEST,CHUNK,EMBED,INDEX processClass
    class TUTOR,EXPLAIN,PAGEQA,QUIZ,GRADE tutorClass
    class RAG,SEARCH,ANSWER ragClass
    class SUMMARY aiClass
```

## 🔄 User Journey: How Students Learn with PaperQA

```mermaid
journey
    title Student Learning Experience
    section Upload & Setup
        Upload PDF: 5: Student
        Process Document: 3: System
        Build Knowledge Base: 3: System
    section Page-by-Page Learning
        Read Page: 4: Student
        Get AI Explanation: 5: Student, AI
        Generate Local Summary: 5: Student, Gemma-3-270M
        Ask Page Questions: 4: Student, AI
    section Test Understanding
        Take Quiz: 4: Student
        Get Instant Feedback: 5: Student, AI
        Review Mistakes: 4: Student
    section Explore Whole Paper
        Ask Global Questions: 5: Student, RAG
        Get Cited Answers: 5: Student, AI
        Cross-reference Pages: 4: Student
```

## 🎯 Simple Process Flow

```mermaid
flowchart LR
    %% Simple 5-step process
    A[📄 1. Upload Paper] --> B[🔧 2. AI Processes]
    B --> C[🎓 3. Learn Page-by-Page]
    C --> D[🧪 4. Take Quiz]
    D --> E[🔍 5. Ask Questions]
    
    %% What happens at each step
    B -.->|"Creates searchable<br/>knowledge base"| B1[🗂️ Vector Database]
    C -.->|"Explanations, summaries<br/>page-specific Q&A"| C1[📚 Study Tools]
    D -.->|"Auto-graded with<br/>instant feedback"| D1[📊 Smart Assessment]
    E -.->|"Answers with<br/>source citations"| E1[💬 AI Tutor]
    
    %% Styling
    classDef stepClass fill:#e3f2fd,stroke:#1976d2,stroke-width:3px,color:#000
    classDef helpClass fill:#f1f8e9,stroke:#388e3c,stroke-width:1px,color:#000
    
    class A,B,C,D,E stepClass
    class B1,C1,D1,E1 helpClass
```

## 🧠 AI Components Working Together

```mermaid
mindmap
  root((🎓 PaperQA<br/>Tutor))
    🔧 Processing Engine
      📖 PDF Reader
        Extracts text per page
        Preserves formatting
      ✂️ Smart Chunking
        Breaks into segments
        Maintains context
      🧠 Embeddings
        MiniLM-L6-v2
        Semantic understanding
    🎓 Learning Modules
      📚 Page Explainer
        Conceptual summaries
        Key terms extraction
        Gemini 2.5 powered
      🤖 Local Summarizer
        Gemma-3-270M offline
        No API costs
        Private processing
      ❓ Q&A Agent
        Page-specific answers
        Forced citations
        Context aware
    🧪 Assessment System
      Quiz Generator
        5 targeted questions
        JSON structured
      📊 Smart Grader
        3-level evaluation
        Constructive feedback
        Learning guidance
    🔍 Search & Discovery
      🗂️ Vector Database
        FAISS indexing
        Cosine similarity
      💬 RAG Answers
        Source citations
        Evidence-based
        Global knowledge
```

### 2️⃣ Global RAG Q&A Sequence

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant UI as 🖥️ Streamlit UI
    participant IDX as 🔍 RAGIndexer (FAISS)
    participant LLM as 🧠 Gemini 2.5 Flash Lite

    U->>UI: ❓ Ask RAG question
    UI->>IDX: 🔎 retrieve(question, k)
    IDX-->>UI: 📄 top-k chunks (+page numbers)
    UI->>LLM: 🧠 synthesize answer from chunks
    LLM-->>UI: ✅ concise answer + citations
    UI-->>U: 📋 Answer + evidence (snippets with page nos.)
    
    Note over U,LLM: 🌐 Global document knowledge with citations
```

### 3️⃣ Page Tutor Learning Sequence

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant UI as 🖥️ Streamlit UI
    participant EXP as 📝 Explainer (Gemini)
    participant SUM as 🤖 Summarizer (Gemma-3-270M)
    participant PQA as ❓ PageQAAgent

    U->>UI: 📚 Open Page Tutor
    UI->>EXP: 📖 Explain page (with context)
    EXP-->>UI: 📋 TL;DR + Breakdown + Key Terms
    
    U->>UI: 🔘 Click "Summarize (Gemma-3-270M)"
    UI->>SUM: 🤖 summarize_page(text)
    SUM-->>UI: 📝 local offline summary
    
    U->>UI: ❓ Ask page-scoped question
    UI->>PQA: 🔍 retrieve + answer
    PQA-->>UI: ✅ answer + forced citations
    
    Note over U,PQA: 📄 Page-specific learning with local AI
```

### 4️⃣ Quiz Generation & Evaluation Flow

```mermaid
flowchart TD
    A[📚 Page Content] --> B[🧪 Quiz Generator<br/>Gemini API]
    B --> C[📝 5 Questions Generated<br/>JSON Format]
    C --> D[👤 User Attempts Quiz]
    D --> E[📊 Answer Evaluator<br/>Gemini Grading]
    E --> F{📈 Evaluation Result}
    
    F -->|✅| G[Correct<br/>+ Positive Feedback]
    F -->|⚠️| H[Partial<br/>+ Constructive Hints]
    F -->|❌| I[Incorrect<br/>+ Learning Guidance]
    
    G --> J[📋 Performance Summary]
    H --> J
    I --> J
    
    style A fill:#e8f5e8
    style C fill:#fff3e0
    style F fill:#f3e5f5
    style J fill:#e1f5fe
```

---

## 🔧 Technical Design Choices

### 🎯 Core Principles

| **Principle** | **Implementation** | **Benefit** |
|---------------|-------------------|-------------|
| **🔄 Singleton Embeddings** | `@lru_cache` for sentence-transformer loading | Memory efficiency, faster response times |
| **📐 Cosine via IP Normalized** | L2-normalized vectors + FAISS `IndexFlatIP` | Accurate similarity without cosine overhead |
| **📌 Forced Citations** | Automatic citation appending if Gemini omits | Consistent UI experience and source tracking |
| **🤖 Local Summarizer** | Gemma-3-270M (LoRA merged) direct calls | Zero API latency, cost-free offline operation |
| **🏗️ Separation of Concerns** | UI orchestrates, agents handle specific tasks | Maintainable, testable, scalable architecture |

### 🚀 Performance Optimizations

```mermaid
graph LR
    subgraph "⚡ Speed Optimizations"
        A[🧠 Singleton Embedding Model]
        B[🔍 Efficient FAISS Indexing]
        C[🤖 Local Gemma-3-270M]
    end
    
    subgraph "💾 Memory Management"
        D[📦 Lazy Loading]
        E[🗂️ Chunked Processing]
        F[🔄 Resource Reuse]
    end
    
    subgraph "🎯 User Experience"
        G[📱 Streamlit Responsiveness]
        H[📊 Real-time Feedback]
        I[🔊 TTS Integration]
    end
    
    A --> D
    B --> E
    C --> F
    D --> G
    E --> H
    F --> I
```

---

## 📈 System Capabilities Matrix

| Feature Category | Capability | Technology Stack |
|-----------------|------------|------------------|
| **📄 Document Processing** | Multi-format PDF ingestion | PyMuPDF |
| **🔍 Information Retrieval** | Semantic search with citations | FAISS + MiniLM-L6-v2 |
| **🧠 AI Understanding** | Context-aware explanations | Gemini 2.5 Flash Lite |
| **🤖 Local Processing** | Offline summarization | Fine-tuned Gemma-3-270M |
| **🎓 Learning Assessment** | Intelligent quiz grading | Custom evaluation algorithms |
| **🔊 Accessibility** | Text-to-speech support | Browser TTS APIs |
| **📊 Progress Tracking** | Learning analytics | JSONL logging system |

---

## 🎭 Author

**👨‍🎓 Somya Jangir** • B23CI1036  
**🏛️ Civil & Infrastructure Engineering**  
**🏫 Indian Institute of Technology, Jodhpur**