# 📚 PaperQA Section Tutor — IIT Jodhpur

<div align="center">

**An intelligent AI agent for interactive research paper learning**

*Study research papers page-by-page with AI-powered explanations, Q&A, and personalized quizzes*

---

**👨‍🎓 Author:** Somya Jangir (B23CI1036)  
**🏛️ Department:** Civil and Infrastructure Engineering  
**🏫 Institution:** IIT Jodhpur

---

</div>

## 🎯 Overview

An interactive AI agent that transforms how you study research papers:

- 📖 **Page-by-page explanations** in clean, structured English
- ❓ **Intelligent Q&A** with RAG over uploaded PDFs (with citations)
- 🧪 **Quiz generation** with smart grading (Correct / Partial / Incorrect)
- 🤖 **Local summarizer** using fine-tuned Gemma-3-270M (LoRA) — runs offline, no API required

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **PDF Ingestion** | PyMuPDF → Structured Page Plan |
| 🔍 **RAG System** | sentence-transformers `all-MiniLM-L6-v2` + FAISS with citations |
| 🎓 **Page Tutor** | Gemini 2.5 Flash Lite — *Conceptual Summary / Breakdown / Key Terms* |
| 📝 **Local Summary** | Fine-tuned Gemma-3-270M — 3-4+ sentence summaries, offline |
| 🎯 **Quiz Generation** | Gemini → Smart Grading with detailed feedback |
| 🔊 **TTS Playback** | Audio explanations and summaries |
| 📊 **Lightweight Logging** | JSONL event tracking (disabled for submission) |

---

## 🏗️ Repository Structure

```
.
├─ 🤖 agent/
│   ├─ embeddings.py          # Shared singleton embedding model (MiniLM-L6-v2)
│   ├─ ingestor.py            # PyMuPDF page extractor
│   ├─ indexer.py             # Word-chunker + FAISS RAG index
│   ├─ page_qa.py             # Per-page QA agent (retrieval + Gemini synthesis)
│   ├─ explainer.py           # Page explainer (Gemini 2.5 Flash Lite)
│   ├─ qg_generator.py        # Quiz generator (Gemini → JSON)
│   ├─ evaluator.py           # Grader (Gemini → feedback)
│   ├─ summarizer.py          # 🌟 Fine-tuned Gemma-3-270M local summarizer
│   └─ logger.py              # JSONL event logging
│
├─ 🖥️ ui/
│   └─ app.py                 # Streamlit app: Viewer + Page Tutor + RAG
│
├─ 📊 Finetuning results, graphs and samples/
│   ├─ loss_curve.png
│   ├─ val_metrics_beam4.png
│   ├─ training_metrics.csv
│   ├─ sample_predictions.csv
│   └─ run_summary.json       # Gemma-3-270M LoRA run (CNN/DM 3.0.0)
│
├─ 🧠 Gemma 3 Finetuning.py   # Fine-tuning notebook/script (RunPod/Colab)
├─ 🏋️ merged_full_model/      # Model weights directory
├─ 📋 requirements.txt
└─ 📚 docs/
    ├─ ARCHITECTURE.md
    └─ DATA_SCIENCE_REPORT.md
```

> **📁 Note:** The **"Finetuning results, graphs and samples"** folder contains all artifacts produced during fine-tuning (plots, metrics, samples, run summary).

---

## 🚀 Quick Start Guide

### 1️⃣ Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ API Configuration

Create a `.env` file in the repository root:

```bash
GOOGLE_API_KEY=your_gemini_key_here

# Optional: Custom model path
# SUMMARIZER_MODEL_DIR=/absolute/path/to/merged_full_model
```

### 3️⃣ Local Summarizer Weights

**Preferred method:** Download the **merged Gemma-3-270M** folder and place it at:

```
merged_full_model/
```

Alternatively, set `SUMMARIZER_MODEL_DIR` in your `.env` file.

### 4️⃣ Launch Application

```bash
streamlit run ui/app.py
```

---

## 🎮 How to Use

1. **📤 Upload** a PDF from the sidebar
2. **🎓 Page Tutor** tab → Get explanations + **Local Summary (Gemma-3-270M)** 
3. **❓ Ask Questions** → Page-scoped inquiries with intelligent responses
4. **🧪 Generate Quiz** → Test your understanding with auto-graded questions
5. **🔍 RAG Tab** → Ask about the entire document with evidence citations

---

## 🧪 Evaluation & Performance

- **📊 Summarizer (Gemma-3-270M LoRA):** Evaluated on CNN/DailyMail 3.0.0 with ROUGE/BLEU metrics
- **🎯 Agent Performance:** Quiz grader provides **Correct/Partial/Incorrect** classifications with constructive feedback
- **📈 Results:** See `Finetuning results, graphs and samples/run_summary.json` for detailed metrics

---

## 📜 Important Notes

### 🔐 Security & Ethics
- ⚠️ **Never commit API keys or HF tokens**
- 📦 **Large model weights are not committed** (see Local summarizer weights section)
- 🎓 **Academic Integrity:** Use as a study aid and cite sources appropriately

---

