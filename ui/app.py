# ui/app.py
import streamlit as st
import os, sys
import base64
from gtts import gTTS
from agent import summarizer
import io
import google.generativeai as genai
from dotenv import load_dotenv

# imports from your project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent import ingestor, indexer, explainer, page_qa, qg_generator, evaluator, logger
from streamlit_pdf_viewer import pdf_viewer

# ---------- page setup (UI only) ----------
st.set_page_config(page_title="PaperQA Tutor", layout="wide", initial_sidebar_state="expanded")

# keep your env/dirs exactly as before
load_dotenv()
OUTPUT_DIR = "outputs"
UPLOAD_DIR = os.path.join(OUTPUT_DIR, "uploads")
LORA_PATH = os.path.join(OUTPUT_DIR, "lora", "t5_qg")
RETRIEVAL_DIR = os.path.join(OUTPUT_DIR, "retrieval")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RETRIEVAL_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "lora", "t5_qg"), exist_ok=True)
logger.setup_logger()

# ---------- cached loaders (unchanged wiring) ----------
@st.cache_resource
def load_qg_agent():
    try:
        return qg_generator.QuestionGenerator()
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load QuestionGenerator: {e}")
        return None

@st.cache_resource
def load_rag_agent():
    try:
        return indexer.RAGIndexer()
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load RAG agent: {e}")
        return None

# ---------- helpers ----------
def speak_text(text):
    try:
        tts = gTTS(text=text, lang='en')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        print(f"gTTS Error: {e}")
        return None

def init_session_state():
    defaults = {
        "pdf_bytes": None,
        "paper_id": "",
        "page_roadmap": [],
        "current_page_index": 0,
        "memory_summaries": ["- No sections covered yet."],
        "current_explanation": None,
        "current_quiz": [],
        "quiz_report": None,
        "page_agent": None,
        "current_screen": "upload",
        "chat_messages": [],
        # UI state
        "viewer_height": 780,
        "rag_last_results": [],
        "rag_k": 3,
        "rag_input": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# preload agents
if "qg_agent" not in st.session_state:
    st.session_state.qg_agent = load_qg_agent()
if "rag_agent" not in st.session_state:
    st.session_state.rag_agent = load_rag_agent()

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("📚 PaperQA Tutor")

    # API key handling (unchanged)
    api_key_present = bool(os.environ.get("GOOGLE_API_KEY"))
    if not api_key_present:
        st.warning("Google API Key not found in .env file.")
        api_key_input = st.text_input("Enter Google API Key:", type="password")
        if api_key_input:
            os.environ["GOOGLE_API_KEY"] = api_key_input
            st.success("Google API Key set for this session!")
    else:
        st.success("Google API Key loaded from .env file.")

    if st.session_state.qg_agent is None:
        st.error("Quiz Agent could not be loaded. Check logs.")

    uploaded_file = st.file_uploader("Upload your Research Paper (PDF)", type="pdf")

    if uploaded_file and st.session_state.paper_id != uploaded_file.name:
        with st.spinner("Processing PDF..."):
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            pdf_bytes = uploaded_file.getvalue()
            with open(file_path, "wb") as f:
                f.write(pdf_bytes)

            paper_id = uploaded_file.name
            pages_data = ingestor.ingest_pdf(file_path, paper_id)  # page-by-page plan
            st.session_state.rag_agent.build_index(pages_data, paper_id)

            init_session_state()  # reset loop state
            st.session_state.page_roadmap = pages_data
            st.session_state.pdf_bytes = pdf_bytes
            st.session_state.paper_id = paper_id
            st.session_state.current_screen = "tutor"
            logger.log_interaction(paper_id, "N/A", "pdf_upload", {"pages": len(pages_data)})
            st.rerun()

    # page list (navigation)
    if st.session_state.page_roadmap:
        st.subheader("Pages")
        for i, page_data in enumerate(st.session_state.page_roadmap):
            page_num = page_data["page"]
            page_btn = st.button(
                f"Page {page_num}",
                key=f"page_nav_{i}",
                type=("primary" if i == st.session_state.current_page_index else "secondary"),
                use_container_width=True
            )
            if page_btn:
                st.session_state.current_page_index = i
                st.session_state.current_explanation = None
                st.session_state.current_quiz = []
                st.session_state.quiz_report = None
                st.session_state.page_agent = None
                st.session_state.current_screen = "tutor"
                st.rerun()

# ========== MAIN AREA ==========
has_pdf = bool(st.session_state.pdf_bytes)
st.markdown(
    """
    <style>
      .rag-banner {padding: 10px 14px; border-radius: 10px; background:#eef6ff; border:1px solid #cfe6ff; margin-bottom: 14px;}
      .evidence-box {max-height: 520px; overflow-y: auto; padding-right: 8px;}
      .tight {margin-top: -8px;}
    </style>
    """,
    unsafe_allow_html=True
)

# two main columns: viewer (left) + workspace (right)
col_viewer, col_work = st.columns([6, 7], gap="large")

# ---------- LEFT: PDF VIEWER ----------
with col_viewer:
    st.subheader("📄 Document Viewer")
    if has_pdf:
        current_page_num = st.session_state.page_roadmap[st.session_state.current_page_index]['page']
        # controls
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            prev_dis = st.session_state.current_page_index == 0
            if st.button("◀ Prev", disabled=prev_dis, use_container_width=True):
                if not prev_dis:
                    st.session_state.current_page_index -= 1
                    st.session_state.current_explanation = None
                    st.session_state.current_quiz = []
                    st.session_state.quiz_report = None
                    st.session_state.page_agent = None
                    st.rerun()
        with c2:
            next_dis = st.session_state.current_page_index >= len(st.session_state.page_roadmap) - 1
            if st.button("Next ▶", disabled=next_dis, use_container_width=True):
                if not next_dis:
                    st.session_state.current_page_index += 1
                    st.session_state.current_explanation = None
                    st.session_state.current_quiz = []
                    st.session_state.quiz_report = None
                    st.session_state.page_agent = None
                    st.rerun()
        with c3:
            st.session_state.viewer_height = st.slider(
                "Viewer height", 500, 1100, st.session_state.viewer_height, 50
            )

        # IMPORTANT: key on page index to force re-render when navigating
        pdf_viewer(
            input=st.session_state.pdf_bytes,
            key=f"pdf_viewer_{st.session_state.current_page_index}",
            height=int(st.session_state.viewer_height),
            pages_to_render=[current_page_num]
        )
        st.caption(f"Showing page {current_page_num} of {len(st.session_state.page_roadmap)}")
    else:
        st.info("Upload a research paper using the sidebar to see it here.")

# ---------- RIGHT: WORKSPACE (Tutor + RAG) ----------
with col_work:
    tutor_tab, rag_tab = st.tabs(["🧩 Page Tutor", "🧠 RAG — Ask anything from the uploaded document"])

    # ===== Page Tutor =====
    with tutor_tab:
        if not has_pdf:
            st.info("Upload a paper to enable the Page Tutor.")
        else:
            idx = st.session_state.current_page_index
            current_page_data = st.session_state.page_roadmap[idx]
            page_text = current_page_data['text']

            st.markdown(f"### Page {current_page_data['page']} / {len(st.session_state.page_roadmap)}")

            if not st.session_state.current_explanation:
                with st.spinner(f"Explaining page {current_page_data['page']}..."):
                    memory_context = "\n".join(st.session_state.memory_summaries[-3:])
                    expl_dict, new_summary = explainer.get_explanation(page_text, memory_context)
                    st.session_state.current_explanation = expl_dict
                    st.session_state.memory_summaries.append(new_summary)
                    st.session_state.page_agent = page_qa.PageQAAgent(
                        page_text, st.session_state.paper_id, current_page_data["page"]
                    )
                    logger.log_interaction(
                        st.session_state.paper_id, f"Page {current_page_data['page']}",
                        "explain", {"summary_len": len(expl_dict.get('summary', ''))},
                        context_len=len(page_text), model="gemini-2.5-flash-lite"
                    )

            if st.session_state.current_explanation:
                expl = st.session_state.current_explanation
                st.subheader("TL;DR")
                st.markdown(expl.get('summary', ''))
                st.subheader("Breakdown")
                st.markdown(expl.get('breakdown', ''))
                st.subheader("Key Terms")
                st.markdown(expl.get('terms', ''))

                tts_col1, tts_col2 = st.columns([1, 4])
                with tts_col1:
                    if st.button("🔊 Play", use_container_width=True):
                        audio = speak_text(f"{expl.get('summary','')} {expl.get('breakdown','')}")
                        if audio: st.audio(audio, format="audio/mp3")

            # --- Local summarizer (updated labels to Gemma-3-270M) ---
            st.markdown("### Local Summary (fine-tuned Gemma-3-270M)")
            sum_c1, sum_c2 = st.columns([1, 5])
            with sum_c1:
                do_sum = st.button("▶ Summarize (Gemma-3-270M)", key=f"btn_sum_{idx}", use_container_width=True)
            with sum_c2:
                st.caption("Runs locally using your fine-tuned Gemma-3-270M model (no API).")

            if do_sum:
                with st.spinner("Summarizing this page with fine-tuned Gemma-3-270M…"):
                    try:
                        summary_text = summarizer.summarize_page(page_text)
                    except Exception as e:
                        summary_text = f"⚠️ Summarizer error: {e}"
                    st.session_state[f"local_summary_{idx}"] = summary_text

            if st.session_state.get(f"local_summary_{idx}"):
                st.success(st.session_state[f"local_summary_{idx}"])
                # Optional: TTS for the local summary (re-uses your existing speak_text)
                play_c1, _ = st.columns([1, 4])
                with play_c1:
                    if st.button("🔊 Play summary", key=f"sum_tts_{idx}", use_container_width=True):
                        audio2 = speak_text(st.session_state[f"local_summary_{idx}"])
                        if audio2:
                            st.audio(audio2, format="audio/mp3")
            # --- END summarizer block ---

            st.markdown("### Ask about this page")
            page_q = st.text_input(
                "Your question (page-scoped)",
                key=f"pqa_input_{idx}",
                placeholder="Type your question and press Enter…"
            )

            if page_q:
                with st.spinner("Searching this page..."):
                    answer = st.session_state.page_agent.answer_question(page_q)
                    st.info(answer)
                    logger.log_interaction(
                        st.session_state.paper_id, f"Page {current_page_data['page']}",
                        "page_qa_query", {"query": page_q}, model="gemini-2.5-flash-lite"
                    )

            st.markdown("---")
            if st.button("✅ Quiz me on this page", use_container_width=True):
                st.session_state.current_screen = "quiz"
                st.rerun()

            # quiz/report flow (unchanged logic, larger space)
            if st.session_state.current_screen == "quiz":
                st.subheader(f"Quiz — Page {current_page_data['page']}")
                if not st.session_state.current_quiz:
                    with st.spinner("Generating quiz..."):
                        quiz_qas = st.session_state.qg_agent.generate_questions(current_page_data, 5)
                        st.session_state.current_quiz = quiz_qas
                        logger.log_interaction(
                            st.session_state.paper_id, f"Page {current_page_data['page']}",
                            "quiz_generated", {"count": len(quiz_qas)}, model="Mock-QG-Gemini"
                        )

                if st.session_state.current_quiz:
                    with st.form(key="quiz_form"):
                        user_answers = []
                        for i, qa in enumerate(st.session_state.current_quiz):
                            st.markdown(f"**Q{i+1}. {qa['question']}**")
                            user_ans = st.text_area(
                                "Your answer", key=f"q_{idx}_{i}",
                                placeholder="Type your answer…"
                            )
                            user_answers.append(user_ans)
                            st.write("")  # space

                        submitted = st.form_submit_button("Grade my answers")
                        if submitted:
                            with st.spinner("Evaluating..."):
                                report = evaluator.evaluate_quiz(st.session_state.current_quiz, user_answers)
                                st.session_state.quiz_report = report
                                st.session_state.current_screen = "report"
                                logger.log_interaction(
                                    st.session_state.paper_id, f"Page {current_page_data['page']}",
                                    "quiz_graded", report
                                )
                                st.rerun()
                else:
                    st.error("Could not generate quiz. Check quiz agent prompts.")

            if st.session_state.current_screen == "report":
                st.subheader("Quiz Report")
                report = st.session_state.quiz_report
                st.markdown(f"**{report['summary']}**")
                st.divider()
                for item in report['items']:
                    status_emoji = {"Correct": "✅", "Partial Credit": "⚠️", "Incorrect": "❌"}
                    st.markdown(f"{status_emoji.get(item['status'],'❌')} **{item['status']}**")
                    st.markdown(f"**Q:** {item['question']}")
                    st.write(f"**Your Answer:** {item['your_answer'] if item['your_answer'] else '—'}")
                    if item['status'] != "Correct":
                        st.info(f"{item['correction']}")
                    st.write("---")

                if st.button("Next page ▶", use_container_width=True):
                    next_idx = st.session_state.current_page_index + 1
                    if next_idx >= len(st.session_state.page_roadmap):
                        st.success("🎉 You reached the end. You can still use Global RAG Chat.")
                    else:
                        st.session_state.current_page_index = next_idx
                        st.session_state.current_explanation = None
                        st.session_state.current_quiz = []
                        st.session_state.quiz_report = None
                        st.session_state.page_agent = None
                        st.session_state.current_screen = "tutor"
                    st.rerun()

    # ===== Global RAG (bonus) =====
    with rag_tab:
        st.markdown('<div class="rag-banner">This is <b>RAG-based</b> global Q&A. Ask anything about your uploaded document. I will retrieve top-k snippets and answer grounded in them.</div>', unsafe_allow_html=True)

        left, right = st.columns([3, 2], gap="large")

        with left:
            rag_query = st.text_area(
                "RAG question (whole document)",
                key="rag_input",
                disabled=not has_pdf,
                height=80,
                placeholder="Type your question and click Ask…"
            )

            c1, c2 = st.columns([4, 1])
            with c1:
                ask = st.button("Ask with RAG", use_container_width=True, disabled=not has_pdf)
            with c2:
                st.session_state.rag_k = st.number_input("k", 1, 10, st.session_state.rag_k, 1, help="Number of snippets", disabled=not has_pdf)

            # transcript
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if ask and has_pdf and rag_query.strip():
                user_q = rag_query.strip()
                st.session_state.chat_messages.append({"role": "user", "content": user_q})
                with st.chat_message("user"):
                    st.markdown(user_q)

                with st.chat_message("assistant"):
                    with st.spinner("Retrieving evidence & synthesizing…"):
                        results = st.session_state.rag_agent.retrieve(user_q, k=st.session_state.rag_k)
                        st.session_state.rag_last_results = results
                        ctx = "\n\n".join([f"Source [{i+1}] (p.{res['page']}):\n{res['text']}" for i, res in enumerate(results)])
                        synthesis_prompt = f"""Use ONLY the context to answer. If not present, say so. Be concise.

CONTEXT:
```{ctx}```

QUESTION:
```{user_q}```

ANSWER:
"""
                        try:
                            model = genai.GenerativeModel("gemini-2.5-flash-lite")
                            response = model.generate_content(synthesis_prompt)
                            final_answer = response.text
                        except Exception as e:
                            final_answer = f"API error: {e}. Check your key."

                        st.markdown(final_answer)
                        st.session_state.chat_messages.append({"role": "assistant", "content": final_answer})
                        logger.log_interaction(
                            st.session_state.paper_id, "GLOBAL_RAG",
                            "rag_query_synthesized", {"query": user_q, "response": final_answer}
                        )

            if not has_pdf:
                st.caption("Upload a paper to enable Global RAG Q&A.")

            if st.button("Clear chat", use_container_width=True):
                st.session_state.chat_messages = []
                st.session_state.rag_last_results = []

        with right:
            st.subheader("Evidence (retrieved passages)")
            if not st.session_state.rag_last_results:
                st.caption("Ask a RAG question to see sources here.")
            else:
                with st.container():
                    st.markdown('<div class="evidence-box">', unsafe_allow_html=True)
                    for i, res in enumerate(st.session_state.rag_last_results, start=1):
                        st.markdown(f"**[{res['paper_id']}] {res.get('section_title','Page')} (p.{res['page']})**")
                        st.info(res['text'])
                    st.markdown('</div>', unsafe_allow_html=True)
