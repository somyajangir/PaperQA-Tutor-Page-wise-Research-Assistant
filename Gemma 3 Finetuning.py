
# Cell 0: Minimal, stable stack (no bitsandbytes, no triton)
!pip -q install "torch>=2.3.0" \
                "transformers>=4.43.0" \
                "trl==0.10.1" \
                "datasets==2.19.1" \
                "accelerate>=0.31.0" \
                "peft==0.11.1" \
                "sacrebleu==2.4.2" \
                "rouge-score==0.1.2" \
                "pandas==2.2.2" \
                "matplotlib==3.8.4" \
                sentencepiece

import os, torch, transformers, trl, datasets, peft, accelerate
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "trl": trl.__version__,
    "datasets": datasets.__version__,
    "peft": peft.__version__,
    "accelerate": accelerate.__version__,
})

# Cell 1: Imports, constants, login (⚠️ token via env — no hardcoding)

import time, json, random, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login
from rouge_score import rouge_scorer
import sacrebleu
import torch, os

# ✅ Added for env-based token loading (only change):
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if not hf_token:
    raise RuntimeError("HF token not found. Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN in your environment or .env")
os.environ["HF_TOKEN"] = hf_token
os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
login(token=hf_token, add_to_git_credential=False)
print("✅ HF login successful.")

# Repro
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Model & data
BASE_MODEL_ID = "google/gemma-3-270m"
DATASET_ID    = "xsum"

# Output dirs
RUN_DIR  = "outputs/gemma270m_lora_xsum"
BEST_DIR = f"{RUN_DIR}/best_checkpoint"
ART_DIR  = f"{RUN_DIR}/artifacts"
os.makedirs(ART_DIR, exist_ok=True)

# Subsets & lengths (fits in ~1–1.5 hr on A40)
TRAIN_N, VAL_N, TEST_N = 2000, 100, 100
MAX_LENGTH   = 1024     # prompt+target concatenated length
MAX_NEW_TOK  = 96
MIN_NEW_TOK  = 16

# Enable bf16 on A40
bf16_ok = torch.cuda.is_available()
print("bf16 enabled:", bf16_ok)

# Cell 2 (robust): Load dataset & format to 'text' = prompt + reference summary

from datasets import load_dataset

def try_load_xsum_then_cnndm():
    # Try canonical XSum first
    try:
        ds = load_dataset("EdinburghNLP/xsum")
        used = "EdinburghNLP/xsum"
        return used, ds
    except Exception as e_xsum:
        print("⚠️ XSum load failed, falling back to CNN/DailyMail 3.0.0. Error:", repr(e_xsum))
        try:
            ds = load_dataset("cnn_dailymail", "3.0.0")
            # Normalize to XSum-like keys: article->document, highlights->summary
            for split in ds.keys():
                ds[split] = ds[split].rename_column("article", "document")
                ds[split] = ds[split].rename_column("highlights", "summary")
            used = "cnn_dailymail/3.0.0"
            return used, ds
        except Exception as e_cnn:
            raise RuntimeError(
                "Failed to load both XSum and CNN/DailyMail.\n"
                f"XSum error: {repr(e_xsum)}\nCNN/DM error: {repr(e_cnn)}"
            )

# Load robustly
DATASET_ID, raw = try_load_xsum_then_cnndm()
print(f"✅ Using dataset: {DATASET_ID}")
print(raw)

def subsample(ds, n, seed=SEED):
    n = min(n, len(ds))
    return ds.shuffle(seed=seed).select(range(n))

# Build subsets
train_raw = subsample(raw["train"],      TRAIN_N, seed=SEED)
val_raw   = subsample(raw["validation"], VAL_N,   seed=SEED+1)
test_raw  = subsample(raw["test"],       TEST_N,  seed=SEED+2)

INSTR       = "Summarize the following content in 3–4 sentences."
RESP_PREFIX = "\n\nSummary: "

def to_text(ex):
    doc  = (ex["document"] or "").strip()
    summ = (ex["summary"]  or "").strip()
    return {"text": f"{INSTR}\n\nArticle:\n{doc}{RESP_PREFIX}{summ}", "target": summ}

train_ds = train_raw.map(to_text, remove_columns=train_raw.column_names)
val_ds   = val_raw.map(  to_text, remove_columns=val_raw.column_names)
test_ds  = test_raw.map( to_text, remove_columns=test_raw.column_names)

# Filter empties
train_ds = train_ds.filter(lambda e: len(e["text"].strip())>0 and len(e["target"].strip())>0)
val_ds   = val_ds.filter(  lambda e: len(e["text"].strip())>0 and len(e["target"].strip())>0)
test_ds  = test_ds.filter( lambda e: len(e["text"].strip())>0 and len(e["target"].strip())>0)

print("Subsets ->", len(train_ds), len(val_ds), len(test_ds))
print("Preview:", train_ds[0]["text"][:280].replace("\n"," "), "…")

# Cell 3: Tokenizer & model (no quantization; LoRA will be light)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=hf_token, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    token=hf_token,
    torch_dtype=torch.bfloat16 if bf16_ok else torch.float32,
    device_map="auto",
    attn_implementation="eager",  # safer across environments
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

print("✅ Base model ready.")

# Cell 4: LoRA (attention + MLP) and SFTConfig kept minimal/stable

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

training_config = SFTConfig(
    output_dir=RUN_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # effective 16
    learning_rate=2e-4,
    logging_steps=25,
    save_strategy="epoch",
    save_total_limit=2,
    optim="adamw_torch",             # no bitsandbytes
    dataset_text_field="text",           # applies to tokenization in SFT pipeline
    report_to="none",
)

print("✅ LoRA & SFT config ready.")

# Cell 5: Trainer & training loop

trainer = SFTTrainer(
    model=model,
 # avoids tokenizer deprecation
    train_dataset=train_ds,
    eval_dataset=val_ds,
    peft_config=lora_config,
    args=training_config,
)

print("🚀 Starting fine-tuning…")
t0 = time.time()
trainer.train()
print(f"⏱️ Training finished in {(time.time()-t0)/60:.1f} min")

# Save best/final adapters + tokenizer
print("💾 Saving adapters…")
trainer.save_model(BEST_DIR)
tokenizer.save_pretrained(BEST_DIR)
print("✅ Adapters saved to:", BEST_DIR)

# Cell 6: Plot training loss curve (clean)

logs = trainer.state.log_history
loss_logs = [l for l in logs if "loss" in l]
df = pd.DataFrame(loss_logs)
csv_path = f"{ART_DIR}/training_metrics.csv"
df.to_csv(csv_path, index=False)
print("📈 Metrics CSV:", csv_path)

plt.figure(figsize=(8,4.5))
plt.plot(df["step"], df["loss"], marker="o", linewidth=1.5)
plt.title("Training Loss (Gemma-3-270M LoRA on XSum)")
plt.xlabel("Step"); plt.ylabel("Loss"); plt.grid(True, alpha=0.3)
loss_png = f"{ART_DIR}/loss_curve.png"
plt.tight_layout(); plt.savefig(loss_png, dpi=150); plt.close()
print("🖼️ Saved:", loss_png)

# Cell 7: Manual beam evaluation on subsets (fast & realistic)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_eval = trainer.model.eval().to(device)

INSTR = "Summarize the following article in 1–3 sentences."
RESP_PREFIX = "\n\nSummary: "

def prompt_from_full(full_text: str) -> str:
    # Keep everything up to RESP_PREFIX + the prefix itself for a clean continuation
    head = full_text.split(RESP_PREFIX)[0] + RESP_PREFIX
    return head

def generate_batch(prompts, max_new=MAX_NEW_TOK, min_new=MIN_NEW_TOK, beams=4):
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        out = model_eval.generate(
            **enc,
            num_beams=beams,
            max_new_tokens=max_new,
            min_new_tokens=min_new,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            early_stopping=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    dec = tokenizer.batch_decode(out, skip_special_tokens=True)
    # Remove any prompt copy at the start
    clean = []
    for d, p in zip(dec, prompts):
        clean.append(d[len(p):].strip() if d.startswith(p) else d.strip())
    return clean

def eval_subset(ds, n=300, bs=8):
    n = min(n, len(ds))
    rows = [ds[i] for i in range(n)]
    prompts = [prompt_from_full(r["text"]) for r in rows]
    refs    = [r["target"].strip() for r in rows]

    preds=[]
    for i in range(0, n, bs):
        preds.extend(generate_batch(prompts[i:i+bs]))

    # BLEU
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    # ROUGE
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeLsum"], use_stemmer=True)
    r1=r2=rl=0.0
    for ref, pred in zip(refs, preds):
        s = scorer.score(ref, pred)
        r1 += s["rouge1"].fmeasure; r2 += s["rouge2"].fmeasure; rl += s["rougeLsum"].fmeasure
    n = max(1, len(refs))
    r1, r2, rl = 100*r1/n, 100*r2/n, 100*rl/n
    return {"BLEU": bleu, "ROUGE1_F": r1, "ROUGE2_F": r2, "ROUGEL_F": rl, "samples": len(refs)}

val_metrics  = eval_subset(val_ds,  n=300, bs=8)
test_metrics = eval_subset(test_ds, n=300, bs=8)
print("VAL  (beam=4):", val_metrics)
print("TEST (beam=4):", test_metrics)

# Save & small bar
with open(f"{ART_DIR}/val_metrics.json","w") as f: json.dump(val_metrics, f, indent=2)
with open(f"{ART_DIR}/test_metrics.json","w") as f: json.dump(test_metrics, f, indent=2)

labels = ["BLEU","ROUGE1_F","ROUGE2_F","ROUGEL_F"]
vals   = [val_metrics[k] for k in labels]
plt.figure(figsize=(6.4,4))
plt.bar(labels, vals)
plt.title("Validation Metrics (beam=4, min_new=16)")
plt.ylabel("Score"); plt.tight_layout()
val_png = f"{ART_DIR}/val_metrics_beam4.png"
plt.savefig(val_png, dpi=150); plt.close()
print("🖼️ Saved:", val_png)

# Cell 8: Qualitative samples for the report

def prompt_only(row):
    return row["text"].split(RESP_PREFIX)[0] + RESP_PREFIX

N = min(20, len(val_ds))
rows   = [val_ds[i] for i in range(N)]
prompts= [prompt_only(r) for r in rows]
refs   = [r["target"] for r in rows]
preds  = generate_batch(prompts, max_new=MAX_NEW_TOK, min_new=MIN_NEW_TOK, beams=4)

pred_df = pd.DataFrame({
    "input_prompt_preview": [p.replace("\n"," ")[:260] + ("..." if len(p)>260 else "") for p in prompts],
    "reference": refs,
    "prediction": preds
})
pred_csv = f"{ART_DIR}/sample_predictions.csv"
pred_df.to_csv(pred_csv, index=False)
print("🧪 Saved:", pred_csv)

# Cell 9: Merge adapters to a single-folder model (easiest inference)

# Load a fresh FP32 base on CPU then merge
base_f32 = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, torch_dtype=torch.float32, device_map="cpu", token=hf_token
)
PeftModel  # (ensures import used)
peft_model = PeftModel.from_pretrained(base_f32, BEST_DIR)
merged = peft_model.merge_and_unload()
merged.config.use_cache = True

merge_dir = f"{RUN_DIR}/merged_full_model"
os.makedirs(merge_dir, exist_ok=True)
merged.save_pretrained(merge_dir, safe_serialization=True)
tokenizer.save_pretrained(merge_dir)
print("✅ Merged model saved to:", merge_dir)

# Cell 10: Save run summary & zip all artifacts for download

summary = {
    "base_model": BASE_MODEL_ID,
    "dataset": {"id": DATASET_ID, "train_n": len(train_ds), "val_n": len(val_ds), "test_n": len(test_ds)},
    "lora": {"r":16, "alpha":32, "dropout":0.05, "targets":["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]},
    "train": {"epochs":1, "per_device_bsz":1, "grad_accum":16, "lr":2e-4, "optim":"adamw_torch"},
    "seq": {"max_length": MAX_LENGTH, "max_new_tokens": MAX_NEW_TOK, "min_new_tokens": MIN_NEW_TOK},
    "val_metrics":  val_metrics,
    "test_metrics": test_metrics,
    "artifacts": {
        "metrics_csv": f"{ART_DIR}/training_metrics.csv",
        "loss_curve_png": f"{ART_DIR}/loss_curve.png",
        "val_metrics_beam4_png": f"{ART_DIR}/val_metrics_beam4.png",
        "sample_predictions_csv": f"{ART_DIR}/sample_predictions.csv",
        "best_adapters_dir": BEST_DIR,
        "merged_model_dir": f"{RUN_DIR}/merged_full_model",
    }
}
with open(f"{ART_DIR}/run_summary.json","w") as f: json.dump(summary, f, indent=2)
with open(f"{ART_DIR}/run_summary.txt","w") as f: f.write(json.dumps(summary, indent=2))
print("💾 Saved run summary.")

import shutil
zip_path = shutil.make_archive("gemma270m_lora_xsum_artifacts", "zip", root_dir=RUN_DIR)
print("📦 ZIP:", zip_path)
