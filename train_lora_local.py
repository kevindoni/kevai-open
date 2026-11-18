# ================================================================
# KevAI-OPEN TRAINER — LoRA CPU di GitHub Actions (gratis, 24/7)
# - Auto-resume dari checkpoint (actions/cache) — tahan mati/job-6-jam
# - Dataset: model-latihan.jsonl (di repo)
# - Base: Qwen/Qwen2.5-1.5B-Instruct (publik, diunduh runner)
# - Selesai: work/DONE ditulis -> chain workflow_run berhenti anggun
# ================================================================
import json, os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

BASE = os.environ.get("KEVAI_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DATA = "model-latihan.jsonl"
WORK = os.environ.get("KEVAI_WORK", "work")
ADAPTER_DIR = os.path.join(WORK, "adapter")
MERGED_DIR = os.path.join(WORK, "merged")
EPOCHS = 3

# [1/5] dataset
print("[1/5] Memuat dataset:", DATA)
data = []
with open(DATA, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line)["messages"])
print("     entri:", len(data))

def to_text(msgs):
    t = ""
    for m in msgs:
        if m["role"] == "system":
            t += "<|im_start|>system\n" + m["content"] + "<|im_end|>\n"
        elif m["role"] == "user":
            t += "<|im_start|>user\n" + m["content"] + "<|im_end|>\n"
        else:
            t += "<|im_start|>assistant\n" + m["content"] + "<|im_end|>\n"
    return t + "<|im_start|>assistant\n"

texts = [to_text(m) for m in data]

# [2/5] tokenizer + base model (publik, diunduh runner)
print("[2/5] Memuat tokenizer + base model...")
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float32,
    device_map="cpu",
)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

enc = tok(texts, truncation=True, max_length=512, padding=False)
ds = Dataset.from_dict({"input_ids": enc["input_ids"]})

def add_labels(batch):
    batch["labels"] = [
        [(tid if tid != tok.pad_token_id else -100) for tid in seq]
        for seq in batch["input_ids"]
    ]
    return batch

ds = ds.map(add_labels, batched=True)

# [3/5] LoRA adapter
print("[3/5] Memasang LoRA adapter (hanya ini yang dilatih)...")
lcfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()
model.enable_input_require_grads()
model.gradient_checkpointing_enable()

args = TrainingArguments(
    output_dir=ADAPTER_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=30,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=60,
    save_total_limit=2,
    optim="adamw_torch",
    report_to=[],
    seed=42,
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=DataCollatorForSeq2Seq(tok, padding=True),
)

# [4/5] training — auto-resume dari checkpoint (tahan mati/job-6-jam)
print("[4/5] TRAINING — chaining workflow_run melanjutkan otomatis...")
ckpt = None
if os.path.isdir(ADAPTER_DIR):
    cands = [d for d in os.listdir(ADAPTER_DIR) if d.startswith("checkpoint-")]
    if cands:
        ckpt = os.path.join(ADAPTER_DIR, max(cands, key=lambda x: int(x.split("-")[-1])))
        print(f"[4/5] Melanjutkan dari checkpoint: {ckpt}")
trainer.train(resume_from_checkpoint=ckpt)

# [5/5] merge + simpan + penanda DONE
print("[5/5] Merge adapter + simpan model KevAI-Edge...")
model = model.merge_and_unload()
model = model.to(torch.float16)
os.makedirs(MERGED_DIR, exist_ok=True)
model.save_pretrained(MERGED_DIR, safe_serialization=True)
tok.save_pretrained(MERGED_DIR, safe_serialization=True)

with open(os.path.join(WORK, "DONE"), "w", encoding="utf-8") as f:
    f.write("done")

print("SELESAI — model KevAI-Edge siap dikonversi ke GGUF")