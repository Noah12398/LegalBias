# Install dependencies if not already installed
!pip install transformers datasets accelerate evaluate

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from transformers import AdamW, get_scheduler
from datasets import load_dataset
from tqdm import tqdm

# --------------------------
# 1. Load Pseudo-Labeled Dataset
# --------------------------
dataset = load_dataset("json", data_files={"train": "silver_train.json", "test": "silver_test.json"})

# Concatenate passage + context for input
def preprocess_classification(example):
    text = example["highlighted_passage"] + " " + example["case_context"]
    return {"input_text": text, "label": example["teacher_label"]}

def preprocess_explanation(example):
    text = example["highlighted_passage"] + " " + example["case_context"]
    return {"input_text": text, "target_text": example["teacher_explanation"]}

dataset_cls = dataset.map(preprocess_classification)
dataset_exp = dataset.map(preprocess_explanation)

# --------------------------
# 2. Tokenizers
# --------------------------
# Student model for classification
cls_model_name = "distilbert-base-multilingual-cased"
tokenizer_cls = AutoTokenizer.from_pretrained(cls_model_name)
model_cls = AutoModelForSequenceClassification.from_pretrained(cls_model_name, num_labels=2)

# Student model for explanation generation
exp_model_name = "facebook/mbart-large-50"
tokenizer_exp = AutoTokenizer.from_pretrained(exp_model_name)
model_exp = AutoModelForSeq2SeqLM.from_pretrained(exp_model_name)

# --------------------------
# 3. Data Collators
# --------------------------
def collate_fn_cls(batch):
    encodings = tokenizer_cls([x["input_text"] for x in batch], truncation=True, padding=True, return_tensors="pt")
    labels = torch.tensor([x["label"] for x in batch])
    encodings["labels"] = labels
    return encodings

def collate_fn_exp(batch):
    inputs = tokenizer_exp([x["input_text"] for x in batch], truncation=True, padding=True, return_tensors="pt")
    targets = tokenizer_exp([x["target_text"] for x in batch], truncation=True, padding=True, return_tensors="pt")
    labels = targets["input_ids"].clone()
    labels[labels == tokenizer_exp.pad_token_id] = -100  # ignore padding
    inputs["labels"] = labels
    return inputs

# --------------------------
# 4. Dataloaders
# --------------------------
train_loader_cls = DataLoader(dataset_cls["train"], batch_size=16, shuffle=True, collate_fn=collate_fn_cls)
train_loader_exp = DataLoader(dataset_exp["train"], batch_size=8, shuffle=True, collate_fn=collate_fn_exp)

# --------------------------
# 5. Optimizers & Scheduler
# --------------------------
optimizer_cls = AdamW(model_cls.parameters(), lr=5e-5)
optimizer_exp = AdamW(model_exp.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps_cls = num_epochs * len(train_loader_cls)
num_training_steps_exp = num_epochs * len(train_loader_exp)

scheduler_cls = get_scheduler("linear", optimizer=optimizer_cls, num_warmup_steps=0, num_training_steps=num_training_steps_cls)
scheduler_exp = get_scheduler("linear", optimizer=optimizer_exp, num_warmup_steps=0, num_training_steps=num_training_steps_exp)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cls.to(device)
model_exp.to(device)

# --------------------------
# 6. Training Loops
# --------------------------
# Classification Training
for epoch in range(num_epochs):
    model_cls.train()
    loop = tqdm(train_loader_cls, desc=f"Epoch {epoch+1} CLS")
    for batch in loop:
        batch = {k:v.to(device) for k,v in batch.items()}
        outputs = model_cls(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer_cls.step()
        scheduler_cls.step()
        optimizer_cls.zero_grad()
        loop.set_postfix(loss=loss.item())

# Explanation Generation Training
for epoch in range(num_epochs):
    model_exp.train()
    loop = tqdm(train_loader_exp, desc=f"Epoch {epoch+1} EXP")
    for batch in loop:
        batch = {k:v.to(device) for k,v in batch.items()}
        outputs = model_exp(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer_exp.step()
        scheduler_exp.step()
        optimizer_exp.zero_grad()
        loop.set_postfix(loss=loss.item())

# --------------------------
# 7. Inference Example
# --------------------------
def predict(text):
    # Classification
    enc = tokenizer_cls(text, truncation=True, padding=True, return_tensors="pt").to(device)
    logits = model_cls(**enc).logits
    pred_label = torch.argmax(logits, dim=-1).item()
    
    # Explanation
    enc_exp = tokenizer_exp(text, return_tensors="pt").to(device)
    generated_ids = model_exp.generate(**enc_exp, max_length=50)
    explanation = tokenizer_exp.decode(generated_ids[0], skip_special_tokens=True)
    
    return pred_label, explanation

sample_text = "Highlighted passage here. Case context here."
label, explanation = predict(sample_text)
print("Bias Label:", label, "Explanation:", explanation)
