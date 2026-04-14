# train_transformer_simple.py

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

MODEL_CKPT = "xlm-roberta-base"
TARGET_COLS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
MAX_LEN = 128  # shorter max length speeds up training

# Load and prepare data
df = pd.read_csv("data/cleaned_data.csv")
for c in TARGET_COLS:
    df[c] = df[c].astype(int)
df['labels'] = df[TARGET_COLS].values.tolist()

train_df, val_df = train_test_split(df[['comment_text', 'labels']], test_size=0.15, random_state=42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)

def preprocess(examples):
    # tokenize with truncation and padding to max length
    encodings = tokenizer(examples['comment_text'], truncation=True, padding='max_length', max_length=MAX_LEN)
    encodings['labels'] = examples['labels']  # labels as list of ints (0/1)
    return encodings

train_ds = Dataset.from_pandas(train_df).map(preprocess, batched=True, remove_columns=['comment_text'])
val_ds = Dataset.from_pandas(val_df).map(preprocess, batched=True, remove_columns=['comment_text'])

# Load model for multi-label classification
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CKPT,
    problem_type="multi_label_classification",
    num_labels=len(TARGET_COLS)
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))  # apply sigmoid to logits
    preds = (probs >= 0.5).astype(int)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    return {"f1_micro": f1_micro, "f1_macro": f1_macro}

training_args = TrainingArguments(
    output_dir="models/transformer_trollguard",
    per_device_train_batch_size=16,  # increase if you have GPU memory
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    num_train_epochs=2,
    fp16=False,
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",
    logging_steps=50,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

if __name__ == "__main__":
    print("Starting training...")
    trainer.train()
    print("Training completed. Saving model and tokenizer...")
    trainer.save_model("models/transformer_trollguard")
    tokenizer.save_pretrained("models/transformer_trollguard")
