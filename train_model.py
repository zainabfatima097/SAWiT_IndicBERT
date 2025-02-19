import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" 
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrices import accuracy_score, precision_score, recall_score, f1_score
import json
 
# 1. Load Dataset
dataset = load_dataset("json", data_files={"train": "train.jsonl", "test": "test.jsonl"})

# 2. Create Label Mapping Dynamically (Use BOTH Train and Test Sets)
unique_labels_train = set(dataset["train"]["Responses"])
unique_labels_test = set(dataset["test"]["Responses"])
all_unique_labels = unique_labels_train.union(unique_labels_test)

label_mapping = {label: idx for idx, label in enumerate(all_unique_labels)}
print("Label mapping:", label_mapping)

# Save the label mapping (essential for consistency)
with open("label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, indent=4, ensure_ascii=False)

# 3. Remap Labels Function (with robust error handling)
def remap_labels(examples):
    labels = []
    for label in examples["Responses"]:
        if label not in label_mapping:
            raise KeyError(f"Label '{label}' not found in label_mapping: {label}")
        labels.append(label_mapping[label])
    return {"label": labels}

# 4. Apply Label Remapping
dataset = dataset.map(remap_labels, batched=True)

# 5. Load Pre-trained Model and Tokenizer
model_name = "bert-base-multilingual-cased"  # Or a suitable multilingual model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(all_unique_labels))

# 6. Tokenize Data
def tokenize_function(examples):
    return tokenizer(examples["Responses"], truncation=True)  # Add padding if needed

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 7. Define Training Metrics Function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=1)  # Handle potential zero division
    recall = recall_score(labels, predictions, average='weighted', zero_division=1)      # Handle potential zero division
    f1 = f1_score(labels, predictions, average='weighted', zero_division=1)          # Handle potential zero division
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# 8. Set Training Arguments (adjust as needed)
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    num_train_epochs=3,  # Adjust number of training epochs
    per_device_train_batch_size=16,  # Adjust batch size
    per_device_eval_batch_size=16,   # Adjust batch size
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# 9. Create Trainer and Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# 10. Save Fine-Tuned Model
trainer.save_model("./fine_tuned_model")

print("Training complete!")