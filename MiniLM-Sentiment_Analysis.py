import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments, Trainer, DataCollatorWithPadding
import pandas as pd
from datasets import Dataset
import html
import re
import evaluate
import numpy as np

#model
model_id="sentence-transformers/all-MiniLM-L12-v2"
tokenizer=AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3)


#load dataset
full_dataset = pd.read_csv(
    "data/training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    names=["label", "id", "date", "flag", "user", "text"]
)
full_dataset=full_dataset[["text","label"]]
print(len(full_dataset))
print(full_dataset.info())

label_map = {0: 0, 2: 1, 4: 2}

unexpected_labels = set(full_dataset["label"].unique()) - set(label_map)
if unexpected_labels:
    raise ValueError(f"Unexpected labels in dataset: {unexpected_labels}")

full_dataset["label"] = full_dataset["label"].map(label_map).astype("int64")

def clean_tweet(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r'https?://\S+|www\.\S+', 'http', text)
    text = re.sub(r'@\w+', '@user', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#apply cleaning
full_dataset['text'] = full_dataset['text'].apply(clean_tweet)
print(full_dataset["text"].head())

#convert Dataset to Hugging Face Dataset
dataset=Dataset.from_pandas(full_dataset)

#split dataset
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
print(len(split_dataset["train"]))
print(len(split_dataset["test"]))

print(model.config)

max_length=256
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True,max_length=max_length)
tokenized_train_data = split_dataset["train"].map(tokenize_function, batched=True)
tokenized_test_data = split_dataset["test"].map(tokenize_function, batched=True)


accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels,average="weighted")["f1"]
    return {"accuracy": acc, "f1": f1}

#Train/test model
training_args = TrainingArguments(
    output_dir="./model/minilm_sentiment140_output",
    num_train_epochs=2,
    per_device_train_batch_size=128,  
    per_device_eval_batch_size=128,
    learning_rate=3e-5,
    warmup_steps=1425,
    weight_decay=0.01,
    bf16=torch.cuda.is_available(),
    logging_steps=500,
    eval_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_test_data,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    processing_class=tokenizer
)

trainer.train()
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)
#Save the final fine-tuned model and tokenizer
model_save_path = "./models/minilm_finetuned/final"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model and tokenizer saved successfully to {model_save_path}")