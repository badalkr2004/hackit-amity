import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer

# load dataset
df = pd.read_csv("./train.csv", encoding="cp1252")
# print(df.head())

# necessary columns

df = df[["text", "sentiment"]]

label_map = {"negative": 0, "neutral": 1, "positive": 2}

df["label"] = df["sentiment"].map(label_map)

# drop missing values
df = df.dropna()

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)


model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# tokenize function
def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=128
    )


# Apply tokenization

dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.train_test_split(test_size=0.2)


# Load DistilBERT with classification head
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

trainer.train()

model.save_pretrained("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")

print("Training Complete! Model Saved.")
