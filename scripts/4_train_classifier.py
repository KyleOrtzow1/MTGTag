import pandas as pd
import torch
import json
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import ast
import numpy as np

# --- Configuration ---
# CORRECTED: The base model is now the domain-adapted model you just created.
MODEL_NAME = "./domain_adapted_model"
FINAL_MODEL_OUTPUT_PATH = "./final_classifier_model"
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"--- Using device: {DEVICE} ---")

# --- 1. Load and Preprocess Data ---
try:
    df = pd.read_csv("labeled_subset_clean.csv")
    with open("tag_definitions.json", 'r') as f:
        tag_data = json.load(f)
    official_labels = sorted(list(tag_data.keys()))
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure labeled_subset_clean.csv and tag_definitions.json are present.")
    exit()

# --- Label Preparation ---
def parse_card_tags(tags_str):
    try:
        tags = ast.literal_eval(str(tags_str))
        return tags if isinstance(tags, list) else []
    except (ValueError, SyntaxError):
        return []
df['card_tags_list'] = df['card_tags'].apply(parse_card_tags)
mlb = MultiLabelBinarizer(classes=official_labels)
encoded_labels = mlb.fit_transform(df['card_tags_list'])
labels_df = pd.DataFrame(encoded_labels, columns=mlb.classes_, index=df.index)
df = pd.concat([df, labels_df], axis=1)

for col in ['oracle_text', 'type_line', 'name', 'mana_cost', 'power', 'toughness']:
    if col in df.columns:
        df[col].fillna('', inplace=True)

def create_structured_input(row: pd.Series) -> str:
    power_toughness = f"P/T: {row.get('power', '')}/{row.get('toughness', '')}" if 'Creature' in row.get('type_line', '') else ""
    return f"Name: {row.get('name', '')} Cost: {row.get('mana_cost', '')} Type: {row.get('type_line', '')} Text: {row.get('oracle_text', '')} {power_toughness}"
df['structured_text'] = df.apply(create_structured_input, axis=1)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")


# --- 2. Calculate Class Weights ---
print("Calculating weights for class imbalance...")
pos_counts = train_df[official_labels].sum()
neg_counts = len(train_df) - pos_counts
pos_weight = torch.tensor(neg_counts.values / pos_counts.values, dtype=torch.float).to(DEVICE)
print("Class weights calculated.")


# --- 3. Tokenization and Dataset Creation ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
class CardDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, labels_list):
        self.df = df
        self.tokenizer = tokenizer
        self.labels_list = labels_list
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['structured_text']
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        labels = torch.tensor(row[self.labels_list].values.astype(float), dtype=torch.float)
        return {'input_ids': inputs['input_ids'].squeeze(0), 'attention_mask': inputs['attention_mask'].squeeze(0), 'labels': labels}
    def __len__(self):
        return len(self.df)

train_dataset = CardDataset(train_df, tokenizer, official_labels)
val_dataset = CardDataset(val_df, tokenizer, official_labels)


# --- 4. Custom Trainer with Weighted Loss ---
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


# --- 5. Model and Metrics ---
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(official_labels), problem_type="multi_label_classification").to(DEVICE)

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(preds))
    y_pred = (probs > 0.5).int()
    y_true = p.label_ids
    f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    return {'f1': f1, 'precision': precision, 'recall': recall}


# --- 6. Training ---
training_args = TrainingArguments(
    output_dir='./results_final',
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none"
)

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("\nStarting FINAL fine-tuning on domain-adapted model...")
trainer.train()


# --- 7. Save the Final Model ---
print(f"\nSaving the final, high-performance model to '{FINAL_MODEL_OUTPUT_PATH}'")
trainer.save_model(FINAL_MODEL_OUTPUT_PATH)
tokenizer.save_pretrained(FINAL_MODEL_OUTPUT_PATH)
torch.save(mlb, f"{FINAL_MODEL_OUTPUT_PATH}/mlb.pkl")

print("\n--- Project Complete ---")
print("You have successfully trained the final classification model.")
print("Run 'classify_and_tune.py' one last time (pointing to the new model path) to see the final performance.")