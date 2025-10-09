import pandas as pd
import torch
import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from tqdm import tqdm
import ast

# --- Configuration ---
MODEL_PATH = "./final_classifier_model"
TOKENIZER_PATH = "./final_classifier_model"
MLB_PATH = "./final_classifier_model/mlb.pkl"
DATA_PATH = "labeled_subset_clean.csv"
THRESHOLDS_OUTPUT_PATH = "optimal_thresholds.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# --- 1. Load Model, Tokenizer, and Data ---
print("Loading model, tokenizer, and data...")
try:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    mlb = torch.load(MLB_PATH, weights_only=False)
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError as e:
    print(f"Error loading files: {e}. Ensure the model was trained and files are in the correct paths.")
    exit()

from sklearn.model_selection import train_test_split
_, val_df = train_test_split(df, test_size=0.2, random_state=42)

official_labels = list(mlb.classes_)
print(f"Loaded {len(official_labels)} labels.")

# --- 2. Generate Predictions on Validation Set ---
class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        
        # CORRECTED SECTION:
        # Squeeze each tensor to remove the leading '1' dimension.
        # This ensures the batch has the correct 2D shape [batch_size, seq_length].
        return {key: val.squeeze(0) for key, val in encoding.items()}

    def __len__(self):
        return len(self.texts)

def create_structured_input(row: pd.Series) -> str:
    power_toughness = f"P/T: {row.get('power', '')}/{row.get('toughness', '')}" if 'Creature' in row.get('type_line', '') else ""
    return f"Name: {row.get('name', '')} Cost: {row.get('mana_cost', '')} Type: {row.get('type_line', '')} Text: {row.get('oracle_text', '')} {power_toughness}"

def parse_card_tags(tags_str):
    try:
        tags = ast.literal_eval(str(tags_str))
        return tags if isinstance(tags, list) else []
    except (ValueError, SyntaxError):
        return []
val_df['card_tags_list'] = val_df['card_tags'].apply(parse_card_tags)

val_df['structured_text'] = val_df.apply(create_structured_input, axis=1)
val_texts = val_df['structured_text'].tolist()
val_dataset = InferenceDataset(val_texts, tokenizer)

trainer = Trainer(model=model)
print("Generating predictions for the validation set...")
predictions = trainer.predict(val_dataset)

logits = predictions.predictions
sigmoid = torch.nn.Sigmoid()
probabilities = sigmoid(torch.Tensor(logits)).numpy()

# --- 3. Find Optimal Thresholds ---
print("\nFinding optimal thresholds for each label...")
y_true = mlb.transform(val_df['card_tags_list'])
optimal_thresholds = {}

for i, label in enumerate(tqdm(official_labels, desc="Optimizing Thresholds")):
    best_threshold = 0.5
    best_f1 = 0.0
    search_space = np.arange(0.01, 1.0, 0.01)
    for threshold in search_space:
        y_pred = (probabilities[:, i] >= threshold).astype(int)
        f1 = f1_score(y_true[:, i], y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    optimal_thresholds[label] = best_threshold

with open(THRESHOLDS_OUTPUT_PATH, 'w') as f:
    json.dump(optimal_thresholds, f, indent=4)
print(f"Optimal thresholds saved to '{THRESHOLDS_OUTPUT_PATH}'.")

# --- 4. Evaluate with New Thresholds ---
print("\nEvaluating performance with new optimal thresholds...")
final_predictions = np.zeros_like(probabilities)
for i, label in enumerate(official_labels):
    threshold = optimal_thresholds[label]
    final_predictions[:, i] = (probabilities[:, i] >= threshold).astype(int)

final_f1 = f1_score(y_true, final_predictions, average='micro')
final_precision = precision_score(y_true, final_predictions, average='micro', zero_division=0)
final_recall = recall_score(y_true, final_predictions, average='micro', zero_division=0)

print("\n--- Final Performance Report ---")
print(f"F1 Score (Micro):    {final_f1:.4f}")
print(f"Precision (Micro):   {final_precision:.4f}")
print(f"Recall (Micro):      {final_recall:.4f}")
print("---------------------------------")