import pandas as pd
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- Configuration ---
MODEL_PATH = "./final_classifier_model"
TOKENIZER_PATH = "./final_classifier_model"
MLB_PATH = "./final_classifier_model/mlb.pkl"
THRESHOLDS_PATH = "optimal_thresholds.json"
FULL_DATA_PATH = "full_card_database.csv"
OUTPUT_PATH = "classified_card_database.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

print(f"--- Using device: {DEVICE} ---")

# --- 1. Load All Necessary Components ---
print("Loading model, tokenizer, and supporting files...")
try:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    mlb = torch.load(MLB_PATH, weights_only=False)
    with open(THRESHOLDS_PATH, 'r') as f:
        optimal_thresholds = json.load(f)
    df = pd.read_csv(FULL_DATA_PATH)
except FileNotFoundError as e:
    print(f"Error loading files: {e}. Ensure all necessary files are present.")
    exit()

official_labels = list(mlb.classes_)
print("All components loaded successfully.")

# --- 2. Prepare Data for Inference ---
def create_structured_input(row: pd.Series) -> str:
    type_line = row.get('type_line', '') or ''
    power_toughness = f"P/T: {row.get('power', '')}/{row.get('toughness', '')}" if 'Creature' in type_line else ""
    return f"Name: {row.get('name', '')} Cost: {row.get('mana_cost', '')} Type: {type_line} Text: {row.get('oracle_text', '')} {power_toughness}"

print(f"Preparing {len(df)} cards for classification...")
df['structured_text'] = df.apply(create_structured_input, axis=1)

class BulkInferenceDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        return {key: val.squeeze(0) for key, val in encoding.items()}

dataset = BulkInferenceDataset(df['structured_text'].tolist(), tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)


# --- 3. Run Bulk Classification ---
all_predictions = []
print("Starting bulk classification...")
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Classifying Batches"):
        inputs = {key: val.to(DEVICE) for key, val in batch.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        all_predictions.append(logits.cpu())

all_logits = torch.cat(all_predictions, dim=0)
sigmoid = torch.nn.Sigmoid()
all_probabilities = sigmoid(all_logits).numpy()
print("Classification complete.")


# --- 4. Apply Optimal Thresholds and Format Output ---
print("Applying optimal thresholds and formatting output...")
# CORRECTED LINE: Initialize as a NumPy array instead of a PyTorch tensor.
final_predictions_binary = np.zeros_like(all_probabilities, dtype=int)
for i, label in enumerate(official_labels):
    threshold = optimal_thresholds.get(label, 0.5)
    final_predictions_binary[:, i] = (all_probabilities[:, i] >= threshold).astype(int)

predicted_tags_list = mlb.inverse_transform(final_predictions_binary)


# --- 5. Save Final Results ---
output_df = pd.DataFrame({
    'uuid': df['uuid'],
    'name': df['name'],
    'predicted_tags': predicted_tags_list
})

output_df['predicted_tags'] = output_df['predicted_tags'].apply(list)

print(f"Saving final classified data to '{OUTPUT_PATH}'...")
output_df.to_csv(OUTPUT_PATH, index=False)

print("\n--- Bulk Classification Complete ---")
print(f"The file '{OUTPUT_PATH}' has been created with the predicted tags for all cards.")