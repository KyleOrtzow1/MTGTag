import pandas as pd
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# --- Configuration ---
BASE_MODEL = "roberta-base"
# NOTE: This script should run on your FULL card database, not just the labeled subset.
# Make sure you have exported this file.
FULL_DATA_PATH = "full_card_database.csv"
ADAPTED_MODEL_OUTPUT_PATH = "./domain_adapted_model"

MAX_LENGTH = 256 # Shorter max length is fine for this task
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
EPOCHS = 1 # One epoch is often sufficient for domain adaptation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Using device: {DEVICE} ---")
print(f"--- Transformers Version: {transformers.__version__} ---")


# --- 1. Load and Prepare Data ---
try:
    df = pd.read_csv(FULL_DATA_PATH)
except FileNotFoundError:
    print(f"Error: '{FULL_DATA_PATH}' not found.")
    print("Please run the 'export_all_cards.py' script from your 'export_data' directory first.")
    exit()

# Re-use the structured input format
def create_structured_input(row: pd.Series) -> str:
    # Ensure NaN values are handled gracefully
    type_line = row.get('type_line', '') or ''
    power_toughness = f"P/T: {row.get('power', '')}/{row.get('toughness', '')}" if 'Creature' in type_line else ""
    return f"Name: {row.get('name', '')} Cost: {row.get('mana_cost', '')} Type: {type_line} Text: {row.get('oracle_text', '')} {power_toughness}"

print(f"Processing {len(df)} cards for domain adaptation...")
df['structured_text'] = df.apply(create_structured_input, axis=1)
text_corpus = df['structured_text'].tolist()


# --- 2. Tokenization ---
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, texts, max_length):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        encoding = self.tokenizer(
            self.texts[i],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_special_tokens_mask=True
        )
        return {key: torch.tensor(val) for key, val in encoding.items()}

train_dataset = TextDataset(tokenizer, text_corpus, MAX_LENGTH)

# The data collator will dynamically mask tokens for the MLM objective
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)


# --- 3. Model and Trainer ---
# Load the model specifically for Masked Language Modeling
model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL).to(DEVICE)

training_args = TrainingArguments(
    output_dir='./results_mlm',
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True, # We only care about the loss for this task
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)


# --- 4. Run Domain Adaptation Training ---
print("\nStarting unsupervised domain adaptation (MLM)...")
trainer.train()


# --- 5. Save the Adapted Model ---
print(f"\nSaving the domain-adapted model to '{ADAPTED_MODEL_OUTPUT_PATH}'")
trainer.save_model(ADAPTED_MODEL_OUTPUT_PATH)
tokenizer.save_pretrained(ADAPTED_MODEL_OUTPUT_PATH)

print("\n--- Domain Adaptation Complete ---")
print("Next, you will re-run the classification training, but starting with this new, domain-adapted model.")