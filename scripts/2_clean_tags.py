import pandas as pd
import json
import ast

# --- Configuration ---
LABELED_DATA_PATH = "labeled_subset.csv"
CLEANED_OUTPUT_PATH = "labeled_subset_clean.csv"

# This mapping defines how to fix the bad tags.
# We will replace the 'key' with the 'value'.
TAG_CORRECTION_MAP = {
    "Top-Deck Manipulation": "Scry / Surveil / Top-Deck Manipulation",
    "Double Strike": "First Strike / Double Strike",
    "First Strike": "First Strike / Double Strike",
    "Forced Combat": "Forced Combat / Goad",
    "CardDraw": "Card Draw" # Fixing the typo
}

# --- 1. Load Data ---
print(f"Loading data from '{LABELED_DATA_PATH}'...")
try:
    df = pd.read_csv(LABELED_DATA_PATH)
except FileNotFoundError:
    print(f"Error: Could not find '{LABELED_DATA_PATH}'. Please ensure it's in the correct directory.")
    exit()

# --- 2. Define Cleaning Function ---
def clean_and_remap_tags(tags_str):
    """
    Parses a tag list, corrects known bad tags using the mapping,
    and returns a clean list of tags.
    """
    try:
        # Safely parse the string representation of the list
        tags_list = ast.literal_eval(str(tags_str))
        if not isinstance(tags_list, list):
            return []
    except (ValueError, SyntaxError):
        return []

    # Apply the corrections
    cleaned_tags = [TAG_CORRECTION_MAP.get(tag, tag) for tag in tags_list]
    
    return cleaned_tags

# --- 3. Apply Cleaning and Save ---
print("Applying tag corrections...")
# Apply the function to the 'card_tags' column
df['card_tags'] = df['card_tags'].apply(lambda x: str(clean_and_remap_tags(x)))

# Save the cleaned dataframe to a new file
print(f"Saving cleaned data to '{CLEANED_OUTPUT_PATH}'...")
df.to_csv(CLEANED_OUTPUT_PATH, index=False)

print("\n--- Cleaning Complete ---")
print(f"A new file '{CLEANED_OUTPUT_PATH}' has been created with the corrected tags.")
print("You can now re-run 'diagnose_tags.py' to confirm the fix, then proceed with training using the new clean file.")