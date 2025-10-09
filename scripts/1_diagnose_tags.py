import pandas as pd
import json
import ast
from collections import Counter

# --- Configuration ---
LABELED_DATA_PATH = "labeled_subset.csv"
TAG_DEFINITIONS_PATH = "tag_definitions.json"

# --- 1. Load Data ---
print("Loading datasets...")
try:
    df = pd.read_csv(LABELED_DATA_PATH)
    with open(TAG_DEFINITIONS_PATH, 'r') as f:
        tag_data = json.load(f)
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure both '{LABELED_DATA_PATH}' and '{TAG_DEFINITIONS_PATH}' are present.")
    exit()

# --- 2. Extract Official and Actual Tags ---

# CORRECTED SECTION:
# This new logic correctly handles the {"TagName": "Description"} format.
if not isinstance(tag_data, dict):
    print(f"Error: Expected '{TAG_DEFINITIONS_PATH}' to contain a JSON object (a dictionary).")
    exit()

# The official tags are now the keys of the dictionary.
official_labels = set(tag_data.keys())
print(f"Loaded {len(official_labels)} official tags from '{TAG_DEFINITIONS_PATH}'.")


# Safely parse the 'card_tags' column
def parse_card_tags(tags_str):
    try:
        tags = ast.literal_eval(str(tags_str))
        return tags if isinstance(tags, list) else []
    except (ValueError, SyntaxError):
        return []

df['card_tags_list'] = df['card_tags'].apply(parse_card_tags)

# Get a flat list of all tags used in the dataset
all_tags_in_data_flat = [tag for tags_list in df['card_tags_list'] for tag in tags_list]
# Get the unique set of tags used
all_unique_tags_in_data = set(all_tags_in_data_flat)


# --- 3. Diagnose and Report ---

# Find tags that are in the CSV but NOT in the official definitions
extraneous_tags = all_unique_tags_in_data - official_labels

if not extraneous_tags:
    print("\n✅ Validation Successful: All tags in 'labeled_subset.csv' are officially defined.")
else:
    print(f"\n⚠️ Validation Failed: Found {len(extraneous_tags)} undefined tags in '{LABELED_DATA_PATH}'.")
    print("These tags will be IGNORED by the training script. It is recommended to fix them in the CSV.")
    
    # Count occurrences of each extraneous tag for prioritization
    extraneous_tag_counts = Counter(tag for tag in all_tags_in_data_flat if tag in extraneous_tags)
    
    print("\nUndefined Tag Report (tag: count):")
    for tag, count in extraneous_tag_counts.most_common():
        print(f"  - '{tag}': {count} occurrences")