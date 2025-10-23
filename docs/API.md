# MTGTag API Reference

## Command-Line Interface

### `mtgtag-diagnose`
Validates tag consistency between labeled data and tag definitions.

```bash
mtgtag-diagnose [OPTIONS]
```

**Options:**
- `--input`: Path to labeled subset CSV file (default: data/mtg_ml_sample.csv)
- `--tag-definitions`: Path to tag definitions JSON (default: data/most_important_tags.json)
- `--log-level`: Logging level (DEBUG|INFO|WARNING|ERROR)

**Notes:**
- Tags column is hardcoded to `tags`
- Checks for undefined tags, tag distribution, and data quality

**Example:**
```bash
mtgtag-diagnose --input data/mtg_ml_sample.csv
```

---

### `mtgtag-clean`
Cleans and corrects tags in labeled dataset.

```bash
mtgtag-clean <input_file> <output_file> [OPTIONS]
```

**Arguments:**
- `input_file`: Path to input labeled data CSV
- `output_file`: Path for cleaned output CSV

**Options:**
- `--log-level`: Logging level

**Notes:**
- Tags column is hardcoded to `tags`
- Applies tag corrections defined in config (currently empty by default)
- Validates against tag definitions in `data/most_important_tags.json`

**Example:**
```bash
mtgtag-clean data/mtg_ml_sample.csv data/mtg_ml_sample_clean.csv
```

---

### `mtgtag-domain-adapt`
Fine-tunes transformer model for MTG domain using Masked Language Modeling (MLM).

```bash
mtgtag-domain-adapt [OPTIONS]
```

**Options:**
- `--base-model`: Base transformer model (default: distilbert-base-uncased)
- `--output-dir`: Output directory for adapted model (default: models/domain_adapted)
- `--data-path`: Path to full card database (default: data/mtg_cards_database.csv)
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Training batch size (default: 16)
- `--learning-rate`: Learning rate (default: 5e-5)

**Notes:**
- Trains on full 33,424 card database (no labels needed)
- Uses position-based text format: `{mana_cost} | {cmc} | {type_line} | {oracle_text} | {power}/{toughness}`
- Excludes card names and keywords to prevent memorization
- Takes ~2 hours on GPU for 3 epochs

**Example:**
```bash
mtgtag-domain-adapt --epochs 3 --batch-size 16
```

---

### `mtgtag-train`
Trains multi-label classifier on clean labeled data.

```bash
mtgtag-train [OPTIONS]
```

**Options:**
- `--model-path`: Path to domain-adapted model (default: models/domain_adapted)
- `--data-path`: Path to clean labeled data (default: data/mtg_ml_sample_clean.csv)
- `--output-dir`: Output directory for trained classifier (default: models/classifier)
- `--epochs`: Training epochs (default: 10, **15 recommended**)
- `--batch-size`: Batch size (default: 16)
- `--learning-rate`: Learning rate (default: 2e-5)

**Notes:**
- Trains on 5,000 labeled cards with 83 tags
- Uses same position-based text format as domain adaptation
- Saves model using HuggingFace `save_pretrained()` format
- Takes ~35 minutes for 15 epochs on GPU
- Achieves F1 ~0.84 before threshold optimization

**Example:**
```bash
mtgtag-train --epochs 15 --batch-size 16
```

---

### `mtgtag-optimize`
Optimizes classification thresholds for balanced metrics (per-label).

```bash
mtgtag-optimize <data_path> [OPTIONS]
```

**Arguments:**
- `data_path`: Path to validation data CSV

**Options:**
- `--model-path`: Path to trained classifier (default: models/classifier)
- `--metric`: Optimization metric (default: f1)
- `--min-threshold`: Minimum threshold to search (default: 0.1)
- `--max-threshold`: Maximum threshold to search (default: 0.9)
- `--steps`: Number of threshold steps to try (default: 50)
- `--output-file`: Path to save optimized thresholds (default: models/classifier/optimal_thresholds.json)

**Notes:**
- Searches for optimal threshold per tag (83 thresholds total)
- Improves F1 from ~0.84 to ~0.91 (+8.5%)
- Average optimal threshold: ~0.391 (varies per tag)
- Takes ~40 seconds on GPU

**Example:**
```bash
mtgtag-optimize data/mtg_ml_sample_clean.csv --metric f1
```

---

### `mtgtag-classify`
Applies trained models to classify cards in bulk.

```bash
mtgtag-classify <input_path> [OPTIONS]
```

**Arguments:**
- `input_path`: Input card database CSV path

**Options:**
- `--output`: Output classified CSV (default: data/mtg_cards_classified.csv)
- `--model-path`: Path to trained classifier (default: models/classifier)
- `--thresholds`: Path to optimized thresholds JSON (default: models/classifier/optimal_thresholds.json)
- `--batch-size`: Inference batch size (default: 16)

**Notes:**
- Classifies all cards with 83 functional tags
- Uses optimized thresholds if available
- Output includes predicted tags as comma-separated string
- Processes ~8-10 cards/second on GPU

**Example:**
```bash
mtgtag-classify data/mtg_cards_database.csv --output data/mtg_cards_classified.csv
```

## Python API

### Core Classes

#### Loading Trained Models

Load trained classifiers using HuggingFace's AutoModel:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("models/classifier")
tokenizer = AutoTokenizer.from_pretrained("models/classifier")

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Get predictions
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.sigmoid(outputs.logits)
```

**Note:** Models are saved using HuggingFace's `save_pretrained()` format, not custom PyTorch state dicts.

### Utility Functions

#### Data Utilities (`mtgtag.utils.data`)

```python
from mtgtag.utils.data import (
    load_tag_definitions,
    load_card_database,
    parse_tags_column,
    get_tag_statistics
)

# Load tag definitions (supports list-of-dicts format)
tags = load_tag_definitions("data/most_important_tags.json")
# Returns: {"removal": "Cards that can remove...", "ramp": "...", ...}

# Load card database
df = load_card_database("data/mtg_cards_database.csv")

# Parse tags from string
tags_list = parse_tags_column("removal,ramp,card-advantage")
# Returns: ["removal", "ramp", "card-advantage"]

# Get tag statistics
stats = get_tag_statistics(df, tags_column='tags')
```

**Note:** Tag definitions in `most_important_tags.json` use format:
```json
[
  {"tag": "removal", "definition": "Cards that can remove..."},
  {"tag": "ramp", "definition": "Cards that produce extra mana..."}
]
```

#### Logging Utilities (`mtgtag.utils.logging`)

```python
from mtgtag.utils.logging import setup_logging
from pathlib import Path

# Setup logger
logger = setup_logging(
    name="my_script",
    level="INFO",
    log_file=Path("logs/my_script.log"),
    format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger.info("Processing started")
logger.error("Error occurred: %s", error_msg)
```

### Pipeline Functions

#### Diagnosis (`mtgtag.pipeline.diagnose`)

```python
from mtgtag.pipeline.diagnose import diagnose_tags

# Diagnose tag consistency
success = diagnose_tags(
    labeled_data_path=Path("data/mtg_ml_sample.csv"),
    tag_definitions_path=Path("data/most_important_tags.json"),
    tags_column="tags"  # Hardcoded throughout pipeline
)
```

#### Cleaning (`mtgtag.pipeline.clean`)

```python
from mtgtag.pipeline.clean import clean_tags_dataset

# Clean tags in dataset
clean_tags_dataset(
    input_path=Path("data/mtg_ml_sample.csv"),
    output_path=Path("data/mtg_ml_sample_clean.csv"),
    tags_column="tags",  # Hardcoded throughout pipeline
    tag_corrections={}  # Empty by default
)
```

### Configuration

```python
from mtgtag.config import (
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    TAG_DEFINITIONS_PATH,
    DEFAULT_CONFIG
)

# Access paths
print(f"Data directory: {DATA_DIR}")
print(f"Models directory: {MODELS_DIR}")

# Access configuration
domain_config = DEFAULT_CONFIG["domain_adaptation"]
print(f"Default epochs: {domain_config['epochs']}")
```

## Error Handling

All functions use consistent error handling patterns:

- **FileNotFoundError**: For missing input files
- **ValueError**: For invalid data or parameters
- **RuntimeError**: For model loading/training failures

Example:
```python
try:
    df = load_card_database("nonexistent.csv")
except FileNotFoundError as e:
    logger.error(f"Database file not found: {e}")
    sys.exit(1)
```

## Return Types and Data Formats

### Tag Data Format
Tags are represented as lists of strings:
```python
tags = ["Card Draw", "Cantrip", "Instant"]
```

### Prediction Output
Model predictions include probabilities and binary classifications:
```python
{
    "predictions": [[0, 1, 15, 23]],  # Label indices
    "probabilities": [[0.95, 0.87, 0.73, 0.82]],  # Confidence scores
    "labels": ["Card Draw", "Cantrip", "Instant", "Blue"]  # Label names
}
```

### Card Database Schema
Expected columns in card database:
- `id`: Unique card identifier (string)
- `name`: Card name (string) - **excluded from model input**
- `mana_cost`: Mana cost (string, e.g., "{2}{U}")
- `cmc`: Converted mana cost (integer/float)
- `type_line`: Card type (string, e.g., "Instant")
- `oracle_text`: Rules text (string)
- `colors`: Card colors (string/list)
- `color_identity`: Color identity (string/list)
- `keywords`: Card keywords (string/list) - **excluded from model input**
- `power`: Creature power (string/float, nullable)
- `toughness`: Creature toughness (string/float, nullable)
- `loyalty`: Planeswalker loyalty (string/int, nullable)
- `tags`: Functional tags (comma-separated string or list, required for training)

**Model Input Format (position-based):**
```
{mana_cost} | {cmc} | {type_line} | {oracle_text} | {power}/{toughness}
```
Example: `{2}{U} | 3.0 | Instant | Draw two cards. | `