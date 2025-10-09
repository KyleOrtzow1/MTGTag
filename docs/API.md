# MTGTag API Reference

## Command-Line Interface

### `mtgtag-diagnose`
Validates tag consistency between labeled data and tag definitions.

```bash
mtgtag-diagnose <labeled_data> [OPTIONS]
```

**Arguments:**
- `labeled_data`: Path to labeled subset CSV file

**Options:**
- `--tag-definitions`: Path to tag definitions JSON (default: data/tag_definitions.json)
- `--tags-column`: Column name containing tags (default: card_tags)
- `--log-level`: Logging level (DEBUG|INFO|WARNING|ERROR)

**Example:**
```bash
mtgtag-diagnose data/labeled_subset.csv --log-level DEBUG
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
- `--tags-column`: Column name containing tags (default: card_tags)
- `--log-level`: Logging level

**Example:**
```bash
mtgtag-clean data/raw_labels.csv data/clean_labels.csv
```

---

### `mtgtag-domain-adapt`
Fine-tunes transformer model for MTG domain.

```bash
mtgtag-domain-adapt [OPTIONS]
```

**Options:**
- `--base-model`: Base transformer model (default: distilbert-base-uncased)
- `--output-dir`: Output directory for adapted model (default: models/domain_adapted)
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--learning-rate`: Learning rate

**Example:**
```bash
mtgtag-domain-adapt --epochs 5 --batch-size 32
```

---

### `mtgtag-train`
Trains multi-label classifier on clean data.

```bash
mtgtag-train [OPTIONS]
```

**Options:**
- `--model-path`: Path to domain-adapted model
- `--data-path`: Path to clean labeled data
- `--output-dir`: Output directory for trained classifier
- `--epochs`: Training epochs
- `--batch-size`: Batch size
- `--learning-rate`: Learning rate

---

### `mtgtag-optimize`
Optimizes classification thresholds for balanced metrics.

```bash
mtgtag-optimize [OPTIONS]
```

**Options:**
- `--model-path`: Path to trained classifier
- `--data-path`: Path to validation data
- `--metric`: Optimization metric (f1|precision|recall)
- `--output-file`: Path to save optimized thresholds

---

### `mtgtag-classify`
Applies trained models to classify cards in bulk.

```bash
mtgtag-classify [OPTIONS]
```

**Options:**
- `--input`: Input card database CSV
- `--output`: Output classified CSV
- `--model-path`: Path to trained classifier
- `--thresholds`: Path to optimized thresholds JSON
- `--batch-size`: Inference batch size

## Python API

### Core Classes

#### `MTGMultiLabelClassifier`

Multi-label classifier for MTG functional tags.

```python
from mtgtag.models.classifier import MTGMultiLabelClassifier

# Initialize classifier
classifier = MTGMultiLabelClassifier(
    model_name_or_path="models/domain_adapted",
    num_labels=72,
    dropout_rate=0.1
)

# Get predictions
probabilities = classifier.predict_probabilities(input_ids, attention_mask)
predictions = classifier.predict_with_threshold(input_ids, attention_mask, thresholds)
```

**Methods:**
- `forward(input_ids, attention_mask, labels=None)`: Forward pass
- `predict_probabilities(input_ids, attention_mask)`: Get probability scores
- `predict_with_threshold(input_ids, attention_mask, thresholds, default_threshold)`: Apply custom thresholds

### Utility Functions

#### Data Utilities (`mtgtag.utils.data`)

```python
from mtgtag.utils.data import (
    load_tag_definitions,
    load_card_database,
    parse_tags_column,
    validate_card_data,
    save_processed_data,
    get_tag_statistics
)

# Load tag definitions
tags = load_tag_definitions("data/tag_definitions.json")

# Load card database
df = load_card_database("data/full_card_database.csv")

# Parse tags from string
tags_list = parse_tags_column("['Card Draw', 'Cantrip']")

# Validate required columns
validate_card_data(df, required_columns=['name', 'card_text', 'tags'])

# Save processed data
save_processed_data(df, "output/processed.csv")

# Get tag statistics
stats = get_tag_statistics(df, tags_column='tags')
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
    labeled_data_path=Path("data/labeled_subset.csv"),
    tag_definitions_path=Path("data/tag_definitions.json"),
    tags_column="card_tags"
)
```

#### Cleaning (`mtgtag.pipeline.clean`)

```python
from mtgtag.pipeline.clean import clean_tags_dataset

# Clean tags in dataset
clean_tags_dataset(
    input_path=Path("data/raw_labels.csv"),
    output_path=Path("data/clean_labels.csv"),
    tags_column="card_tags",
    tag_corrections={"BadTag": "GoodTag"}
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
- `name`: Card name (string)
- `mana_cost`: Mana cost (string, e.g., "{2}{U}")
- `card_text`: Rules text (string)
- `type_line`: Card type (string)
- `tags`: Functional tags (list of strings, optional for input)