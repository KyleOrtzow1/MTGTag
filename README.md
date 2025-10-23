# MTGTag - Magic: The Gathering Card Functional Tagging System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-yellow)](https://huggingface.co/transformers/)

A production-ready machine learning pipeline for automatically classifying Magic: The Gathering cards into 83 functional categories like "Card Draw", "Removal", "Ramp", and more.

## 🎯 Overview

MTGTag is a standalone ML system extracted from the DeckAgent project, featuring:
- **Domain-adapted transformer models** fine-tuned for MTG card text
- **Multi-label classification** with optimized thresholds
- **Complete training pipeline** from data validation to bulk inference
- **Professional software architecture** with modular design and CLI tools

## ✨ Key Features

- 🧠 **Domain-Adapted Transformer**: Fine-tuned DistilBERT model for MTG text understanding
- 🏷️ **Multi-Label Classification**: Assigns multiple functional tags per card with confidence scores
- ⚖️ **Optimized Thresholds**: Precision/recall balanced per-label thresholds
- 🔄 **Complete ML Pipeline**: End-to-end workflow from data validation to production inference
- 🛠️ **Professional Architecture**: Modular codebase with proper logging, CLI tools, and package structure
- 📊 **Performance Monitoring**: Built-in metrics tracking and model evaluation

## 📁 Project Structure

```
MTGTag/
├── src/mtgtag/                 # Main package source code
│   ├── pipeline/               # Pipeline modules (diagnose, clean, train, etc.)
│   ├── utils/                  # Shared utilities (logging, data handling)
│   └── config.py              # Configuration management
├── models/                     # Trained model artifacts
│   ├── domain_adapted/         # Domain-adapted transformer (~500MB)
│   └── classifier/             # Multi-label classifier with optimal thresholds (~500MB)
├── data/                       # Data files
│   ├── mtg_cards_database.csv  # Complete MTG card dataset (33,424 cards)
│   ├── most_important_tags.json # 83 functional tag definitions
│   └── mtg_ml_sample.csv       # Training sample (5,000 labeled cards)
├── docs/                       # Documentation
├── setup.py & pyproject.toml   # Package configuration
└── README.md                   # This file
```

## 🚀 Installation

### Prerequisites

**Python Version**: Python 3.11 is recommended for GPU support. Python 3.13 does not yet have CUDA-enabled PyTorch builds available.

### Option 1: CPU-Only Installation (Simple)

```bash
# Clone the repository
git clone https://github.com/yourusername/mtgtag.git
cd mtgtag

# Install in development mode
pip install -e .
```

### Option 2: GPU Installation (Recommended for Training)

For GPU acceleration (10x faster training), you must install PyTorch with CUDA support before installing MTGTag:

```bash
# Clone the repository
git clone https://github.com/yourusername/mtgtag.git
cd mtgtag

# Create virtual environment with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support (MUST be done first)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install MTGTag
pip install -e .
```

**GPU Requirements:**
- NVIDIA GPU with CUDA support
- CUDA 12.1 or compatible version
- 8GB+ GPU memory recommended

**Check GPU availability:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Verify Installation

```bash
mtgtag-diagnose --help
```

## 🔧 Usage

### Complete Training Pipeline

Run the 6-step pipeline to train a new model from scratch:

```bash
# 1. Diagnose tag distribution and quality
mtgtag-diagnose --input data/mtg_ml_sample.csv

# 2. Clean and validate tags (optional if data is already clean)
mtgtag-clean data/mtg_ml_sample.csv data/mtg_ml_sample_clean.csv

# 3. Domain adaptation on full card database (adapts transformer to MTG text)
mtgtag-domain-adapt

# 4. Train multi-label classifier (15 epochs recommended)
mtgtag-train

# 5. Optimize classification thresholds (improves F1 by ~8%)
mtgtag-optimize data/mtg_ml_sample_clean.csv

# 6. Classify all cards using trained model
mtgtag-classify data/mtg_cards_database.csv --output data/mtg_cards_classified.csv
```

### Quick Start (Pre-trained Models)

```bash
# Use existing models for immediate classification
mtgtag-classify data/mtg_cards_database.csv --output data/mtg_cards_classified.csv
```

### Python API Usage

```python
from mtgtag.pipeline.classify import classify_cards
from mtgtag.utils.data import load_card_database

# Load and classify cards
df = load_card_database("path/to/cards.csv")
classified_df = classify_cards(df, model_path="models/classifier")
print(classified_df.head())
```

## 📊 Data & Models

### Included Assets
- **Tag Definitions** (`data/most_important_tags.json`): 83 functional categories with descriptions
- **Card Database** (`data/mtg_cards_database.csv`): 33,424 MTG cards with complete metadata
- **Training Sample** (`data/mtg_ml_sample.csv`): 5,000 labeled cards for model training
- **Trained Models** (`models/`): Domain-adapted transformer + multi-label classifier

### Data Structure
Cards include the following fields:
- **Identifiers**: `id`, `name`
- **Mana Information**: `mana_cost`, `cmc` (converted mana cost)
- **Card Properties**: `type_line`, `oracle_text`, `colors`, `color_identity`, `keywords`
- **Combat Stats**: `power`, `toughness`, `loyalty`
- **Labels**: `tags` (multi-label functional classifications)

### Model Artifacts
- **Domain-Adapted Model** (`models/domain_adapted/`): DistilBERT fine-tuned on MTG card text
- **Classifier Model** (`models/classifier/`): Multi-label classifier with optimal thresholds
- **Threshold Config** (`models/classifier/optimal_thresholds.json`): Per-label optimized thresholds

## Functional Tags

The system classifies cards into **83 functional categories** covering:

**Resources**: Card Draw, Ramp, Mana Dorks, Tutors, Card Advantage
**Combat**: Evasion, Pump Effects, Token Generation, Attack Triggers
**Removal**: Burn, Creature Removal, Artifact/Enchantment Removal, Board Wipes
**Protection**: Hexproof/Shroud, Indestructible, Protection from Colors
**Creature Types**: Dragons, Elves, Goblins, Merfolk, Vampires, Zombies, and more
**Color Effects**: Red/Blue/Green/White/Black specific mechanics
**Strategy**: Reanimate, Sacrifice, Lifegain, Mill, ETB/LTB triggers

See `data/most_important_tags.json` for complete definitions and descriptions of all 83 tags.

## 📤 Output Format

The pipeline produces `mtg_cards_classified.csv` with:
- **Original card data**: `id`, `name`, `mana_cost`, `cmc`, `type_line`, `oracle_text`, etc.
- **Predicted tags**: Multi-label functional classifications (comma-separated)
- **Confidence scores**: Per-tag probability scores

**Sample Output:**
```csv
id,name,mana_cost,type_line,oracle_text,predicted_tags
"uuid-123","Lightning Bolt","{R}","Instant","Lightning Bolt deals 3...","burn,burn-creature,burn-player,removal"
"uuid-456","Rampant Growth","{1}{G}","Sorcery","Search your library...","ramp,card-advantage,green-effect"
```

## ⚙️ System Requirements

- **Python**: 3.11 recommended (3.13 lacks CUDA support), 3.8+ minimum
- **Memory**: 8GB+ RAM recommended (16GB for training)
- **GPU Memory**: 8GB+ VRAM for training (NVIDIA GPU with CUDA support)
- **Storage**: ~2GB for models and data
- **GPU**: Optional for inference, **strongly recommended for training** (10x speedup)
- **Dependencies**: PyTorch, Transformers, scikit-learn, pandas, accelerate, datasets

### Performance Benchmarks
- **Training Time**: ~35 minutes (15 epochs on GPU) vs ~6-8 hours (CPU)
- **Domain Adaptation**: ~2 hours on 33,424 cards (GPU)
- **Inference Speed**: ~8-10 cards/second (batch processing with GPU)
- **Model Accuracy**: F1 score of 0.9145 with optimized thresholds

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **Overall F1** | **0.9145** (with optimized thresholds) |
| **Base F1** | 0.8428 (before threshold optimization) |
| **Training Set** | 5,000 labeled cards |
| **Full Dataset** | 33,424 MTG cards |
| **Tag Categories** | 83 functional types |
| **Training Epochs** | 15 (optimal) |
| **Avg Optimal Threshold** | 0.391 (varies per tag) |

### Performance by Category
- **Best performing tags** (F1 > 0.99): `activated-ability`, `burn-planeswalker`, `burn-creature`
- **Strong performance** (F1 > 0.95): `burn`, `attack-trigger`, `acceleration`
- **Minimum performance**: F1 > 0.85 across all 83 tags

## 🤝 Contributing

This project follows standard Python development practices:

```bash
# Set up development environment
pip install -e ".[dev]"

# Run code formatting
black src/ tests/
flake8 src/ tests/

# Run type checking
mypy src/
```

## 📄 License

MIT License - See LICENSE file for details. Models trained on publicly available MTG card data.

## 🎯 Use Cases

- **Deck Building Tools**: Automatic card categorization for recommendation systems
- **Card Analysis**: Functional analysis of new MTG sets
- **Data Science Projects**: Feature engineering for MTG-related ML tasks
- **Educational**: Example of production ML pipeline with transformers

---

*Built with ❤️ for the Magic: The Gathering community*