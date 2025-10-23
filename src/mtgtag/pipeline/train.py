"""Classifier training module for MTGTag pipeline."""

import argparse
from pathlib import Path
import logging
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
from tqdm import tqdm

from ..utils.logging import setup_logging
from ..utils.data import load_card_database, load_tag_definitions, parse_tags_column
from ..models.classifier import MTGMultiLabelClassifier
from ..config import (
    TAG_DEFINITIONS_PATH,
    DOMAIN_ADAPTED_MODEL_PATH,
    CLASSIFIER_MODEL_PATH,
    DEFAULT_CONFIG
)


class MTGCardDataset(Dataset):
    """Dataset for MTG card multi-label classification."""

    def __init__(self, df: pd.DataFrame, tokenizer, label_to_idx: dict, max_length: int = 512):
        """
        Initialize dataset.

        Args:
            df: DataFrame with card data and tags
            tokenizer: Tokenizer for text encoding
            label_to_idx: Mapping from label names to indices
            max_length: Maximum sequence length
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_to_idx = label_to_idx
        self.max_length = max_length
        self.num_labels = len(label_to_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Prepare text (exclude name and keywords)
        # Use position-based format without structural labels
        text_parts = []

        # Mana cost
        if pd.notna(row.get('mana_cost')):
            text_parts.append(str(row['mana_cost']))

        # CMC (converted mana cost)
        if pd.notna(row.get('cmc')):
            text_parts.append(str(row['cmc']))

        # Type line
        if pd.notna(row.get('type_line')):
            text_parts.append(str(row['type_line']))

        # Oracle text (contains keywords already)
        if pd.notna(row.get('oracle_text')):
            text_parts.append(str(row['oracle_text']))

        # Power/Toughness for creatures
        if pd.notna(row.get('power')) and pd.notna(row.get('toughness')):
            text_parts.append(f"{row['power']}/{row['toughness']}")

        text = " | ".join(text_parts) if text_parts else "Unknown Card"

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Parse tags and create multi-label vector
        tags_str = row.get('tags', '')
        tags_list = parse_tags_column(tags_str)

        # Create binary label vector
        labels = torch.zeros(self.num_labels, dtype=torch.float)
        for tag in tags_list:
            if tag in self.label_to_idx:
                labels[self.label_to_idx[tag]] = 1.0

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, label_names):
    """Evaluate model on validation set."""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()

            # Get predictions (using 0.5 threshold for now)
            probs = torch.sigmoid(outputs.logits)
            predictions = (probs > 0.5).float()

            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate all batches
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)

    # Calculate metrics
    f1_micro = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_predictions, average='micro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='micro', zero_division=0)

    avg_loss = total_loss / len(dataloader)

    return {
        'loss': avg_loss,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall
    }


def train_classifier(
    labeled_data_path: Path,
    tag_definitions_path: Path = TAG_DEFINITIONS_PATH,
    model_input_path: Path = DOMAIN_ADAPTED_MODEL_PATH,
    model_output_path: Path = CLASSIFIER_MODEL_PATH,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    validation_split: float = None
) -> None:
    """
    Train multi-label classifier on labeled MTG cards.

    Args:
        labeled_data_path: Path to labeled card data CSV
        tag_definitions_path: Path to tag definitions JSON
        model_input_path: Path to domain-adapted model
        model_output_path: Path to save trained classifier
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        validation_split: Fraction of data for validation
    """
    logger = logging.getLogger(__name__)

    # Use config defaults if not specified
    config = DEFAULT_CONFIG["classifier_training"]
    epochs = epochs or config["epochs"]
    batch_size = batch_size or config["batch_size"]
    learning_rate = learning_rate or config["learning_rate"]
    validation_split = validation_split or config["validation_split"]

    logger.info("Starting classifier training...")
    logger.info(f"Config: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

    # Load data
    logger.info(f"Loading labeled data from {labeled_data_path}")
    df = load_card_database(labeled_data_path)

    logger.info(f"Loading tag definitions from {tag_definitions_path}")
    tag_definitions = load_tag_definitions(tag_definitions_path)

    # Create label mapping
    label_names = sorted(tag_definitions.keys())
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    num_labels = len(label_names)

    logger.info(f"Number of labels: {num_labels}")

    # Load tokenizer and base model
    logger.info(f"Loading model from {model_input_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_input_path))

    # Create dataset
    logger.info("Preparing dataset...")
    dataset = MTGCardDataset(df, tokenizer, label_to_idx)

    # Split into train/validation
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Initialize model
    logger.info("Initializing classifier model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model = MTGMultiLabelClassifier(
        model_name_or_path=str(model_input_path),
        num_labels=num_labels
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    best_f1 = 0.0
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        logger.info(f"Train Loss: {train_loss:.4f}")

        # Evaluate
        val_metrics = evaluate(model, val_loader, device, label_names)
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Val F1 (micro): {val_metrics['f1_micro']:.4f}")
        logger.info(f"Val F1 (macro): {val_metrics['f1_macro']:.4f}")
        logger.info(f"Val Precision: {val_metrics['precision']:.4f}")
        logger.info(f"Val Recall: {val_metrics['recall']:.4f}")

        # Save best model
        if val_metrics['f1_micro'] > best_f1:
            best_f1 = val_metrics['f1_micro']
            logger.info(f"New best F1: {best_f1:.4f}. Saving model...")

            # Save model
            model_output_path.mkdir(parents=True, exist_ok=True)

            # Save the underlying HuggingFace model with config
            model.model.save_pretrained(str(model_output_path))
            tokenizer.save_pretrained(str(model_output_path))

            # Save label mapping
            import json
            with open(model_output_path / "label_mapping.json", 'w') as f:
                json.dump({
                    'label_to_idx': label_to_idx,
                    'idx_to_label': idx_to_label
                }, f, indent=2)

    logger.info(f"\nTraining complete! Best F1: {best_f1:.4f}")
    logger.info(f"Model saved to {model_output_path}")


def main():
    """Main entry point for classifier training."""
    parser = argparse.ArgumentParser(
        description="Train MTG multi-label classifier"
    )
    parser.add_argument(
        "labeled_data",
        type=Path,
        help="Path to labeled card data CSV"
    )
    parser.add_argument(
        "--tag-definitions",
        type=Path,
        default=TAG_DEFINITIONS_PATH,
        help="Path to tag definitions JSON"
    )
    parser.add_argument(
        "--input-model",
        type=Path,
        default=DOMAIN_ADAPTED_MODEL_PATH,
        help="Path to domain-adapted model"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CLASSIFIER_MODEL_PATH,
        help="Path to save trained classifier"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate"
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=None,
        help="Fraction of data for validation"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    # Validate paths
    if not args.labeled_data.exists():
        logger.error(f"Labeled data file not found: {args.labeled_data}")
        return 1

    if not args.tag_definitions.exists():
        logger.error(f"Tag definitions file not found: {args.tag_definitions}")
        return 1

    if not args.input_model.exists():
        logger.error(f"Input model not found: {args.input_model}")
        logger.error("Please run mtgtag-domain-adapt first")
        return 1

    # Train classifier
    try:
        train_classifier(
            args.labeled_data,
            args.tag_definitions,
            args.input_model,
            args.output,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.validation_split
        )
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
