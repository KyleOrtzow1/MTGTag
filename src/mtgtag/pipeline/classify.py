"""Classification module for MTGTag pipeline."""

import argparse
from pathlib import Path
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

from ..utils.logging import setup_logging
from ..utils.data import load_card_database, save_processed_data
from ..models.classifier import MTGMultiLabelClassifier
from ..config import CARD_DATABASE_PATH, CLASSIFIER_MODEL_PATH


class MTGCardInferenceDataset(Dataset):
    """Dataset for MTG card classification inference."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        """
        Initialize dataset.

        Args:
            df: DataFrame with card data
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

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

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'index': idx
        }


def classify_cards(
    input_data_path: Path,
    output_path: Path,
    model_path: Path = CLASSIFIER_MODEL_PATH,
    thresholds_path: Path = None,
    batch_size: int = 32,
    include_probabilities: bool = False
) -> None:
    """
    Classify MTG cards using trained model.

    Args:
        input_data_path: Path to input card database CSV
        output_path: Path to save classified cards CSV
        model_path: Path to trained classifier model
        thresholds_path: Path to optimal thresholds JSON (optional)
        batch_size: Batch size for inference
        include_probabilities: Whether to include prediction probabilities in output
    """
    logger = logging.getLogger(__name__)

    logger.info("Starting card classification...")

    # Load input data
    logger.info(f"Loading card database from {input_data_path}")
    df = load_card_database(input_data_path)
    logger.info(f"Loaded {len(df)} cards")

    # Load label mapping
    label_mapping_path = model_path / "label_mapping.json"
    if not label_mapping_path.exists():
        logger.error(f"Label mapping not found: {label_mapping_path}")
        raise FileNotFoundError("Please train the model first")

    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
        idx_to_label = {int(k): v for k, v in label_mapping['idx_to_label'].items()}

    num_labels = len(idx_to_label)
    logger.info(f"Number of labels: {num_labels}")

    # Load optimal thresholds if provided
    thresholds = None
    if thresholds_path is None:
        thresholds_path = model_path / "optimal_thresholds.json"

    if thresholds_path.exists():
        logger.info(f"Loading optimal thresholds from {thresholds_path}")
        with open(thresholds_path, 'r') as f:
            thresholds_data = json.load(f)
            thresholds = thresholds_data.get('optimal_thresholds', {})
        logger.info(f"Loaded thresholds for {len(thresholds)} labels")
    else:
        logger.info("No optimal thresholds found, using default threshold 0.5")

    # Load tokenizer
    from transformers import AutoTokenizer
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # Create dataset
    logger.info("Preparing dataset...")
    dataset = MTGCardInferenceDataset(df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Load model
    logger.info(f"Loading model from {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model directly from HuggingFace format
    from transformers import AutoModelForSequenceClassification
    hf_model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(device)

    # Create wrapper with loaded model
    import torch.nn as nn

    class LoadedClassifier(nn.Module):
        def __init__(self, hf_model, num_labels):
            super().__init__()
            self.model = hf_model
            self.num_labels = num_labels

        def predict_probabilities(self, input_ids, attention_mask=None):
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.sigmoid(outputs.logits)
                return probabilities

    model = LoadedClassifier(hf_model, num_labels)
    logger.info("Loaded model from HuggingFace checkpoint")

    # Run inference
    model.eval()
    all_predictions = []
    all_probabilities = []

    logger.info("Running classification...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Classifying"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Get probabilities
            probs = model.predict_probabilities(input_ids, attention_mask)
            all_probabilities.append(probs.cpu().numpy())

            # Apply thresholds
            batch_predictions = []
            for prob_row in probs:
                predicted_tags = []
                for label_idx, prob in enumerate(prob_row):
                    label_name = idx_to_label[label_idx]

                    # Get threshold for this label
                    if thresholds and label_name in thresholds:
                        threshold = thresholds[label_name]
                    else:
                        threshold = 0.5

                    # Apply threshold
                    if prob >= threshold:
                        predicted_tags.append(label_name)

                batch_predictions.append(predicted_tags)

            all_predictions.extend(batch_predictions)

    # Concatenate probabilities
    all_probabilities = np.vstack(all_probabilities)

    logger.info("Processing results...")

    # Add predictions to dataframe
    df['predicted_tags'] = [','.join(tags) if tags else '' for tags in all_predictions]
    df['num_predicted_tags'] = [len(tags) for tags in all_predictions]

    # Optionally add probabilities
    if include_probabilities:
        logger.info("Adding probability scores...")
        for label_idx, label_name in idx_to_label.items():
            df[f'prob_{label_name}'] = all_probabilities[:, label_idx]

    # Calculate statistics
    total_predictions = sum(df['num_predicted_tags'])
    avg_tags_per_card = total_predictions / len(df) if len(df) > 0 else 0
    cards_with_tags = (df['num_predicted_tags'] > 0).sum()

    logger.info(f"\nClassification Statistics:")
    logger.info(f"  Total cards classified: {len(df)}")
    logger.info(f"  Cards with at least one tag: {cards_with_tags} ({cards_with_tags/len(df)*100:.1f}%)")
    logger.info(f"  Average tags per card: {avg_tags_per_card:.2f}")
    logger.info(f"  Total tag predictions: {total_predictions}")

    # Save results
    logger.info(f"\nSaving results to {output_path}")
    save_processed_data(df, output_path)

    logger.info("Classification complete!")


def main():
    """Main entry point for card classification."""
    parser = argparse.ArgumentParser(
        description="Classify MTG cards using trained multi-label classifier"
    )
    parser.add_argument(
        "input_data",
        type=Path,
        nargs="?",
        default=CARD_DATABASE_PATH,
        help="Path to input card database CSV"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save classified cards CSV"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=CLASSIFIER_MODEL_PATH,
        help="Path to trained classifier model"
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="Path to optimal thresholds JSON (default: model_path/optimal_thresholds.json)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)"
    )
    parser.add_argument(
        "--include-probabilities",
        action="store_true",
        help="Include prediction probabilities in output CSV"
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
    if not args.input_data.exists():
        logger.error(f"Input data file not found: {args.input_data}")
        return 1

    if not args.model.exists():
        logger.error(f"Model not found: {args.model}")
        logger.error("Please run mtgtag-train first")
        return 1

    # Classify cards
    try:
        classify_cards(
            args.input_data,
            args.output,
            args.model,
            args.thresholds,
            args.batch_size,
            args.include_probabilities
        )
        return 0
    except Exception as e:
        logger.error(f"Classification failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
