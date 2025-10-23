"""Threshold optimization module for MTGTag pipeline."""

import argparse
from pathlib import Path
import logging
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

from ..utils.logging import setup_logging
from ..utils.data import load_card_database, load_tag_definitions, parse_tags_column
from ..models.classifier import MTGMultiLabelClassifier
from ..config import (
    TAG_DEFINITIONS_PATH,
    CLASSIFIER_MODEL_PATH,
    DEFAULT_CONFIG
)
from .train import MTGCardDataset


def get_predictions_and_labels(model, dataloader, device):
    """
    Get model predictions and true labels for threshold optimization.

    Args:
        model: Trained classifier model
        dataloader: DataLoader for validation data
        device: Device to run on

    Returns:
        Tuple of (predictions, labels) as numpy arrays
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Getting predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Get probabilities
            probs = model.predict_probabilities(input_ids, attention_mask)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate all batches
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    return all_probs, all_labels


def optimize_threshold_for_label(
    probs: np.ndarray,
    labels: np.ndarray,
    metric: str = 'f1',
    search_range: tuple = (0.1, 0.9),
    search_steps: int = 50
) -> tuple:
    """
    Find optimal threshold for a single label.

    Args:
        probs: Prediction probabilities for the label
        labels: True binary labels
        metric: Metric to optimize ('f1', 'precision', 'recall')
        search_range: Range of thresholds to search
        search_steps: Number of steps in the search

    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    thresholds = np.linspace(search_range[0], search_range[1], search_steps)
    best_threshold = 0.5
    best_metric_value = 0.0

    for threshold in thresholds:
        predictions = (probs >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(labels, predictions, zero_division=0)
        elif metric == 'precision':
            score = precision_score(labels, predictions, zero_division=0)
        elif metric == 'recall':
            score = recall_score(labels, predictions, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_metric_value:
            best_metric_value = score
            best_threshold = threshold

    return best_threshold, best_metric_value


def optimize_thresholds(
    labeled_data_path: Path,
    tag_definitions_path: Path = TAG_DEFINITIONS_PATH,
    model_path: Path = CLASSIFIER_MODEL_PATH,
    output_path: Path = None,
    metric: str = None,
    search_range: tuple = None,
    search_steps: int = None,
    batch_size: int = 16
) -> dict:
    """
    Optimize classification thresholds for each label.

    Args:
        labeled_data_path: Path to labeled validation data
        tag_definitions_path: Path to tag definitions JSON
        model_path: Path to trained classifier
        output_path: Path to save optimal thresholds JSON
        metric: Metric to optimize
        search_range: Range of thresholds to search
        search_steps: Number of search steps
        batch_size: Batch size for inference

    Returns:
        Dictionary of optimal thresholds per label
    """
    logger = logging.getLogger(__name__)

    # Use config defaults if not specified
    config = DEFAULT_CONFIG["threshold_optimization"]
    metric = metric or config["metric"]
    search_range = search_range or config["search_range"]
    search_steps = search_steps or config["search_steps"]

    if output_path is None:
        output_path = model_path / "optimal_thresholds.json"

    logger.info("Starting threshold optimization...")
    logger.info(f"Metric: {metric}, Search range: {search_range}, Steps: {search_steps}")

    # Load data
    logger.info(f"Loading labeled data from {labeled_data_path}")
    df = load_card_database(labeled_data_path)

    logger.info(f"Loading tag definitions from {tag_definitions_path}")
    tag_definitions = load_tag_definitions(tag_definitions_path)

    # Load label mapping
    label_mapping_path = model_path / "label_mapping.json"
    if not label_mapping_path.exists():
        logger.error(f"Label mapping not found: {label_mapping_path}")
        raise FileNotFoundError("Please train the model first")

    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
        label_to_idx = label_mapping['label_to_idx']
        idx_to_label = {int(k): v for k, v in label_mapping['idx_to_label'].items()}

    num_labels = len(label_to_idx)
    logger.info(f"Number of labels: {num_labels}")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # Create dataset
    logger.info("Preparing dataset...")
    dataset = MTGCardDataset(df, tokenizer, label_to_idx)
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

    # Get predictions and labels
    logger.info("Getting model predictions...")
    all_probs, all_labels = get_predictions_and_labels(model, dataloader, device)

    # Optimize threshold for each label
    logger.info(f"Optimizing thresholds for {num_labels} labels...")
    optimal_thresholds = {}
    threshold_metrics = {}

    for label_idx in tqdm(range(num_labels), desc="Optimizing thresholds"):
        label_name = idx_to_label[label_idx]
        label_probs = all_probs[:, label_idx]
        label_labels = all_labels[:, label_idx]

        # Skip if no positive examples
        if label_labels.sum() == 0:
            logger.warning(f"No positive examples for label '{label_name}', using default threshold 0.5")
            optimal_thresholds[label_name] = 0.5
            threshold_metrics[label_name] = 0.0
            continue

        # Optimize threshold
        optimal_threshold, best_metric_value = optimize_threshold_for_label(
            label_probs,
            label_labels,
            metric,
            search_range,
            search_steps
        )

        optimal_thresholds[label_name] = float(optimal_threshold)
        threshold_metrics[label_name] = float(best_metric_value)

    # Calculate overall statistics
    avg_threshold = np.mean(list(optimal_thresholds.values()))
    avg_metric = np.mean([v for v in threshold_metrics.values() if v > 0])

    logger.info(f"\nOptimization complete!")
    logger.info(f"Average optimal threshold: {avg_threshold:.3f}")
    logger.info(f"Average {metric}: {avg_metric:.4f}")

    # Show some examples
    logger.info("\nSample optimal thresholds:")
    for i, (label, threshold) in enumerate(list(optimal_thresholds.items())[:10]):
        metric_value = threshold_metrics[label]
        logger.info(f"  {label}: {threshold:.3f} ({metric}={metric_value:.4f})")

    # Save thresholds
    logger.info(f"\nSaving optimal thresholds to {output_path}")
    output_data = {
        'optimal_thresholds': optimal_thresholds,
        'threshold_metrics': threshold_metrics,
        'optimization_config': {
            'metric': metric,
            'search_range': search_range,
            'search_steps': search_steps
        },
        'statistics': {
            'avg_threshold': float(avg_threshold),
            f'avg_{metric}': float(avg_metric)
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info("Threshold optimization complete!")

    return optimal_thresholds


def main():
    """Main entry point for threshold optimization."""
    parser = argparse.ArgumentParser(
        description="Optimize classification thresholds for MTG multi-label classifier"
    )
    parser.add_argument(
        "labeled_data",
        type=Path,
        help="Path to labeled validation data CSV"
    )
    parser.add_argument(
        "--tag-definitions",
        type=Path,
        default=TAG_DEFINITIONS_PATH,
        help="Path to tag definitions JSON"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=CLASSIFIER_MODEL_PATH,
        help="Path to trained classifier"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save optimal thresholds JSON (default: model_path/optimal_thresholds.json)"
    )
    parser.add_argument(
        "--metric",
        choices=["f1", "precision", "recall"],
        default=None,
        help="Metric to optimize (default: f1)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference"
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

    if not args.model.exists():
        logger.error(f"Model not found: {args.model}")
        logger.error("Please run mtgtag-train first")
        return 1

    # Optimize thresholds
    try:
        optimize_thresholds(
            args.labeled_data,
            args.tag_definitions,
            args.model,
            args.output,
            args.metric,
            batch_size=args.batch_size
        )
        return 0
    except Exception as e:
        logger.error(f"Threshold optimization failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
