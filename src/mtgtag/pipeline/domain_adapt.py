"""Domain adaptation module for MTGTag pipeline."""

import argparse
from pathlib import Path
import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import pandas as pd

from ..utils.logging import setup_logging
from ..utils.data import load_card_database
from ..config import CARD_DATABASE_PATH, DOMAIN_ADAPTED_MODEL_PATH, DEFAULT_CONFIG


def prepare_mlm_dataset(df: pd.DataFrame, tokenizer, max_length: int = 512):
    """
    Prepare dataset for masked language modeling.

    Args:
        df: DataFrame containing card data
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length

    Returns:
        Dataset ready for MLM training
    """
    logger = logging.getLogger(__name__)

    # Combine card text fields for training (exclude name and keywords)
    # Use position-based format without structural labels
    texts = []
    for _, row in df.iterrows():
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

        if text_parts:
            texts.append(" | ".join(text_parts))

    logger.info(f"Prepared {len(texts)} text samples for domain adaptation")

    # Create dataset
    dataset = Dataset.from_dict({"text": texts})

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    return tokenized_dataset


def domain_adapt_model(
    input_data_path: Path,
    output_model_path: Path = DOMAIN_ADAPTED_MODEL_PATH,
    base_model: str = None,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    max_length: int = None
) -> None:
    """
    Fine-tune a transformer model on MTG card text for domain adaptation.

    Args:
        input_data_path: Path to card database CSV
        output_model_path: Path to save the adapted model
        base_model: Base model to use (default from config)
        epochs: Number of training epochs (default from config)
        batch_size: Training batch size (default from config)
        learning_rate: Learning rate (default from config)
        max_length: Maximum sequence length (default from config)
    """
    logger = logging.getLogger(__name__)

    # Use config defaults if not specified
    config = DEFAULT_CONFIG["domain_adaptation"]
    base_model = base_model or config["base_model"]
    epochs = epochs or config["epochs"]
    batch_size = batch_size or config["batch_size"]
    learning_rate = learning_rate or config["learning_rate"]
    max_length = max_length or config["max_length"]

    logger.info(f"Starting domain adaptation with base model: {base_model}")
    logger.info(f"Training config: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

    # Load card database
    logger.info(f"Loading card database from {input_data_path}")
    df = load_card_database(input_data_path)

    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForMaskedLM.from_pretrained(base_model)

    # Prepare dataset
    logger.info("Preparing dataset for masked language modeling...")
    dataset = prepare_mlm_dataset(df, tokenizer, max_length)

    # Split into train/validation
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(eval_dataset)}")

    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_model_path),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=str(output_model_path / "logs"),
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    # Train
    logger.info("Starting domain adaptation training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving domain-adapted model to {output_model_path}")
    trainer.save_model(str(output_model_path))
    tokenizer.save_pretrained(str(output_model_path))

    logger.info("Domain adaptation complete!")


def main():
    """Main entry point for domain adaptation."""
    parser = argparse.ArgumentParser(
        description="Domain adapt transformer model on MTG card text"
    )
    parser.add_argument(
        "input_data",
        type=Path,
        nargs="?",
        default=CARD_DATABASE_PATH,
        help="Path to card database CSV file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DOMAIN_ADAPTED_MODEL_PATH,
        help="Path to save the adapted model"
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model to use (default: distilbert-base-uncased)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (default: 16)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default: 5e-5)"
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

    # Validate input path
    if not args.input_data.exists():
        logger.error(f"Input data file not found: {args.input_data}")
        return 1

    # Run domain adaptation
    try:
        domain_adapt_model(
            args.input_data,
            args.output,
            args.base_model,
            args.epochs,
            args.batch_size,
            args.learning_rate
        )
        return 0
    except Exception as e:
        logger.error(f"Domain adaptation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
