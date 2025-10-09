"""Tag cleaning module for MTGTag pipeline."""

import argparse
from pathlib import Path
from typing import Dict, List
import logging

from ..utils.logging import setup_logging
from ..utils.data import load_card_database, parse_tags_column, save_processed_data

# Default tag correction mappings
DEFAULT_TAG_CORRECTIONS = {
    "Top-Deck Manipulation": "Scry / Surveil / Top-Deck Manipulation",
    "Double Strike": "First Strike / Double Strike",
    "First Strike": "First Strike / Double Strike",
    "Forced Combat": "Forced Combat / Goad",
    "CardDraw": "Card Draw"  # Fix typo
}

def clean_and_remap_tags(
    tags_str: str,
    tag_correction_map: Dict[str, str]
) -> List[str]:
    """
    Parse tag list, correct known bad tags, and return clean list.

    Args:
        tags_str: String representation of tags list
        tag_correction_map: Mapping of incorrect tags to correct tags

    Returns:
        List of cleaned tag strings
    """
    tags_list = parse_tags_column(tags_str)
    if not tags_list:
        return []

    # Apply corrections
    cleaned_tags = [tag_correction_map.get(tag, tag) for tag in tags_list]
    return cleaned_tags

def clean_tags_dataset(
    input_path: Path,
    output_path: Path,
    tags_column: str = "card_tags",
    tag_corrections: Dict[str, str] = None
) -> None:
    """
    Clean tags in dataset and save to new file.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output cleaned CSV file
        tags_column: Name of column containing tags
        tag_corrections: Dictionary mapping incorrect to correct tags
    """
    logger = logging.getLogger(__name__)

    if tag_corrections is None:
        tag_corrections = DEFAULT_TAG_CORRECTIONS

    # Load data
    logger.info(f"Loading data from {input_path}")
    df = load_card_database(input_path)

    # Apply cleaning
    logger.info("Applying tag corrections...")
    df[tags_column] = df[tags_column].apply(
        lambda x: str(clean_and_remap_tags(x, tag_corrections))
    )

    # Save cleaned data
    logger.info(f"Saving cleaned data to {output_path}")
    save_processed_data(df, output_path)

    # Report corrections applied
    corrections_applied = sum(
        1 for correction in tag_corrections.keys()
        if correction in df[tags_column].str.contains(correction, na=False).sum()
    )

    logger.info(f"Tag cleaning complete. Applied {len(tag_corrections)} correction rules")
    logger.info("Re-run mtgtag-diagnose to confirm fixes, then proceed with training")

def main():
    """Main entry point for tag cleaning."""
    parser = argparse.ArgumentParser(
        description="Clean and correct tags in MTGTag labeled data"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to input labeled data CSV file"
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to output cleaned CSV file"
    )
    parser.add_argument(
        "--tags-column",
        default="card_tags",
        help="Name of column containing tags"
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
    if not args.input_file.exists():
        logger.error(f"Input file not found: {args.input_file}")
        return 1

    # Clean tags
    try:
        clean_tags_dataset(
            args.input_file,
            args.output_file,
            args.tags_column
        )
        return 0
    except Exception as e:
        logger.error(f"Tag cleaning failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())