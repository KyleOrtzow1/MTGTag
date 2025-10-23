"""Tag diagnosis module for MTGTag pipeline."""

import argparse
from pathlib import Path
from collections import Counter
import logging

from ..utils.logging import setup_logging
from ..utils.data import load_tag_definitions, load_card_database, parse_tags_column
from ..config import TAG_DEFINITIONS_PATH

def diagnose_tags(
    labeled_data_path: Path,
    tag_definitions_path: Path = TAG_DEFINITIONS_PATH
) -> bool:
    """
    Diagnose tag consistency between labeled data and tag definitions.

    Args:
        labeled_data_path: Path to labeled subset CSV file
        tag_definitions_path: Path to tag definitions JSON file

    Returns:
        True if all tags are valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    tags_column = "tags"  # Hardcoded column name

    # Load data
    logger.info("Loading datasets...")
    try:
        df = load_card_database(labeled_data_path)
        tag_data = load_tag_definitions(tag_definitions_path)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return False

    # Extract official tags
    official_labels = set(tag_data.keys())
    logger.info(f"Loaded {len(official_labels)} official tags")

    # Parse tags from labeled data
    df[f'{tags_column}_list'] = df[tags_column].apply(parse_tags_column)

    # Get all tags used in the dataset
    all_tags_in_data_flat = [
        tag for tags_list in df[f'{tags_column}_list'] for tag in tags_list
    ]
    all_unique_tags_in_data = set(all_tags_in_data_flat)

    # Find extraneous tags
    extraneous_tags = all_unique_tags_in_data - official_labels

    if not extraneous_tags:
        logger.info("✅ Validation Successful: All tags in labeled data are officially defined")
        return True
    else:
        logger.warning(f"⚠️ Validation Failed: Found {len(extraneous_tags)} undefined tags")
        logger.warning("These tags will be IGNORED by the training script")

        # Count occurrences for prioritization
        extraneous_tag_counts = Counter(
            tag for tag in all_tags_in_data_flat if tag in extraneous_tags
        )

        logger.info("Undefined Tag Report:")
        for tag, count in extraneous_tag_counts.most_common():
            logger.info(f"  - '{tag}': {count} occurrences")

        return False

def main():
    """Main entry point for tag diagnosis."""
    parser = argparse.ArgumentParser(
        description="Diagnose tag consistency in MTGTag labeled data"
    )
    parser.add_argument(
        "labeled_data",
        type=Path,
        help="Path to labeled subset CSV file"
    )
    parser.add_argument(
        "--tag-definitions",
        type=Path,
        default=TAG_DEFINITIONS_PATH,
        help="Path to tag definitions JSON file"
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

    # Run diagnosis
    success = diagnose_tags(
        args.labeled_data,
        args.tag_definitions
    )

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())