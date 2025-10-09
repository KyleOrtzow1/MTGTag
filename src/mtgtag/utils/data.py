"""Data handling utilities for MTGTag pipeline."""

import pandas as pd
import json
import ast
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def load_tag_definitions(tag_definitions_path: Path) -> Dict[str, str]:
    """
    Load tag definitions from JSON file.

    Args:
        tag_definitions_path: Path to tag definitions JSON file

    Returns:
        Dictionary mapping tag names to descriptions
    """
    try:
        with open(tag_definitions_path, 'r') as f:
            tag_data = json.load(f)

        if not isinstance(tag_data, dict):
            raise ValueError(f"Expected dictionary in {tag_definitions_path}, got {type(tag_data)}")

        logger.info(f"Loaded {len(tag_data)} tag definitions")
        return tag_data

    except FileNotFoundError:
        logger.error(f"Tag definitions file not found: {tag_definitions_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in tag definitions file: {e}")
        raise

def load_card_database(database_path: Path) -> pd.DataFrame:
    """
    Load card database from CSV file.

    Args:
        database_path: Path to card database CSV file

    Returns:
        DataFrame containing card data
    """
    try:
        df = pd.read_csv(database_path)
        logger.info(f"Loaded card database with {len(df)} cards")
        return df

    except FileNotFoundError:
        logger.error(f"Card database file not found: {database_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Card database file is empty: {database_path}")
        raise

def parse_tags_column(tags_str: str) -> List[str]:
    """
    Parse tags from string representation to list.

    Args:
        tags_str: String representation of tags (list or comma-separated)

    Returns:
        List of tag strings
    """
    if pd.isna(tags_str):
        return []

    # Try to parse as Python literal (list)
    try:
        tags = ast.literal_eval(tags_str)
        if isinstance(tags, list):
            return [str(tag).strip() for tag in tags]
    except (ValueError, SyntaxError):
        pass

    # Fall back to comma-separated parsing
    if isinstance(tags_str, str):
        return [tag.strip() for tag in tags_str.split(',') if tag.strip()]

    return []

def validate_card_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame contains required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        True if all required columns are present

    Raises:
        ValueError: If required columns are missing
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    logger.info("Card data validation passed")
    return True

def save_processed_data(df: pd.DataFrame, output_path: Path, index: bool = False) -> None:
    """
    Save processed DataFrame to CSV file.

    Args:
        df: DataFrame to save
        output_path: Path to save the file
        index: Whether to include row indices in output
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index)
    logger.info(f"Saved processed data to {output_path} ({len(df)} rows)")

def get_tag_statistics(df: pd.DataFrame, tags_column: str = "tags") -> Dict[str, Any]:
    """
    Calculate statistics about tag distribution.

    Args:
        df: DataFrame containing tags
        tags_column: Name of the column containing tags

    Returns:
        Dictionary with tag statistics
    """
    all_tags = []
    for tags_str in df[tags_column].dropna():
        all_tags.extend(parse_tags_column(tags_str))

    tag_counts = pd.Series(all_tags).value_counts()

    stats = {
        "total_cards": len(df),
        "cards_with_tags": df[tags_column].notna().sum(),
        "unique_tags": len(tag_counts),
        "most_common_tags": tag_counts.head(10).to_dict(),
        "average_tags_per_card": len(all_tags) / len(df) if len(df) > 0 else 0
    }

    logger.info(f"Tag statistics: {stats['total_cards']} cards, {stats['unique_tags']} unique tags")
    return stats