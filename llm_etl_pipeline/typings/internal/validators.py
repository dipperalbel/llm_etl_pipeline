import re
from typing import Any

import pandas as pd


def _validate_regex_syntax(regex: str) -> str:
    """
    Validates the syntax of a given regular expression string.

    Args:
        regex: The regular expression string to validate.

    Returns:
        The validated regular expression string if its syntax is valid.

    Raises:
        ValueError: If the regular expression string has invalid syntax.
    """
    if regex is None:  # If the field is optional and None, do nothing
        return regex
    try:
        re.compile(regex)
    except re.error as e:
        # If compilation fails, raise a ValueError with a clear message
        raise ValueError(f"Invalid regular expression syntax: {e}")
    return regex


def _ensure_dataframe_type(v: Any) -> pd.DataFrame:
    """
    Ensures the input is a pandas DataFrame and is not None.

    This validator is intended to run early in a validation chain.

    Args:
        v: The input value to validate.

    Returns:
        The validated pandas DataFrame.

    Raises:
        ValueError: If the input is None.
        TypeError: If the input is not a pandas DataFrame.
    """
    if v is None:
        raise ValueError("DataFrame cannot be None.")
    return v  # Return the DataFrame for subsequent validators/Pydantic parsing


def _validate_non_empty_dataframe(v: pd.DataFrame) -> pd.DataFrame:
    """
    Validates that a pandas DataFrame is not empty (i.e., contains at least one row).

    Args:
        v: The pandas DataFrame to validate.

    Returns:
        The validated pandas DataFrame if it's not empty.

    Raises:
        ValueError: If the DataFrame is empty.
    """
    if v.empty:
        raise ValueError(
            "DataFrame must contain at least one row (i.e., not be empty)."
        )
    return v
