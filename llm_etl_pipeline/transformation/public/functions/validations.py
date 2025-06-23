"""
This module provides a collection of robust data validation and quality assurance
functions for pandas DataFrames within the LLM ETL pipeline's transformation phase.

It includes specialized checks for column content, ensuring data types (numeric, string,
list of integers), adherence to regular expression patterns, and the absence of
missing values, empty strings, or negative numbers. These functions are designed
to enforce data integrity and consistency before further processing.
"""

import re

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pydantic import validate_call

from llm_etl_pipeline.customized_logger import logger
from llm_etl_pipeline.typings import NonEmptyDataFrame, NonEmptyListStr, RegexPattern


@validate_call
def verify_list_column_contains_only_ints(
    input_df: NonEmptyDataFrame, columns_to_check: NonEmptyListStr
) -> pd.DataFrame:
    """
    Verifies that for each specified column in a DataFrame:
    1. It exists.
    2. It contains only lists.
    3. Every element within these lists is an integer.

    Args:
        input_df (NonEmptyDataFrame): The pandas DataFrame to check.
        columns_to_check (NonEmptyListStr): A list of column names to verify.

    Returns:
        pd.DataFrame: The original DataFrame if all checks pass for all specified columns.

    Raises:
        ValueError: If any specified column is not found, contains non-list elements,
                    or contains lists with non-integer elements.
    """
    df = input_df.copy()
    columns_to_check_copy = columns_to_check.copy()

    logger.info(
        f"Starting verification for columns {columns_to_check_copy} "
        "to ensure they contain lists of only integers."
    )

    for col_name in columns_to_check_copy:
        logger.info(f"Processing column: '{col_name}'")

        # 1. Check if column exists
        if col_name not in df.columns:
            message = (
                f"Column '{col_name}' not found in the DataFrame. "
                "Cannot verify its contents."
            )
            logger.error(message)
            raise ValueError(message)

        # 2. Check if column is empty
        if df[col_name].empty:
            logger.warning(
                f"Column '{col_name}' is empty. "
                "No list content to verify for this column."
            )
            continue

        # Iterate through each row in the specified column
        for index, cell_value in df[col_name].items():
            # Check for missing values first using explicit checks for None and numpy.nan
            # This avoids the ambiguous truth value error if cell_value is an array-like NaN
            if cell_value is None or (
                isinstance(cell_value, float) and np.isnan(cell_value)
            ):
                message = (
                    f"Column '{col_name}' contains a missing value (NaN/None) "
                    f"at index {index}. Expected a list."
                )
                logger.error(message)
                raise ValueError(message)

            # 3. Check if the cell value is a list
            if not isinstance(cell_value, list):
                message = (
                    f"Cell at index {index} in column '{col_name}' is not a list. "
                    f"Found type: {type(cell_value)}. Expected a list."
                )
                logger.error(message)
                raise ValueError(message)

            # 4. Check if all elements within the list are integers
            for element_index, element in enumerate(cell_value):
                if not isinstance(element, int):
                    message = (
                        f"Element at index {element_index} within the list at row {index}, "
                        f"column '{col_name}' is not an integer. Found value: {element} (type: {type(element)})."
                    )
                    logger.error(message)
                    raise ValueError(message)
        logger.info(f"Verification successful for column '{col_name}'.")

    logger.success(
        f"Verification successful: All specified columns ({columns_to_check_copy}) "
        "contain only lists of integers."
    )
    return df


@validate_call
def check_string_columns(
    input_df: NonEmptyDataFrame, columns_to_check: NonEmptyListStr
) -> pd.DataFrame:
    """
    Verifies that specified columns in a pandas DataFrame contain ONLY string values and no nulls.

    This function rigorously checks each designated column to ensure it meets the following criteria:
    1. The column exists within the DataFrame.
    2. The column is not empty.
    3. There are absolutely no null (e.g., `None` or `NaN`) values present in the column.
    4. Every value within the column is a confirmed instance of the Python `str` type.

    Args:
        input_df (NonEmptyDataFrame): The pandas DataFrame to be inspected.
        columns_to_check (NonEmptyListStr): A list of column names (strings) to validate.

    Returns:
        pd.DataFrame: A copy of the original DataFrame. This function's primary purpose
                      is to perform checks and raise errors upon failure, not to modify the DataFrame.
                      The returned DataFrame allows for method chaining if desired.

    Raises:
        ValueError: If any column in `columns_to_check` is not found, is empty,
                    contains null values, or contains any non-string elements.
    """
    df = input_df.copy()
    columns_to_check_copy = columns_to_check.copy()

    logger.info(f"Starting check for string columns: {columns_to_check_copy}.")
    for col in columns_to_check_copy:
        if col not in df.columns:
            logger.error(f"Column '{col}' not found in the DataFrame.")
            raise ValueError(f"Column '{col}' not found in the DataFrame.")

        if df[col].empty:
            logger.error(
                f"Column '{col}' is empty. "
                f"Cannot perform string verification on an empty column."
            )
            raise ValueError(f"Column '{col}' is empty or contains only nulls.")

        # Check for null values
        if df[col].isnull().any():
            null_indices = df[col][df[col].isnull()].index.tolist()
            message = (
                f"Column '{col}' contains 'None' or missing values at indices: {null_indices}. "
                f"All values must be non-null."
            )
            logger.error(message)
            raise ValueError(message)

        non_string_elements = df[col].apply(lambda x: not isinstance(x, str))
        if non_string_elements.any():
            message = (
                f"ERROR: Column '{col}' contains non-string elements "
                f"(e.g., numbers, lists, etc. stored as objects)."
            )
            logger.error(message)
            raise ValueError(message)

    logger.success(
        "All specified columns successfully verified as containing only non-null string values."
    )
    return df


@validate_call
def check_columns_satisfy_regex(
    input_df: NonEmptyDataFrame,
    columns_to_check: NonEmptyListStr,
    regex_pattern: RegexPattern,
) -> pd.DataFrame:
    """
    Checks if every non-null, string-type value in the specified DataFrame columns
    fully satisfies the given regular expression.

    This function iterates through the specified columns and their string values,
    verifying if each value completely matches the provided regular expression.
    It raises errors for any discrepancies found,
    including non-existent columns, null values, empty columns, non-string data,
    or values that fail to satisfy the regex.

    Args:
        input_df (NonEmptyDataFrame): The pandas DataFrame to check.
        columns_to_check (NonEmptyListStr): A list of column names to verify.
        regex_pattern (RegexPattern): The regular expression pattern that each string
                                      value must fully match. The pattern is compiled
                                      with `re.IGNORECASE` and `re.DOTALL` flags.

    Returns:
        pd.DataFrame: The original DataFrame (`input_df`'s copy). This function
                      primarily performs checks and raises errors in case of failures.

    Raises:
        ValueError: If a specified column is not found in the DataFrame,
                    contains null values, is empty, contains non-string elements,
                    or if any string value does not fully satisfy the regex.
    """

    df = input_df.copy()
    columns_to_check_copy = columns_to_check.copy()

    compiled_regex = re.compile(regex_pattern, re.IGNORECASE | re.DOTALL)

    logger.info(
        f"Checking columns  '{columns_to_check_copy}' against regex: '{regex_pattern}'"
    )

    for col in columns_to_check_copy:
        if col not in df.columns:
            message = (
                f"Column '{col}' not found in the DataFrame. " f"Cannot check regex."
            )
            logger.error(message)
            raise ValueError(message)

        if df[col].empty:
            message = (
                f"Column '{col}' is empty. "
                f"Cannot perform regex check on an empty column."
            )
            logger.error(message)
            raise ValueError(message)

        # Check for null values
        if df[col].isnull().any():
            null_indices = df[col][df[col].isnull()].index.tolist()
            message = (
                f"Column '{col}' contains 'None' or missing values at indices: {null_indices}. "
                f"All values must be non-null for regex check."
            )
            logger.error(message)
            raise ValueError(message)

        # Check if all values are actually strings
        # Using .apply and checking isinstance
        non_string_elements = df[col].apply(lambda x: not isinstance(x, str))
        if non_string_elements.any():
            non_string_indices = df[col][non_string_elements].index.tolist()
            message = (
                f"Column '{col}' contains non-string elements (e.g., numbers, lists, etc.) "
                f"at indices: {non_string_indices}. Only string values can be checked against regex."
            )
            logger.error(message)
            raise ValueError(message)

        # Iterate and check each string value against the regex
        for index, value in df[col].items():
            # At this point, we've guaranteed 'value' is a non-null string
            if not compiled_regex.search(value):
                message = (
                    f"Column '{col}', Row Index {index}: Value '{value}' "
                    f"does NOT fully satisfy the regex '{regex_pattern}'."
                )
                logger.error(message)
                raise ValueError(message)

        logger.info(
            f"SUCCESS: Column '{col}' fully satisfies the regex '{regex_pattern}'."
        )

    logger.success(
        "All specified columns successfully validated against the regex pattern."
    )
    return df


@validate_call
def check_numeric_columns(
    input_df: NonEmptyDataFrame, columns_to_check: NonEmptyListStr
) -> pd.DataFrame:
    """
    Checks if specified columns in a pandas DataFrame contain only numeric values and no nulls.

    This function rigorously verifies each designated column to ensure it meets the following criteria:
    1. The column exists within the DataFrame.
    2. The column is not empty.
    3. There are no null (e.g., `None` or `NaN`) values present in the column.
    4. All values within the column are of a numeric data type (e.g., int, float).

    Args:
        input_df (NonEmptyDataFrame): The pandas DataFrame to be inspected.
        columns_to_check (NonEmptyListStr): A list of column names (strings) to validate.

    Returns:
        pd.DataFrame: A copy of the original DataFrame. This function's primary purpose
                      is to perform checks and raise errors upon failure, not to modify the DataFrame.
                      The returned DataFrame allows for method chaining if desired.

    Raises:
        ValueError: If any column in `columns_to_check` is not found, is empty,
                    contains null values, or contains any non-numeric data.
    """
    df = input_df.copy()
    columns_to_check_copy = columns_to_check.copy()

    logger.info(f"Starting check for numeric columns: {columns_to_check_copy}.")

    for col in columns_to_check_copy:
        if col not in df.columns:
            logger.error(f"Column '{col}' not found in the DataFrame.")
            raise ValueError(f"Column '{col}' not found in the DataFrame.")

        if df[col].empty:
            message = (
                f"Column '{col}' is empty or contains only nulls. "
                f"Cannot check an empty column for numeric values."
            )
            logger.error(message)
            raise ValueError(message)

        # Check for null values
        if df[col].isnull().any():
            null_indices = df[col][df[col].isnull()].index.tolist()
            message = (
                f"Column '{col}' contains 'None' or missing values at indices: {null_indices}. "
                f"All values must be non-null for numeric check."
            )
            logger.error(message)
            raise ValueError(message)

        # Check for numeric data type
        if not is_numeric_dtype(df[col]):
            # If the entire column isn't numeric, find the specific non-numeric values
            non_numeric_values = [
                value
                for value in df[col]
                if not pd.isna(value) and not isinstance(value, (int, float))
            ]

            # To avoid printing too many values, just take a sample
            sample_non_numeric = ", ".join(map(str, non_numeric_values[:5]))

            message = (
                f"Column '{col}' contains non-numeric data. "
                f"Sample non-numeric values: [{sample_non_numeric}]. "
                "All values in this column must be numeric."
            )
            logger.error(message)
            raise ValueError(message)

    logger.success(
        "All specified columns successfully contain only numeric values and no nulls."
    )
    return df


@validate_call
def verify_no_empty_strings(input_df: NonEmptyDataFrame) -> pd.DataFrame:
    """
    Checks a pandas DataFrame for the presence of None values or empty strings and raises errors if found.

    This function iterates through all columns of the input DataFrame.
    It performs the following checks:
    1. For columns with an 'object' dtype (typically strings), it checks for empty string values ('').

    Args:
        input_df (NonEmptyDataFrame): The DataFrame to check.

    Returns:
        pd.DataFrame: A copy of the original DataFrame. This function's primary purpose
                      is to perform validation checks and raise errors upon failure,
                      not to modify the DataFrame.

    Raises:
        ValueError: If any 'object' dtype column contains empty string values ('').
    """
    df = input_df.copy()

    for column in df.columns:
        # Check for empty strings only if the column is of object (string) type
        if df[column].dtype == "object":
            if (df[column] == "").any():
                empty_string_indices = df[column][df[column] == ""].index.tolist()
                message = (
                    f"Column '{column}' contains empty strings ('') at indices: "
                    f"{empty_string_indices}. Empty strings are not allowed."
                )
                logger.error(message)
                raise ValueError(message)

    logger.success(
        "Verification complete. No empty strings found in object type columns."
    )
    return df


@validate_call
def verify_no_negatives(input_df: NonEmptyDataFrame) -> pd.DataFrame:
    """
    Checks a pandas DataFrame for the presence of negative values in its numeric columns.

    This function iterates through all columns of the input DataFrame.
    For each column identified as numeric, it verifies that no value is less than zero.

    Args:
        input_df (NonEmptyDataFrame): The pandas DataFrame to check.

    Returns:
        pd.DataFrame: A copy of the original DataFrame. This function's primary purpose
                      is to perform validation checks and raise errors upon failure,
                      not to modify the DataFrame.

    Raises:
        ValueError: If any numeric column contains one or more negative values.
    """
    df = input_df.copy()
    logger.info("Verifying DataFrame for negative values in numeric columns.")

    for column in df.columns:
        # Check for None values (NaN for numeric types, None for objects)
        if is_numeric_dtype(df[column]):
            # Exclude NaN values from the negative check to avoid false positives
            if (df[column] < 0).any():
                negative_indices = df[column][df[column] < 0].index.tolist()
                message = (
                    f"Found negative values in numeric column '{column}' at indices: "
                    f"{negative_indices}. All numeric values must be non-negative."
                )
                logger.error(message)
                raise ValueError(message)

    logger.success(
        "No negative values found in numeric columns." " Verification successful."
    )
    return df


@validate_call
def verify_no_missing_data(input_df: NonEmptyDataFrame) -> pd.DataFrame:
    """
    Checks a pandas DataFrame for the presence of any missing values (None or NaN).

    This function iterates through all columns of the input DataFrame.
    If any column is found to contain `None` or `NaN` values, it immediately
    raises a `ValueError` indicating the column and the indices where missing
    data was detected.

    Args:
        input_df (NonEmptyDataFrame): The pandas DataFrame to check for missing data.

    Returns:
        pd.DataFrame: A copy of the original DataFrame. This function's primary purpose
                      is to perform validation checks and raise errors upon failure,
                      not to modify the DataFrame.

    Raises:
        ValueError: If any column in the DataFrame contains `None` or `NaN` values.
    """
    df = input_df.copy()
    logger.info("Verifying DataFrame for missing data (None or NaN values).")

    for column in df.columns:
        # Check for None values (NaN for numeric types, None for objects)
        if df[column].isnull().any():
            missing_indices = df[column][df[column].isnull()].index.tolist()
            message = (
                f"Found 'None' or missing values in column '{column}' at indices: "
                f"{missing_indices}. No missing data is allowed."
            )
            logger.error(message)
            raise ValueError(message)

    logger.success("No missing data found in the DataFrame. Verification successful.")
    return df
