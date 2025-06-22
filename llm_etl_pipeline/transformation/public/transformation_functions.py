import re
from typing import Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pydantic import StrictFloat, validate_call
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from llm_etl_pipeline.customized_logger import logger
from llm_etl_pipeline.transformation.internal import _cluster_list_sents
from llm_etl_pipeline.typings import (
    NonEmptyDataFrame,
    NonEmptyListStr,
    NonEmptyStr,
    RegexPattern,
)


@validate_call
def remove_semantic_duplicates(
    input_df: NonEmptyDataFrame,
    groupby_columns: NonEmptyListStr = ["document", "price"],
    target_column: NonEmptyStr = "sentence",
    model: NonEmptyStr = "all-mpnet-base-v2",
    threshold: StrictFloat = 0.8,
) -> pd.DataFrame:
    """
    Removes semantically duplicate text entries within DataFrame groups, retaining the longest sentence.

    This function groups the input DataFrame by `groupby_columns`, then applies a semantic
    clustering algorithm to the `target_column` within each group using `_cluster_list_sents`.
    It identifies and selects the longest sentence as the representative for each cluster.
    The final output is a DataFrame with semantically similar, shorter duplicates effectively
    removed, keeping only the chosen representatives.

    Args:
        input_df (NonEmptyDataFrame): The input pandas DataFrame containing the text data.
        groupby_columns (NonEmptyListStr): A list of column names to group the DataFrame by
                                           before performing text de-duplication.
                                           Defaults to `['document', 'price']`.
        target_column (NonEmptyStr): The name of the column containing the text (sentences)
                                     on which to perform semantic de-duplication.
                                     Defaults to `'sentence'`.
        model (NonEmptyStr): The name of the Sentence-BERT model to use for generating
                             embeddings. Defaults to "all-mpnet-base-v2".
        threshold (StrictFloat): The semantic similarity threshold (cosine distance) used
                                 for clustering sentences within each group. Defaults to 0.8.

    Returns:
        pd.DataFrame: A new DataFrame with semantically similar text duplicates removed.
                      For each cluster, only the longest sentence is retained as the representative
                      within its defined group.

    Raises:
        ValueError: If the `target_column` does not exist, or if there's an error loading
                    the SentenceTransformer model or during the clustering process.
        KeyError: If any of the `groupby_columns` do not exist in the DataFrame.
    """
    df = input_df.copy()
    input_columns = groupby_columns.copy()

    logger.info(
        f"Starting semantic duplicate removal process. Groupby columns: {groupby_columns}. Target column for de-duplication: {target_column}. Sentence-BERT model: {model}, Similarity threshold: {threshold}"
    )

    # Validate target_column existence
    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in the DataFrame.")
        raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")

    # Validate groupby_columns existence
    for col in input_columns:
        if col not in df.columns:
            logger.error(f"Groupby column '{col}' not found in the DataFrame.")
            raise KeyError(f"Groupby column '{col}' not found in the DataFrame.")

    try:
        model_st = SentenceTransformer(model)
        logger.info(f"Successfully loaded SentenceTransformer model: {model}")
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer model '{model}'.")
        raise ValueError(f"Failed to load SentenceTransformer model '{model}': {e}")

    # Using AgglomerativeClustering with distance_threshold
    hierarchical_clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
    )

    try:
        grouped_df = df.groupby(input_columns)[target_column].apply(
            lambda x: _cluster_list_sents(
                list(x), threshold, model_st, hierarchical_clustering
            )
        )
        logger.info(f"Sentence clustering applied across {grouped_df.shape[0]} groups.")
    except Exception as e:
        logger.erorr(f"Error during sentence clustering.")
        raise ValueError(f"Error during sentence clustering: {e}")

    df_exploded = grouped_df.reset_index().explode(target_column)

    # Ensure all columns required for the merge are present in df_exploded before merge
    # This might implicitly check for target_column again, but it's fine.
    merge_columns = groupby_columns + [target_column]

    for col in merge_columns:
        if col not in df_exploded.columns:
            logger.error(f"Missing column '{col}' in exploded DataFrame for merge.")
            raise KeyError(f"Missing column '{col}' in exploded DataFrame for merge.")

    try:
        result_df = df_exploded.merge(
            df,
            on=merge_columns,  # Use the combined list of columns for merging
            how="left",
        )
        logger.info(
            f"Merged exploded DataFrame with original. Final shape: {result_df.shape}"
        )
    except Exception as e:
        logger.error(f"Error during final merge operation: {e}")
        raise e

    logger.success(f"Semantic duplicate removal process succeded.")
    return result_df


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
            logger.error(
                f"Found 'None' or missing values in column '{column}' at indices: {missing_indices}. No missing data is allowed."
            )
            raise ValueError(
                f"Found 'None' or missing values in column '{column}' at indices: {missing_indices}. No missing data is allowed."
            )

    logger.success("No missing data found in the DataFrame. Verification successful.")
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
                logger.error(
                    f"Found negative values in numeric column '{column}' at indices: {negative_indices}. All numeric values must be non-negative."
                )
                raise ValueError(
                    f"Found negative values in numeric column '{column}' at indices: {negative_indices}. All numeric values must be non-negative."
                )

    logger.success(
        "No negative values found in numeric columns. Verification successful."
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
                logger.error(
                    f"Column '{column}' contains empty strings ('') at indices: {empty_string_indices}. Empty strings are not allowed."
                )
                raise ValueError(
                    f"Column '{column}' contains empty strings ('') at indices: {empty_string_indices}. Empty strings are not allowed."
                )

    logger.success(
        "Verification complete. No empty strings found in object type columns."
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

    logger.info(f"Starting check for numeric columns: {columns_to_check}.")

    for col in columns_to_check:
        if col not in df.columns:
            logger.error(f"Column '{col}' not found in the DataFrame.")
            raise ValueError(f"Column '{col}' not found in the DataFrame.")

        if df[col].empty:
            logger.error(
                f"Column '{col}' is empty. Cannot check an empty column for numeric values."
            )
            raise ValueError(
                f"Column '{col}' is empty or contains only nulls. Cannot check an empty column for numeric values."
            )

        # Check for null values
        if df[col].isnull().any():
            null_indices = df[col][df[col].isnull()].index.tolist()
            logger.error(
                f"Column '{col}' contains 'None' or missing values at indices: {null_indices}. All values must be non-null for numeric check."
            )
            raise ValueError(
                f"Column '{col}' contains 'None' or missing values at indices: {null_indices}. All values must be non-null for numeric check."
            )

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

            logger.error(
                f"Column '{col}' contains non-numeric data. "
                f"Sample non-numeric values: [{sample_non_numeric}]. "
                "All values in this column must be numeric."
            )
            raise ValueError(
                f"Column '{col}' contains non-numeric data. "
                f"Sample non-numeric values: [{sample_non_numeric}]. "
                "All values in this column must be numeric."
            )

    logger.success(
        "All specified columns successfully contain only numeric values and no nulls."
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

    logger.info(f"Starting check for string columns: {columns_to_check}.")
    for col in columns_to_check:
        if col not in df.columns:
            logger.error(f"Column '{col}' not found in the DataFrame.")
            raise ValueError(f"Column '{col}' not found in the DataFrame.")

        if df[col].empty:
            logger.error(
                f"Column '{col}' is empty. Cannot perform string verification on an empty column."
            )
            raise ValueError(f"Column '{col}' is empty or contains only nulls.")

        # Check for null values
        if df[col].isnull().any():
            null_indices = df[col][df[col].isnull()].index.tolist()
            logger.error(
                f"Column '{col}' contains 'None' or missing values at indices: {null_indices}. All values must be non-null."
            )
            raise ValueError(
                f"Column '{col}' contains 'None' or missing values at indices: {null_indices}. All values must be non-null."
            )

        non_string_elements = df[col].apply(lambda x: not isinstance(x, str))
        if non_string_elements.any():
            logger.error(
                f"ERROR: Column '{col}' contains non-string elements (e.g., numbers, lists, etc. stored as objects)."
            )
            raise ValueError(
                f"ERROR: Column '{col}' contains non-string elements (e.g., numbers, lists, etc. stored as objects)."
            )

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

    compiled_regex = re.compile(regex_pattern, re.IGNORECASE | re.DOTALL)

    logger.info(
        f"Checking columns  '{columns_to_check}' against regex: '{regex_pattern}'"
    )

    for col in columns_to_check:
        if col not in df.columns:
            logger.error(
                f"Column '{col}' not found in the DataFrame. Cannot check regex."
            )
            raise ValueError(
                f"Column '{col}' not found in the DataFrame. Cannot check regex."
            )

        if df[col].empty:
            logger.error(
                f"Column '{col}' is empty. Cannot perform regex check on an empty column."
            )
            raise ValueError(
                f"Column '{col}' is empty or contains only nulls. Cannot perform regex check on an empty column."
            )

        # Check for null values
        if df[col].isnull().any():
            null_indices = df[col][df[col].isnull()].index.tolist()
            logger.error(
                f"Column '{col}' contains 'None' or missing values at indices: {null_indices}. All values must be non-null for regex check."
            )
            raise ValueError(
                f"Column '{col}' contains 'None' or missing values at indices: {null_indices}. All values must be non-null for regex check."
            )

        # Check if all values are actually strings (after ensuring no nulls)
        # Using .apply and checking isinstance is a robust way to check actual types
        non_string_elements = df[col].apply(lambda x: not isinstance(x, str))
        if non_string_elements.any():
            non_string_indices = df[col][non_string_elements].index.tolist()
            logger.error(
                f"Column '{col}' contains non-string elements (e.g., numbers, lists, etc.) "
                f"at indices: {non_string_indices}. Only string values can be checked against regex."
            )
            raise ValueError(
                f"Column '{col}' contains non-string elements (e.g., numbers, lists, etc.) "
                f"at indices: {non_string_indices}. Only string values can be checked against regex."
            )

        # Iterate and check each string value against the regex
        for index, value in df[col].items():
            # At this point, we've guaranteed 'value' is a non-null string
            if not compiled_regex.search(value):
                logger.error(
                    f"Column '{col}', Row Index {index}: Value '{value}' does NOT fully satisfy the regex '{regex_pattern}'."
                )
                raise ValueError(
                    f"Column '{col}', Row Index {index}: Value '{value}' does NOT fully satisfy the regex '{regex_pattern}'."
                )

        logger.info(
            f"SUCCESS: Column '{col}' fully satisfies the regex '{regex_pattern}'."
        )

    logger.success(
        "All specified columns successfully validated against the regex pattern."
    )
    return df


@validate_call
def drop_rows_not_satisfying_regex(
    input_df: NonEmptyDataFrame,
    columns_to_check: NonEmptyListStr,
    regex_pattern: RegexPattern,
) -> pd.DataFrame:
    """
    Checks if every non-null, string-type value in the specified DataFrame columns
    fully satisfies the given regular expression. Rows where the value does not
    satisfy the regex are dropped from the DataFrame.

    This function iterates through the specified columns and their string values,
    verifying if each value completely matches the provided regular expression.
    It drops rows that fail this check. It raises errors for other discrepancies
    found, including non-existent columns, null values, empty columns, or
    non-string data.

    Args:
        input_df (NonEmptyDataFrame): The pandas DataFrame to check and modify.
        columns_to_check (NonEmptyListStr): A list of column names to verify.
        regex_pattern (RegexPattern): The regular expression pattern that each string
                                      value must fully match. The pattern is compiled
                                      with `re.IGNORECASE` and `re.DOTALL` flags.

    Returns:
        pd.DataFrame: A new DataFrame with rows dropped where values in
                      `columns_to_check` did not satisfy the regex.

    Raises:
        ValueError: If a specified column is not found in the DataFrame,
                    contains null values, is empty, or contains non-string elements.
    """

    df = input_df.copy()
    initial_rows = len(df)
    compiled_regex = re.compile(regex_pattern, re.IGNORECASE | re.DOTALL)

    logger.info(
        f"Starting row filtering based on regex non-satisfaction in columns.Regex pattern: '{regex_pattern}'.Columns to check: {columns_to_check}"
    )

    rows_to_drop_indices = set()

    for col in columns_to_check:
        if col not in df.columns:
            logger.error(
                f"Column '{col}' not found in the DataFrame. Cannot check regex."
            )
            raise ValueError(
                f"Column '{col}' not found in the DataFrame. Cannot check regex."
            )

        if df[col].empty:
            logger.error(
                f"Column '{col}' is empty. Cannot perform regex check on an empty column."
            )
            raise ValueError(
                f"Column '{col}' is empty or contains only nulls. Cannot perform regex check on an empty column."
            )

        # Check for null values
        if df[col].isnull().any():
            null_indices = df[col][df[col].isnull()].index.tolist()
            logger.error(
                f"Column '{col}' contains 'None' or missing values at indices: {null_indices}. All values must be non-null for regex check."
            )
            raise ValueError(
                f"Column '{col}' contains 'None' or missing values at indices: {null_indices}. All values must be non-null for regex check."
            )

        # Check if all values are actually strings (after ensuring no nulls)
        non_string_elements = df[col].apply(lambda x: not isinstance(x, str))
        if non_string_elements.any():
            non_string_indices = df[col][non_string_elements].index.tolist()
            logger.error(
                f"Column '{col}' contains non-string elements (e.g., numbers, lists, etc.) "
                f"at indices: {non_string_indices}. Only string values can be checked against regex."
            )
            raise ValueError(
                f"Column '{col}' contains non-string elements (e.g., numbers, lists, etc.) "
                f"at indices: {non_string_indices}. Only string values can be checked against regex."
            )

        # Identify rows to drop for the current column
        for index, value in df[col].items():
            # At this point, we've guaranteed 'value' is a non-null string
            if not compiled_regex.search(value):
                logger.info(
                    f"DROPPING: Column '{col}', Row Index {index}: Value '{value}' does NOT fully satisfy the regex '{regex_pattern}'."
                )
                rows_to_drop_indices.add(index)

    if rows_to_drop_indices:
        df = df.drop(index=list(rows_to_drop_indices))
        dropped_count = initial_rows - len(df)
        logger.success(
            f"Finished dropping rows. Total {dropped_count} rows dropped based on regex non-satisfaction. {len(df)} rows remaining."
        )
    else:
        logger.success(
            "No rows needed to be dropped. All specified values satisfied the regex."
        )
    return df


@validate_call
def drop_rows_with_non_positive_values(
    input_df: NonEmptyDataFrame, columns_to_check: NonEmptyListStr
) -> pd.DataFrame:
    """
    Drops rows from a DataFrame if any value in the specified columns is less than or equal to zero.
    This includes both negative values and zero.

    This function iterates through the specified columns and checks their numeric values.
    If a row contains a value less than or equal to zero in any of the specified columns,
    that row is dropped. Errors are raised for non-existent columns, null values,
    or non-numeric data types in the columns to be checked.

    Args:
        input_df (NonEmptyDataFrame): The pandas DataFrame to check and modify.
        columns_to_check (NonEmptyListStr): A list of column names to verify.

    Returns:
        pd.DataFrame: A new DataFrame with rows dropped that contained
                      non-positive values (negative or zero) in the specified columns.

    Raises:
        ValueError: If a specified column is not found in the DataFrame,
                    contains null values, or contains non-numeric elements.
    """

    df = input_df.copy()
    initial_rows = len(df)
    rows_to_drop_indices = set()

    logger.info(
        f"Starting check for non-positive values (<= 0) in columns and dropping corresponding rows. Columns to check: {columns_to_check}"
    )

    for col in columns_to_check:
        if col not in df.columns:
            logger.error(
                f"Column '{col}' not found in the DataFrame. Cannot perform check."
            )
            raise ValueError(
                f"Column '{col}' not found in the DataFrame. Cannot perform check."
            )

        if df[col].empty:
            logger.warning(
                f"Column '{col}' is empty. No checks will be performed on this column."
            )
            continue  # Skip this column and move to the next

        # Check for null values
        if df[col].isnull().any():
            null_indices = df[col][df[col].isnull()].index.tolist()
            logger.error(
                f"Column '{col}' contains 'None' or missing values at indices: {null_indices}. All values must be non-null for the check."
            )
            raise ValueError(
                f"Column '{col}' contains 'None' or missing values at indices: {null_indices}. All values must be non-null for the check."
            )

        # Check if all values are numeric (int or float)
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_indices = df[col][
                ~pd.to_numeric(df[col], errors="coerce").notna()
            ].index.tolist()
            logger.error(
                f"Column '{col}' contains non-numeric elements (e.g., strings, lists, etc.) "
                f"at indices: {non_numeric_indices}. Only numeric values can be checked for being non-positive."
            )
            raise ValueError(
                f"Column '{col}' contains non-numeric elements (e.g., strings, lists, etc.) "
                f"at indices: {non_numeric_indices}. Only numeric values can be checked for being non-positive."
            )

        # Identify rows to drop for the current column
        # The change is here: using df[col] <= 0 to include zero
        non_positive_value_indices = df.loc[df[col] <= 0, col].index.tolist()

        if non_positive_value_indices:
            logger.info(
                f"Found non-positive values (<= 0) in column '{col}' at indices: {non_positive_value_indices}"
            )
            for index in non_positive_value_indices:
                rows_to_drop_indices.add(index)

    if rows_to_drop_indices:
        df_filtered = df.drop(index=list(rows_to_drop_indices))
        dropped_count = initial_rows - len(df_filtered)
        logger.success(
            f"Finished dropping rows. Total {dropped_count} rows dropped due to non-positive values. {len(df_filtered)} rows remaining."
        )
        return df_filtered
    else:
        logger.success(
            "No rows needed to be dropped. All specified columns do not contain non-positive values."
        )
        return df


@validate_call
def drop_rows_if_no_column_matches_regex(
    input_df: NonEmptyDataFrame,
    columns_to_check: NonEmptyListStr,
    regex_pattern: RegexPattern,
) -> pd.DataFrame:
    """
    Drops rows from a DataFrame ONLY if NONE of the string values in the specified columns
    fully satisfy the given regular expression. If at least one column in a row matches,
    the row is kept.

    This function first performs checks to ensure the specified columns exist,
    contain no null values, and consist only of strings. It then iterates through
    each row and, for each row, checks if any of the target columns' values match
    the regex. Rows where no such match is found are dropped.

    Args:
        input_df (NonEmptyDataFrame): The pandas DataFrame to check and modify.
        columns_to_check (NonEmptyListStr): A list of column names to verify.
        regex_pattern (RegexPattern): The regular expression pattern that each string
                                      value must fully match. The pattern is compiled
                                      with `re.IGNORECASE` and `re.DOTALL` flags.

    Returns:
        pd.DataFrame: A new DataFrame with rows dropped where none of the values in
                      `columns_to_check` satisfied the regex.

    Raises:
        ValueError: If a specified column is not found in the DataFrame,
                    contains null values, or contains non-string elements.
    """

    df = input_df.copy()
    compiled_regex = re.compile(regex_pattern, re.IGNORECASE | re.DOTALL)

    logger.info(
        f"Starting row filtering based on regex match in columns. Columns to check: {columns_to_check} using a regex pattern: '{regex_pattern}'"
    )
    logger.info(
        f"Keeping rows if AT LEAST ONE of '{columns_to_check}' matches regex, otherwise dropping."
    )

    for col in columns_to_check:
        if col not in df.columns:
            logger.error(
                f"Column '{col}' not found in the DataFrame. Cannot perform regex check."
            )
            raise ValueError(
                f"Column '{col}' not found in the DataFrame. Cannot perform regex check."
            )

        if df[col].empty:
            logger.warning(
                f"Column '{col}' is empty. No regex checks will be performed on this column."
            )
            continue

        if df[col].isnull().any():
            null_indices = df[col][df[col].isnull()].index.tolist()
            logger.error(
                f"Column '{col}' contains 'None' or missing values at indices: {null_indices}. All values must be non-null for regex check."
            )
            raise ValueError(
                f"Column '{col}' contains 'None' or missing values at indices: {null_indices}. All values must be non-null for regex check."
            )

        non_string_elements = df[col].apply(lambda x: not isinstance(x, str))
        if non_string_elements.any():
            non_string_indices = df[col][non_string_elements].index.tolist()
            logger.error(
                f"Column '{col}' contains non-string elements (e.g., numbers, lists, etc.) "
                f"at indices: {non_string_indices}. Only string values can be checked against regex."
            )
            raise ValueError(
                f"Column '{col}' contains non-string elements (e.g., numbers, lists, etc.) "
                f"at indices: {non_string_indices}. Only string values can be checked against regex."
            )

    rows_to_keep_indices = []
    dropped_count = 0

    for index, row in df.iterrows():
        keep_this_row = False

        for col in columns_to_check:
            if col not in row or pd.isna(row[col]) or not isinstance(row[col], str):
                continue

            cell_value = row[col]

            if compiled_regex.search(cell_value):
                keep_this_row = True
                break

        if keep_this_row:
            rows_to_keep_indices.append(index)
        else:
            logger.info(
                f"DROPPING: Row Index {index} because NONE of the columns ({columns_to_check}) satisfied the regex '{regex_pattern}'."
            )
            dropped_count += 1

    df_filtered = df.loc[rows_to_keep_indices]

    if dropped_count > 0:
        logger.success(
            f"Finished filtering. Total {dropped_count} rows dropped. {len(df_filtered)} rows remaining."
        )
    else:
        logger.success(
            "No rows needed to be dropped. All rows had at least one specified column satisfying the regex."
        )

    return df_filtered


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
    logger.info(
        f"Starting verification for columns {columns_to_check} to ensure they contain lists of only integers."
    )

    for col_name in columns_to_check:
        logger.info(f"Processing column: '{col_name}'")

        # 1. Check if column exists
        if col_name not in df.columns:
            logger.error(
                f"Column '{col_name}' not found in the DataFrame. Cannot verify its contents."
            )
            raise ValueError(
                f"Column '{col_name}' not found in the DataFrame. Cannot verify its contents."
            )

        # 2. Check if column is empty
        if df[col_name].empty:
            logger.warning(
                f"Column '{col_name}' is empty. No list content to verify for this column."
            )
            continue  # Move to the next column if this one is empty

        # Iterate through each row in the specified column
        for index, cell_value in df[col_name].items():
            # Check for missing values first using explicit checks for None and numpy.nan
            # This avoids the ambiguous truth value error if cell_value is an array-like NaN
            if cell_value is None or (
                isinstance(cell_value, float) and np.isnan(cell_value)
            ):
                logger.error(
                    f"Column '{col_name}' contains a missing value (NaN/None) at index {index}. Expected a list."
                )
                raise ValueError(
                    f"Column '{col_name}' contains a missing value (NaN/None) at index {index}. Expected a list."
                )

            # 3. Check if the cell value is a list
            if not isinstance(cell_value, list):
                logger.error(
                    f"Cell at index {index} in column '{col_name}' is not a list. Found type: {type(cell_value)}. Expected a list."
                )
                raise ValueError(
                    f"Cell at index {index} in column '{col_name}' is not a list. Found type: {type(cell_value)}. Expected a list."
                )

            # 4. Check if all elements within the list are integers
            for element_index, element in enumerate(cell_value):
                if not isinstance(element, int):
                    logger.error(
                        f"Element at index {element_index} within the list at row {index}, "
                        f"column '{col_name}' is not an integer. Found value: {element} (type: {type(element)})."
                    )
                    raise ValueError(
                        f"Element at index {element_index} within the list at row {index}, "
                        f"column '{col_name}' is not an integer. Found value: {element} (type: {type(element)})."
                    )
        logger.info(f"Verification successful for column '{col_name}'.")

    logger.success(
        f"Verification successful: All specified columns ({columns_to_check}) contain only lists of integers."
    )
    return df


@validate_call
def reduce_list_ints_to_unique(
    input_df: NonEmptyDataFrame,  # Using pd.DataFrame; replace with NonEmptyDataFrame if defined
    target_column: NonEmptyStr = "min_entities",  # Makes the column name configurable, with a default
) -> pd.DataFrame:
    """
    Reduces the lists in a specified column of a DataFrame by removing duplicate values
    and maintaining the first-occurrence order of elements.

    This function is designed for columns that contain lists of values (typically integers)
    and gracefully handles missing values (None/NaN) by leaving them unchanged.

    Args:
        input_df (NonEmptyDataFrame): The pandas DataFrame to process.
        target_column (NonEmptyStr): The name of the column containing the lists to reduce.
                           Defaults to 'min_entities'.

    Returns:
        pd.DataFrame: A new DataFrame with the specified column modified
                      to contain only unique values within its lists.

    Raises:
        ValueError: If the specified column does not exist in the DataFrame.
        TypeError: If a value in the column is not a list, None, or np.nan.
    """
    df = input_df.copy()
    logger.info(f"Starting reduction of duplicate values in column '{target_column}'.")

    # 1. Check if the column exists
    if target_column not in df.columns:
        logger.error(f"Error: Column '{target_column}' not found in the DataFrame.")
        raise ValueError(f"Column '{target_column}' not found in the DataFrame.")

    # 2. Apply the uniqueness logic to each cell in the column
    def get_unique_elements_from_list(
        lst: Union[list, None, float],
    ) -> Union[list, None, float]:
        """
        Helper function to get unique elements from a list, preserving order.
        Also handles None and np.nan values.
        """
        if lst is None or (isinstance(lst, float) and np.isnan(lst)):
            return lst  # Leave None/NaN values unchanged
        if not isinstance(lst, list):
            # This case should ideally be prevented by verify_list_column_contains_only_ints
            # but is included for robustness.
            logger.error(
                f"Unexpected value found: {lst} (type: {type(lst)}). Expected a list, None, or NaN."
            )
            raise TypeError(
                f"Value in column '{target_column}' must be a list, None, or NaN. Found: {type(lst)}"
            )

        # Method to get unique elements while preserving order
        # dict.fromkeys() is efficient and preserves insertion order
        return list(dict.fromkeys(lst))

    try:
        df[target_column] = df[target_column].apply(get_unique_elements_from_list)
        logger.success(
            f"Reduction complete: Column '{target_column}' now contains lists with only unique values."
        )
    except Exception as e:
        logger.error(
            f"An error occurred during the reduction of column '{target_column}': {e}"
        )
        raise

    return df


@validate_call
def group_by_document_and_stack_types(
    input_df: NonEmptyDataFrame,  # Using pd.DataFrame; replace with NonEmptyDataFrame if defined
    target_column: str,  # The column to be stacked into a list
    document_id_column: str = "document_id",  # Parameter for the document ID column name
    min_entities_column: str = "min_entities",  # Parameter for the min_entities column name
) -> pd.DataFrame:
    """
    Groups the DataFrame by a specified document ID column, aggregates the `target_column`
    into a single list of strings per document, and keeps a single `min_entities_column` list
    (e.g., the first one, as they are expected to be identical per document after reduction).

    Args:
        input_df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the column whose string values should be
                             collected into a list for each document group.
        document_id_column (str): The name of the column that identifies unique documents.
                                  Defaults to 'document_id'.
        min_entities_column (str): The name of the column containing the lists of integers
                                   (e.g., after unique reduction). Defaults to 'min_entities'.

    Returns:
        pd.DataFrame: A new DataFrame with one row per unique document,
                      `target_column` as a list of strings, and `min_entities_column`
                      as a single list.

    Raises:
        ValueError: If any required column (`document_id_column`, `target_column`,
                    `min_entities_column`) is not found in the DataFrame.
    """
    df = input_df.copy()
    logger.info(
        f"Starting grouping by '{document_id_column}' and stacking '{target_column}'. "
        f"Keeping single value from '{min_entities_column}'."
    )

    # Required columns now use the dynamic parameters
    required_columns = [document_id_column, target_column, min_entities_column]
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Required column '{col}' not found in the DataFrame.")
            raise ValueError(f"Required column '{col}' not found in the DataFrame.")

    # Define the aggregation dictionary dynamically using all parameters
    aggregation_funcs = {
        target_column: lambda x: list(
            x.astype(str).unique()
        ),  # Stack the target_column values
        min_entities_column: "first",  # Take the first occurrence for min_entities_column
    }

    try:
        # Perform the groupby and aggregation using the document_id_column parameter
        grouped_df = df.groupby(document_id_column, as_index=False).agg(
            aggregation_funcs
        )

        # Reorder columns for better readability, using the dynamic parameters
        final_columns_order = [document_id_column, target_column, min_entities_column]
        grouped_df = grouped_df[final_columns_order]

        logger.success(
            f"Grouping completed. Reduced {len(df)} rows to {len(grouped_df)} unique documents."
        )
    except Exception as e:
        logger.error(f"An error occurred during grouping and stacking: {e}")
        raise

    return grouped_df
