import json

import pandas as pd
from pydantic import validate_call

from llm_etl_pipeline.customized_logger import logger


@validate_call
def load_df_from_json(json_path: str) -> pd.DataFrame:
    """
    Loads data from a JSON file, processes it, and consolidates it into a single pandas DataFrame.

    The JSON file is expected to contain a list of dictionaries, where each dictionary
    represents a document. Each document dictionary should have a single key-value pair,
    where the key is the document ID and the value is a dictionary containing document content.
    Within the document content, there should be a 'results' key whose value can be:
    1. A list of dictionaries, each representing an item with amount details (original format).
    2. A list of lists, where the first inner list is a list of dictionaries and subsequent
       inner lists contain other data (e.g., numbers) (previous new format).
    3. A list where the first element is a list of dictionaries, and the second element
       is a dictionary containing 'min_entities' (newest format).

    The function extracts these 'results' into a DataFrame for each document and
    then concatenates all document DataFrames into a single, unified DataFrame,
    adding a 'document_id' column to track the origin of each row. For the newest
    format, it also extracts 'min_entities' and adds it as a column to each row.
    For the old format, the 'min_entities' column will NOT be explicitly added
    to the document's DataFrame, and will appear as NaN/None after concatenation.

    Args:
        json_path (str): The file path to the JSON file to be loaded.

    Returns:
        pd.DataFrame: A consolidated pandas DataFrame containing data from all documents.
                      Returns an empty DataFrame if the JSON file is empty or no valid
                      data can be extracted.

    Raises:
        FileNotFoundError: If the specified `json_path` does not exist.
        json.JSONDecodeError: If the file at `json_path` is not a valid JSON.
        Exception: For other unexpected errors during file processing or DataFrame creation.
    """
    logger.info(f"Attempting to load data from JSON file: {json_path}")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON data from {json_path}.")
    except FileNotFoundError:
        logger.error(f"Error: JSON file not found at {json_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error: Invalid JSON format in file {json_path}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while opening or loading JSON: {e}")
        raise

    all_dfs = []

    if not isinstance(data, list):
        logger.warning(
            f"Expected JSON data to be a list, but got {type(data)}. Returning empty DataFrame."
        )
        return pd.DataFrame()

    for i, document_entry in enumerate(data):
        if not isinstance(document_entry, dict) or len(document_entry) != 1:
            logger.warning(
                f"Unexpected document entry format at index {i}: {document_entry}. Skipping this entry."
            )
            continue

        document_id = list(document_entry.keys())[0]
        document_content = document_entry[document_id]

        results = document_content.get("results", [])

        processed_results_data = []
        min_entities_list = None  # Initialize to None for optional extraction

        if isinstance(results, list):
            if (
                len(results) >= 1
                and isinstance(results[0], list)
                and all(isinstance(item, dict) for item in results[0])
            ):
                # This matches your newest format or the previous 'list of lists'
                # where the first inner list contains dictionaries.
                processed_results_data = results[0]
                logger.info(
                    f"Detected list of dictionaries in the first element of 'results' for document_id: {document_id}."
                )

                # Check for the second element if it's a dictionary with 'min_entities'
                if (
                    len(results) > 1
                    and isinstance(results[1], dict)
                    and "min_entities" in results[1]
                ):
                    min_entities_list = results[1]["min_entities"]
                    logger.info(
                        f"Extracted 'min_entities' for document_id: {document_id}."
                    )
                elif (
                    len(results) > 1
                    and isinstance(results[1], list)
                    and all(isinstance(item, (int, float)) for item in results[1])
                ):
                    # This handles the previous "list of numbers" format if it still appears
                    # As per your latest JSON, it's a dict, so this might not be hit,
                    # but kept for robustness.
                    logger.info(
                        f"Detected numeric list in the second element of 'results' for document_id: {document_id}. Not explicitly handled as 'min_entities' for now."
                    )
            elif all(isinstance(item, dict) for item in results):
                # Original format: 'results' is a list of dictionaries.
                logger.info(
                    f"Detected original 'results' format (list of dictionaries) for document_id: {document_id}."
                )
                processed_results_data = results
            else:
                logger.warning(
                    f"Unexpected 'results' format for document_id: {document_id}. Expected list of dicts or a specific list of lists/dicts. Skipping."
                )
                continue
        else:
            logger.warning(
                f"'results' for document_id: {document_id} is not a list. Skipping DataFrame creation for this document."
            )
            continue

        if not processed_results_data:
            logger.info(
                f"No valid results data found after processing for document_id: {document_id}. Skipping DataFrame creation for this document."
            )
            continue

        try:
            df = pd.DataFrame(processed_results_data)
            df["document_id"] = document_id

            # Add min_entities column ONLY if data was extracted for it (i.e., new format)
            if min_entities_list is not None:
                df["min_entities"] = [min_entities_list] * len(df)
                logger.info(
                    f"Added 'min_entities' column for document_id: {document_id}."
                )
            else:
                # IMPORTANT: For old format documents, 'min_entities' column is NOT added here.
                # When concatenated, this column will appear as NaN for these rows.
                logger.info(
                    f"No 'min_entities' data found for document_id: {document_id}. Column will not be explicitly added for this document."
                )

            all_dfs.append(df)
        except Exception as e:
            logger.error(
                f"Error creating DataFrame for document_id {document_id}: {e}. Skipping this document."
            )
            continue

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(
            f"Successfully concatenated {len(all_dfs)} DataFrames into a final DataFrame with {len(final_df)} rows."
        )
    else:
        final_df = pd.DataFrame()
        logger.info("No valid DataFrames were created. Returning an empty DataFrame.")

    final_df_reset = final_df.reset_index(drop=True)
    final_df_reset["document_id"] = final_df_reset["document_id"].astype(str)
    logger.info("Final DataFrame index reset.")

    logger.success(f"Successfully loaded DataFrame from {json_path}.")
    return final_df_reset
