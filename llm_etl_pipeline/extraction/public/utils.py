import os
import re
from pathlib import Path

from pydantic import validate_call

from llm_etl_pipeline.customized_logger import logger
from llm_etl_pipeline.typings import NonEmptyStr


@validate_call
def get_filtered_fully_general_series_call_pdfs(
    directory_path: NonEmptyStr,
) -> list[Path]:
    """
    Retrieves a list of PDF files from a given directory based on these rules:
    1. Includes all PDFs that contain 'call' in their name but are NOT part of
       the 'PROGRAMCODE-YYYY-TYPE-GRANT-CATEGORY-XX' general series pattern.
    2. Includes ONLY the single PDF from EACH 'PROGRAMCODE-YYYY-TYPE-GRANT-CATEGORY-XX' series
       that contains 'call' in its name and has the lowest numerical 'XX' value for that
       specific combination of PROGRAMCODE, TYPE, GRANT, and CATEGORY.

    Args:
        directory_path (str): The path to the directory containing the files.

    Returns:
        list[pathlib.Path]: A combined list of the selected PDF files.
                            Returns an empty list if no matching files or invalid directory.
    """
    logger.info(f"Attempting to filter PDFs in directory: {directory_path}")
    path = Path(directory_path)
    if not path.is_dir():
        logger.error(
            f"Error: The provided path '{directory_path}' is not a valid directory."
        )
        raise ValueError(
            f"Error: The provided path '{directory_path}' is not a valid directory."
        )

    general_series_pattern = re.compile(
        r"([A-Z0-9]+)-\d{4}-([A-Z0-9]+)-([A-Z0-9]+)-([A-Z]+)-(\d{2})", re.IGNORECASE
    )

    categorized_candidates = {}
    other_call_pdfs = []

    try:
        for file_name in os.listdir(path):
            file_path = path / file_name
            lower_file_name = file_name.lower()

            if (
                file_path.is_file()
                and lower_file_name.endswith(".pdf")
                and "call" in lower_file_name
            ):
                match = general_series_pattern.search(file_name)

                if match:
                    program_code = match.group(1).upper()
                    type_name = match.group(2).upper()
                    grant_name = match.group(3).upper()
                    category_name = match.group(4).upper()
                    extracted_number_str = match.group(5)

                    composite_key = (program_code, type_name, grant_name, category_name)

                    try:
                        extracted_number_int = int(extracted_number_str)

                        if composite_key not in categorized_candidates:
                            categorized_candidates[composite_key] = []
                        categorized_candidates[composite_key].append(
                            (extracted_number_int, file_path)
                        )
                    except ValueError:
                        logger.warning(
                            f"Could not convert '{extracted_number_str}' to int from file: {file_name}. Treating as 'other' call PDF."
                        )
                        other_call_pdfs.append(file_path)
                else:
                    other_call_pdfs.append(file_path)
    except Exception as e:
        logger.error(
            f"An error occurred while listing directory contents or processing files: {e}"
        )
        return []

    final_list_of_pdfs = []

    for composite_key, candidates_list in categorized_candidates.items():
        if candidates_list:
            candidates_list.sort(key=lambda x: x[0])
            lowest_xx_pdf = candidates_list[0][1]
            final_list_of_pdfs.append(candidates_list[0][1])
            logger.info(
                f"Selected lowest XX for series {composite_key}: {lowest_xx_pdf.name}"
            )

    final_list_of_pdfs.extend(other_call_pdfs)
    final_list_of_pdfs.sort()  # Ensure consistent output order

    if other_call_pdfs:
        final_list_of_pdfs.extend(other_call_pdfs)
        logger.info(f"Added {len(other_call_pdfs)} 'other' call PDFs.")

    final_list_of_pdfs.sort()
    logger.success(f"Finished filtering. Total PDFs found: {len(final_list_of_pdfs)}")
    return final_list_of_pdfs


@validate_call
def get_series_titles_from_paths(pdf_paths: list[Path]) -> dict[Path, str]:
    """
    Extracts the 'PROGRAMCODE-YEAR-TYPE-GRANT-CATEGORY-XX' title string
    from a list of PDF Path objects.

    Args:
        pdf_paths (list[Path]): A list of pathlib.Path objects of PDF files.

    Returns:
        dict[Path, str]: A dictionary where keys are the original Path objects
                         and values are the extracted title strings.
                         If a path does not match the expected pattern, it's skipped.
    """
    logger.info(f"Attempting to extract series titles from {len(pdf_paths)} PDF paths.")
    titles = {}
    # Re-use the same generalized pattern to extract the title
    # Note: This pattern must match the *exact* segment you want as the title.
    # Group 0 of a match returns the entire matched string.
    general_series_pattern = re.compile(
        r"([A-Z0-9]+)-\d{4}-([A-Z0-9]+)-([A-Z0-9]+)-([A-Z]+)-(\d{2})", re.IGNORECASE
    )

    for pdf_path in pdf_paths:
        file_name = pdf_path.name  # Get just the filename
        match = general_series_pattern.search(file_name)
        if match:
            # match.group(0) returns the entire substring matched by the regex
            titles[pdf_path] = match.group(0)
        else:
            logger.warning(
                f"File '{file_name}' does not match the general series pattern. No specific title extracted."
            )

        # else:
        # If a path from the input list doesn't match the series pattern (e.g., it's an 'other_call_pdf'),
        # it won't have this specific title extracted. You could add logic here
        # to assign a default or indicate it's not applicable if needed.
    logger.success(
        f"Finished title extraction. Extracted titles for {len(titles)} files."
    )
    return titles


# Function to pretty-print document results
def print_document_results_money(doc_name, results):
    print(f"\nResults from {doc_name}:")
    # print(f"Amounts: {[(a.value, a.currency,a.context,a.original_sentence) for a in results['amounts']]}")
    print(results)

    for i, item in enumerate(results["results"]):
        print(f"  Importo {i+1}:")
        value = getattr(item, "value", item.get("value"))
        currency = getattr(item, "currency", item.get("currency"))
        context = getattr(item, "context", item.get("context"))
        original_sentence = getattr(
            item, "original_sentence", item.get("original_sentence")
        )
        print(f"    value: {value}")
        print(f"    currency: {currency}")
        print(f"    context: {context}")
        print(f"    original_sentence: {original_sentence}")


# Function to pretty-print document results
def print_document_results_entity(doc_name, results):
    print(f"\nResults from {doc_name}:")
    # print(f"Amounts: {[(a.value, a.currency,a.context,a.original_sentence) for a in results['amounts']]}")
    print(results)

    for i, item in enumerate(results["results"]):
        print(f"  Importo {i+1}:")
        organization_type = getattr(item, "value", item.get("organization_type"))
        original_sentence = getattr(item, "currency", item.get("original_sentence"))
        context = getattr(item, "context", item.get("context"))
        print(f"    organization_type: {organization_type}")
        print(f"    original_sentence: {original_sentence}")
        print(f"    context: {context}")
