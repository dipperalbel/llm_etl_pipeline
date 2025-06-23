from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypeVar, get_args

from jinja2 import Environment, Template, nodes
from wtpsplit_lite import SaT

from llm_etl_pipeline.customized_logger import logger

# Assuming these are from your 'typings' package (as per previous discussions)
from llm_etl_pipeline.typings import SaTModelId, StandardSaTModelId

T = TypeVar("T")


def _get_template(
    template_name: str,
    template_type: Literal["prompt", "system"] = "prompt",
    template_extension: Literal["j2", "txt"] = "j2",
) -> Template | str:  # Using Union for explicit compatibility with Python < 3.10
    """
    Retrieves and optionally processes a template from a predefined file path.

    The function constructs the template's file path based on `template_type`,
    reads its content, and then either compiles it into a Jinja2 `Template` object
    (for ".j2" files) or returns it as a plain string (for ".txt" files).
    It includes validation steps for ".j2" templates to ensure structural integrity.

    Args:
        template_name (str): The base name of the template file (e.g., "my_template").
                             Do not include the file extension.
        template_type (Literal["prompt", "system"], optional): Specifies the subdirectory
            from which to load the template.
            - "prompt": Loads from the "template/prompts" subdirectory.
            - "system": Loads from the "template/system" subdirectory.
            Defaults to "prompt".
        template_extension (Literal["j2", "txt"], optional): The file extension of the template,
            determining its format and how it's processed.
            - "j2": The file is treated as a Jinja2 template and compiled using `_setup_jinja2_template`.
                    Requires balanced brackets and no excessive newlines.
            - "txt": The file content is returned as a raw string.
            Defaults to "j2".

    Returns:
        Union[Template, str]: The loaded template.
            - If `template_extension` is "j2", a compiled Jinja2 `Template` object.
            - If `template_extension` is "txt", the plain string content of the file.

    Raises:
        NotImplementedError:
            - If an unsupported `template_type` is provided.
            - If an unsupported `template_extension` is provided.
        AssertionError:
            - If a ".j2" template's brackets are not balanced (`_are_prompt_template_brackets_balanced` fails).
            - If a ".j2" template contains too many consecutive newlines (more than two).
            - If the read template text is empty.
    """

    current_file = Path(__file__).resolve()
    project_root = current_file.parents[
        0
    ]  # Assumes templates are in a folder sibling to the current file's folder

    if template_type == "prompt":
        template_path = (
            project_root
            / "template"
            / "prompts"
            / f"{template_name}.{template_extension}"
        )
    elif template_type == "system":
        template_path = (
            project_root
            / "template"
            / "system"
            / f"{template_name}.{template_extension}"
        )
    else:
        raise NotImplementedError(f"Unknown template type: {template_type}")

    with open(template_path, "r", encoding="utf-8") as file:
        template_text = file.read().strip()
        if not template_text:
            raise ValueError("Template text is empty.")

    if template_extension == "j2":
        # Validate template text
        brackets_balanced = _are_prompt_template_brackets_balanced(template_text)
        if not brackets_balanced:
            raise ValueError("Prompt template brackets are not balanced.")

        if bool(re.search(r"(\r\n|\r|\n){3,}", template_text)):
            raise ValueError("Too many newlines in template.")
        template = _setup_jinja2_template(template_text)
    elif template_extension == "txt":
        template = template_text
    else:
        raise NotImplementedError(
            f"Unsupported template extension: {template_extension}"
        )
    return template


def _are_prompt_template_brackets_balanced(prompt: str) -> bool:
    """
    Checks if all opening brackets ('[' and '{') in a prompt template have
    corresponding matching closing brackets (']' and '}') in the correct order.

    This validation is specifically designed for **prompt templates** to ensure
    structural integrity, particularly for JSON-like object or array structures
    embedded within the template.

    Important:
        This function should **only** be used on unrendered prompt templates.
        It is NOT suitable for use on a fully rendered prompt, as a rendered prompt
        may contain arbitrary user-submitted text with unbalanced or non-matching
        bracket combinations that are valid within natural language but would
        cause this function to incorrectly return `False`.

    Args:
        prompt (str): The text content of the prompt template to be validated.

    Returns:
        bool: `True` if all relevant brackets are balanced and correctly nested;
              `False` otherwise.
    """
    stack = []
    brackets = {
        "]": "[",
        "}": "{",
    }  # Closing brackets mapped to their counterparts

    for char in prompt:
        if char in "[{":  # If opening bracket, push onto stack
            stack.append(char)
        elif char in "]}":  # If closing bracket
            if (
                not stack or stack[-1] != brackets[char]
            ):  # Check for matching opening bracket
                return False
            stack.pop()  # Pop the matching opening bracket off the stack

    return not stack  # If stack is empty, all brackets were matched


def _setup_jinja2_template(template_text: str) -> Template:
    """
    Creates and configures a Jinja2 template from the provided text.

    This function performs the necessary steps to prepare a Jinja2 `Template` object:
    1. Validates that the input `template_text` contains at least one Jinja2 tag,
       ensuring it's a dynamic template.
    2. Instantiates the `Template` object, applying options to trim and lstrip blocks
       for cleaner output.
    3. Sets up essential global functions, such as `enumerate` and `_clean_text_for_llm_prompt`,
       making them directly available for use within the template.

    Args:
        template_text (str): The raw text content of the template. This string
                             is expected to contain Jinja2-specific syntax (e.g.,
                             `{{ variable }}`, `{% for %}`, `{# comment #}`).

    Returns:
        Template: A fully configured Jinja2 `Template` object, ready to be rendered.

    Raises:
        ValueError: If the `template_text` does not contain any detected Jinja2 tags,
                    indicating it might be plain static text.
    """

    if not _contains_jinja2_tags(template_text):
        raise ValueError("Template contains no Jinja2 tags.")

    # Create the Template object with appropriate options
    template = Template(template_text, trim_blocks=True, lstrip_blocks=True)

    # Set up global functions
    template.globals["enumerate"] = enumerate
    template.globals["_clean_text_for_llm_prompt"] = _clean_text_for_llm_prompt

    return template


def _contains_jinja2_tags(text: str) -> bool:
    """
    Determines if a given string contains Jinja2 template tags or expressions.

    This function leverages Jinja2's internal parsing mechanism to analyze the
    abstract syntax tree (AST) of the input text. It identifies whether the text
    contains any dynamic elements, such as variables (`{{ ... }}`), control
    structures (`{% ... %}`), or comments (`{# ... #}`), beyond mere static text.

    Args:
        text (str): The input string to check for Jinja2 template tags.

    Returns:
        bool: `True` if the text contains Jinja2 tags or dynamic content;
              `False` otherwise (i.e., if it's purely static text).
    """
    env = Environment(autoescape=True)
    parsed = env.parse(text)
    # If any node in the top-level body is not TemplateData (and isn't an Output
    # wrapping only TemplateData), it indicates the presence of Jinja2 tags,
    # i.e. dynamic content.
    for node in parsed.body:
        if isinstance(node, nodes.Output):
            if not all(isinstance(child, nodes.TemplateData) for child in node.nodes):
                return True
        elif not isinstance(node, nodes.TemplateData):
            return True
    return False


def _setup_jinja2_template(template_text: str) -> Template:
    """
    Creates and configures a Jinja2 template from the provided text.

    This function prepares a Jinja2 `Template` object by ensuring it contains
    dynamic Jinja2 tags, applying common formatting options (like stripping
    whitespace around blocks), and registering essential global functions
    that can be used within the template.

    Args:
        template_text (str): The raw text content that will be used to create
                             the Jinja2 template.

    Returns:
        Template: A fully configured Jinja2 `Template` object, ready for rendering.

    Raises:
        ValueError: If the `template_text` does not contain any recognizable
                    Jinja2 tags (e.g., `{{ variable }}`, `{% block %}`, `{# comment #}`).
    """

    if not _contains_jinja2_tags(template_text):
        raise ValueError("Template contains no Jinja2 tags.")

    # Create the Template object with appropriate options
    template = Template(template_text, trim_blocks=True, lstrip_blocks=True)

    # Set up global functions
    template.globals["enumerate"] = enumerate
    template.globals["_clean_text_for_llm_prompt"] = _clean_text_for_llm_prompt

    return template


def _clean_text_for_llm_prompt(raw_text: str, preserve_linebreaks: bool = True) -> str:
    """
    Cleans raw text by removing problematic characters and normalizing whitespace,
    making it suitable for input into Large Language Models (LLMs).

    This function addresses common text hygiene issues such as control characters,
    zero-width spaces, and inconsistent linebreaks, which can interfere with LLM
    processing or tokenization.

    Args:
        raw_text (str): The input string to be cleaned.
        preserve_linebreaks (bool, optional): If `True`, newline characters (`\n`)
            are preserved and normalized, while other whitespace is collapsed.
            If `False`, all whitespace (including newlines) is collapsed into
            a single space. Defaults to `True`.

    Returns:
        str: The cleaned and formatted version of the input text.
    """

    if preserve_linebreaks:
        # Normalize newlines to \n
        cleaned = re.sub(r"\r\n|\r", "\n", raw_text)

        # Remove control characters EXCEPT newlines (\n = ASCII 10)
        # This includes:
        # - ASCII control characters except LF (0x00-0x09, 0x0B-0x1F and 0x7F)
        # - Zero-width characters
        # - Bidirectional text markers
        # - Other invisible unicode characters
        cleaned = re.sub(
            r"[\x00-\x09\x0B-\x1F\x7F-\x9F\u200B-\u200F\u2028-\u202F\uFEFF]",
            "",
            cleaned,
        )

        # Replace horizontal whitespace sequences (spaces and tabs) with a single space
        # while preserving linebreaks
        cleaned = re.sub(r"[ \t]+", " ", cleaned)

        # Remove extra blank lines (more than one consecutive newline)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    else:
        # Remove all control characters including newlines
        cleaned = re.sub(
            r"[\x00-\x1F\x7F-\x9F\u200B-\u200F\u2028-\u202F\uFEFF]", "", raw_text
        )

        # Remove all whitespace sequences with a single space
        cleaned = re.sub(r"\s+", " ", cleaned)

    # Strip leading/trailing whitespace
    return cleaned.strip()


def _split_text_into_paragraphs(
    raw_text: str, paragraph_segmentation_mode: str
) -> list[str]:
    """
    Splits a given raw text into a list of paragraphs based on the specified segmentation mode.

    Empty paragraphs (after stripping whitespace) are filtered out from the result.

    Args:
        raw_text (str): The input raw text string to be segmented.
        paragraph_segmentation_mode (str): The strategy to use for splitting the text.
            Accepted values are:
            - "newlines": Splits the text by any occurrence of one or more newline characters
                          (e.g., '\n', '\r', '\r\n').
            - "empty_line": Splits the text by two or more consecutive newline characters,
                            effectively treating empty lines as paragraph separators.

    Returns:
        list[str]: A list of non-empty strings, where each string represents a paragraph.

    Raises:
        ValueError: If `paragraph_segmentation_mode` is not "newlines" or "empty_line".
    """
    if paragraph_segmentation_mode == "newlines":
        # Split by one or more newline characters (CR, LF, or CRLF)
        paragraphs = re.split(r"[\r\n]+", raw_text)
    elif paragraph_segmentation_mode == "empty_line":
        # Split by two or more consecutive newline characters (CR, LF, or CRLF)
        paragraphs = re.split(r"[\r\n]{2,}", raw_text)
    else:
        raise ValueError(
            f"Invalid paragraph_segmentation_mode: '{paragraph_segmentation_mode}'. "
            "Expected 'newlines' or 'empty_line'."
        )

    # Strip leading/trailing whitespace from each paragraph and filter out empty strings
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return paragraphs


@lru_cache(maxsize=3)
def _get_sat_model(model_id: SaTModelId = "sat-3l-sm") -> SaT:
    """
    Retrieves and caches a SaT (Semantic Augmentation Tool) model for text segmentation.

    This function first validates the `model_id` to determine if it refers to a
    standard pre-trained model or a local path. It then attempts to load the
    corresponding SaT model, handling different error scenarios.

    Args:
        model_id (SaTModelId):
            The identifier for the SaT model to be loaded. Defaults to "sat-3l-sm".
            This can be:
            - A string corresponding to a predefined standard SaT model ID (e.g., "sat-3l-sm").
            - A string or Path object pointing to a local directory containing a custom
              SaT model.

    Returns:
        SaT: An instance of the loaded SaT model.

    Raises:
        ValueError:
            If `model_id` is a local path that does not exist or is not a directory.
        RuntimeError:
            If `model_id` refers to a local path that exists but the directory
            does not contain a valid SaT model, or if there's any other error
            during the model loading process.
    """
    # Convert Path object to string if needed
    if isinstance(model_id, Path):
        model_id = str(model_id)

    # Check if it's a standard model ID
    is_standard_model = False
    if isinstance(model_id, str):
        # Get standard models directly from the type definition
        standard_models = get_args(StandardSaTModelId)
        is_standard_model = model_id in standard_models

    # Determine if it's a local path (but not a standard model ID)
    is_local_path = False
    if isinstance(model_id, str) and not is_standard_model:
        path = Path(model_id)

        # Validate that the path exists and is a directory
        if not path.exists() or not path.is_dir():
            raise ValueError(
                f"The provided SaT model path '{model_id}' does not exist or is not a directory."
            )

        is_local_path = True

    # Attempt to load the model
    try:
        model = SaT(model_id)
        return model
    except Exception as e:
        if is_local_path:
            # If it's a local path that exists but isn't a valid SaT model
            raise RuntimeError(
                f"The directory at '{model_id}' exists but does not contain a valid SaT model. "
                f"Error: {str(e)}"
            ) from e
        else:
            # For standard model IDs or other errors (e.g., download failed, invalid ID)
            # This covers cases where SaT(model_id) fails for non-local paths.
            raise RuntimeError(
                f"Failed to load SaT model '{model_id}'. Error: {str(e)}"
            ) from e


def _when_all_is_lost(inputs: Any) -> Any:
    """Fallback function to execute when the primary chain fails."""
    logger.warning(
        "LLM EXTRACTION FAILED!!"
    )  # TO-DO: Use a logger for the error message.
    return inputs
