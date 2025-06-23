"""
This module defines a comprehensive set of custom type aliases using Pydantic's
`Annotated` types and `Literal` for robust data validation and clear type hinting
across the LLM ETL pipeline.

These aliases enforce specific constraints such as non-emptiness for strings
and DataFrames, positive integer values, valid regex syntax, and predefined
literal choices for categories like reference depth, extraction types, and
SaT model identifiers. They serve to enhance type safety and improve code
readability throughout the project.
"""

from pathlib import Path
from typing import Annotated, Literal, Union

import pandas as pd
from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    InstanceOf,
    StrictInt,
    StrictStr,
    StringConstraints,
)

from llm_etl_pipeline.typings.internal.validators import (
    _ensure_dataframe_type,
    _validate_non_empty_dataframe,
    _validate_regex_syntax,
)

# --- Type Definitions ---

NonZeroInt = Annotated[StrictInt, Field(ge=1)]
"""
Represents an int that must be greater or equal to 1.
"""

NonEmptyStr = Annotated[
    StrictStr, StringConstraints(strip_whitespace=True, min_length=1)
]
"""
Represents a string that must not be empty after stripping whitespace.
"""

NonEmptyListStr = Annotated[
    list[NonEmptyStr],
    Field(min_length=1, description="A list of strings that must not be empty."),
]
"""
Represents a list of strings where each string is non-empty and the list itself contains at least one item.
"""

NonEmptyDataFrame = Annotated[
    InstanceOf[pd.DataFrame],
    BeforeValidator(_ensure_dataframe_type),
    AfterValidator(_validate_non_empty_dataframe),
]
"""
Represents a pandas DataFrame that must not be None and must contain at least one row.
"""

RegexPattern = Annotated[NonEmptyStr, AfterValidator(_validate_regex_syntax)]
"""
Represents a non-empty string that must be a valid regular expression pattern.
"""

ReferenceDepth = Literal["paragraphs", "sentences"]
"""
Defines the valid literal values for specifying reference depth: 'paragraphs' or 'sentences'.
"""

ExtractionType = Literal["money", "entity"]
"""
Defines the valid literal values for specifying the extraction type: 'money' or 'tlr'.
"""


# Define standard SaT model IDs as a separate type
StandardSaTModelId = Literal[
    "sat-1l",
    "sat-1l-sm",
    "sat-3l",
    "sat-3l-sm",
    "sat-6l",
    "sat-6l-sm",
    "sat-9l",
    "sat-12l",
    "sat-12l-sm",
]
"""
Defines the literal values for standard SaT (Semantic Augmentation Tool) model identifiers.
These typically refer to pre-trained models with varying complexities (e.g., number of layers).
"""


# Combined type for sat_model_id parameter
SaTModelId = Union[
    StandardSaTModelId,
    str,  # Local path as a string
    Path,  # Local path as a Path object
]
"""
Represents a SaT (Semantic Augmentation Tool) model identifier, which can be:
- A predefined standard model ID (e.g., 'sat-1l').
- A string representing a local file path to a custom model.
- A `Path` object representing a local file path to a custom model.
"""

LanguageRequirement = Literal["en"]
"""
Defines the valid literal values for language requirements, currently restricted to 'en' (English).
"""
