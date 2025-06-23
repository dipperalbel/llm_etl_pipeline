"""
This package provides internal utilities and components for the LLM ETL pipeline's
extraction phase.

It exposes core functionalities such as loading of Sentence-as-Transformer (SaT)
models for advanced text processing, template management for LLM prompts, and a
fallback mechanism for handling extraction failures. Additionally, it includes
specific warning filters for the Python logger to manage expected warnings
during operation.

These modules are intended for internal use within the 'extraction' package
and are exposed via `__all__` for structured internal access.
"""

from llm_etl_pipeline.extraction.internal.filters import _SpecificWarningFilter
from llm_etl_pipeline.extraction.internal.utils import (
    _get_sat_model,
    _get_template,
    _split_text_into_paragraphs,
    _when_all_is_lost,
)

__all__ = [
    "_split_text_into_paragraphs",
    "_get_sat_model",
    "_when_all_is_lost",
    "_get_template",
    "_SpecificWarningFilter",
]
