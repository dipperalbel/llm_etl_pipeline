"""
This module aggregates and exposes essential classes and functions for:
- Representing structured document components (Document, Paragraph, Sentence).
- Interacting with local Large Language Models (LocalLLM).
- Handling PDF document conversions (PdfConverter).
- Parsing specific information types (MonetaryInformation, ConsortiumComposition).
- Providing utility functions for document series management.
"""

from llm_etl_pipeline.extraction.public.converters import PdfConverter
from llm_etl_pipeline.extraction.public.documents import Document
from llm_etl_pipeline.extraction.public.localllms import LocalLLM
from llm_etl_pipeline.extraction.public.paragraphs import Paragraph
from llm_etl_pipeline.extraction.public.parsers.entities import (
    ConsortiumComposition,
    ConsortiumParticipant,
)
from llm_etl_pipeline.extraction.public.parsers.monetary_informations import (
    MonetaryInformation,
    MonetaryInformationList,
)
from llm_etl_pipeline.extraction.public.sentences import Sentence
from llm_etl_pipeline.extraction.public.utils import (
    get_filtered_fully_general_series_call_pdfs,
    get_series_titles_from_paths,
)

__all__ = [
    "Document",
    "Paragraph",
    "Sentence",
    "LocalLLM",
    "MonetaryInformation",
    "MonetaryInformationList",
    "PdfConverter",
    "ConsortiumComposition",
    "ConsortiumParticipant",
    "get_filtered_fully_general_series_call_pdfs",
    "get_series_titles_from_paths",
]
