"""
This module defines the `Sentence` Pydantic model, designed to represent
an individual sentence as an immutable text unit.

It serves as a foundational building block for structured document analysis,
ensuring the consistency and integrity of textual content throughout the
ETL pipeline. `Sentence` instances are typically generated automatically
by higher-level segmentation processes within the `Document` and `Paragraph` classes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm_etl_pipeline.typings import NonEmptyStr


class Sentence(BaseModel):
    """
    Represents a single sentence within a document or a paragraph.

    `Sentence` instances are designed as **immutable text units**. The `raw_text`
    content is set during initialization and cannot be modified thereafter. This
    immutability is crucial for maintaining data integrity and consistency throughout
    document analysis, ensuring that the fundamental building blocks of text remain
    unchanged.

    Attributes:
        raw_text (NonEmptyStr): The complete, non-empty text content of the sentence.
                                This attribute is **frozen** after initialization,
                                meaning its value cannot be altered.

    Note:
        In most typical workflows, you will not need to manually construct `Sentence`
        objects. They are automatically generated and populated by the `Document`
        class (which segments raw text into paragraphs, and then paragraphs into sentences)
        or by the `Paragraph` class during its internal sentence segmentation.
        This constructor is primarily intended for advanced use cases, such as when
        integrating with a highly customized text segmentation or parsing pipeline.
    """

    raw_text: NonEmptyStr = Field(..., frozen=True)
