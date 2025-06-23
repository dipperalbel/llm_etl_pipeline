"""
This module defines the `Paragraph` Pydantic model, which represents
a fundamental textual segment within a document.

It is designed to hold the raw text content of a paragraph and an ordered
collection of `Sentence` objects. The `Paragraph` model enforces immutability
for its `raw_text` and prevents reassignment of its `sentences` list once populated,
ensuring data integrity for text segmentation and analysis.
"""

from __future__ import annotations

import sys
from typing import Any, List, TypeVar

from pydantic import BaseModel, Field, model_validator

from llm_etl_pipeline.extraction.public.sentences import (  # Assuming Sentence is defined and imported
    Sentence,
)
from llm_etl_pipeline.typings import (  # Assuming NonEmptyStr is defined and imported
    NonEmptyStr,
)

# For Python < 3.11, Self needs to be imported this way
if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = TypeVar("Self")


class Paragraph(BaseModel):
    """
    Represents a fundamental text segment of a document, containing its raw text
    content and an ordered collection of constituent sentences.

    Paragraph instances are designed to be **immutable text segments** at the `raw_text`
    level. Once the `raw_text` is set during initialization, it cannot be changed.
    Similarly, once the `sentences` list is populated (i.e., not empty), it cannot
    be reassigned to maintain data integrity and consistency throughout analysis.

    Attributes:
        raw_text (NonEmptyStr): The complete, non-empty text content of the paragraph.
                                This attribute is **frozen** after initialization,
                                meaning its value cannot be altered.
        sentences (list[Sentence]): A list of `Sentence` objects that are part of this
                                    paragraph. Defaults to an empty list. This list
                                    cannot be reassigned once it contains elements.

    Note:
        Typically, you won't need to manually construct `Paragraph` objects. They are
        automatically generated and populated by the `Document` class during its
        text segmentation process. Use this constructor primarily for advanced scenarios,
        such as when integrating a custom paragraph segmentation tool.
    """

    raw_text: NonEmptyStr = Field(..., frozen=True)
    sentences: List[Sentence] = Field(default_factory=list)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Custom attribute setter that applies restrictions on specific attributes.

        This method specifically prevents the `sentences` attribute from being
        reassigned if it has already been set to a non-empty (truthy) value.
        This mechanism ensures the immutability of the paragraph's segmented
        sentences after they've been established.

        Args:
            name (str): The name of the attribute to set.
            value (Any): The value to assign to the attribute.

        Raises:
            ValueError: If an attempt is made to reassign the `sentences` attribute
                        after it has already been populated with elements.
        """
        if name in ["sentences"]:
            # Prevent sentences reassignment once populated, to prevent inconsistencies in analysis.
            if getattr(self, name, None):
                raise ValueError(
                    f"The attribute `{name}` cannot be changed once populated."
                )
        super().__setattr__(name, value)

    @model_validator(mode="after")
    def _validate_paragraph_post(self) -> Self:
        """
        Pydantic validator that runs after initial model validation.

        This validator ensures data integrity by verifying that every sentence
        listed in the `sentences` attribute (if the list is not empty) has its
        raw text content entirely contained within the paragraph's `raw_text`.

        Returns:
            Self: The validated `Paragraph` instance.

        Raises:
            ValueError: If any sentence's raw text content is not found within
                        the paragraph's `raw_text`, indicating a mismatch or
                        segmentation error.
        """
        if self.sentences:
            if not all(i.raw_text in self.raw_text for i in self.sentences):
                raise ValueError("Not all sentences were matched in paragraph text.")
        return self
