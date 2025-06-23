import itertools
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, validate_call  # Combined Pydantic imports

from llm_etl_pipeline.customized_logger import logger

# Assuming these are utility functions from their respective modules
from llm_etl_pipeline.extraction.internal import (
    _get_sat_model,
    _split_text_into_paragraphs,
)

# Assuming these are specific classes from their respective modules
from llm_etl_pipeline.extraction.public.paragraphs import Paragraph
from llm_etl_pipeline.extraction.public.sentences import Sentence

# Assuming these are from your 'typings' package (as per previous discussions)
from llm_etl_pipeline.typings import (
    NonEmptyStr,
    ReferenceDepth,
    RegexPattern,
    SaTModelId,
)


class Document(BaseModel):
    """
    Represents a document, capable of storing raw text and/or a structured collection
    of paragraphs, which can in turn contain sentences.

    This class provides functionalities for segmenting text into paragraphs and sentences,
    and for retrieving filtered content based on regular expressions and reference depth.

    Attributes:
        raw_text (Optional[NonEmptyStr]): The raw, unsegmented text of the document.
                                           Cannot be reassigned once set to a truthy value.
        paragraphs (list[Paragraph]): A list of Paragraph objects representing the
                                      segmented content of the document.
                                      Cannot be reassigned once populated.
        paragraph_segmentation_mode (Literal["newlines", 'empty_line', "sat"]):
            The method used for segmenting raw_text into paragraphs.
            - "newlines": Segments by newline characters.
            - "empty_line": Segments by empty lines.
            - "sat": Uses a SaT (Semantic Augmentation Tool) model for segmentation.
            Defaults to "empty_line".
        sat_model_id (SaTModelId): The identifier for the SaT model to be used for
                                   segmentation (both paragraphs and sentences, if "sat" mode is used).
                                   Defaults to "sat-3l-sm".
        regex_pattern (Optional[NonEmptyStr]): A regular expression pattern that can
                                               be used for filtering paragraphs or sentences.
                                               Defaults to None.
    """

    raw_text: Optional[NonEmptyStr] = Field(default=None)
    paragraphs: list[Paragraph] = Field(default_factory=list)
    paragraph_segmentation_mode: Literal["newlines", "empty_line", "sat"] = Field(
        default="empty_line"
    )
    sat_model_id: SaTModelId = Field(default="sat-3l-sm")
    regex_pattern: Optional[NonEmptyStr] = Field(default=None)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Sets the attribute of an instance, with additional restrictions on specific attributes.

        Prevents `raw_text` and `paragraphs` from being reassigned once they have
        been initially set to a non-empty (truthy) value, to maintain data consistency
        and prevent accidental modification after initial segmentation.

        Args:
            name (str): The name of the attribute to set.
            value (Any): The value to assign to the attribute.

        Raises:
            ValueError: If attempting to reassign 'raw_text' or 'paragraphs'
                        after it has already been assigned to a *truthy* value.
        """
        if name in ["raw_text", "paragraphs"]:
            # Prevent raw_text/paragraphs reassignment once populated, to prevent inconsistencies in analysis.
            if getattr(self, name, None):
                logger.error(
                    f"Attempted to reassign '{name}' after it was already populated. This operation is not allowed."
                )
                raise ValueError(
                    f"The attribute `{name}` cannot be changed once populated."
                )
        super().__setattr__(name, value)

    @property
    def sentences(self) -> list[Sentence]:
        """
        Provides access to all sentences within the paragraphs of the document by flattening
        and combining sentences from each paragraph into a single list.

        This property iterates through all `Paragraph` objects in the `paragraphs` list
        and chains their respective `sentences` lists together.

        Returns:
            list[Sentence]: A list of all `Sentence` objects contained within all paragraphs.
        """
        return list(itertools.chain.from_iterable(i.sentences for i in self.paragraphs))

    def model_post_init(self, __context: Any) -> None:
        """
        Pydantic hook that runs after model initialization and validation.

        This method orchestrates the segmentation of raw text into paragraphs and
        subsequently, sentences within those paragraphs, if not already provided.
        """
        self._segment_paras_and_sents()

    @validate_call
    def get_paras_or_sents_raw_text(
        self,
        regex_pattern: Optional[RegexPattern] = None,
        reference_depth: ReferenceDepth = "sentences",
    ) -> list[str]:
        """
        Retrieves raw text content from either sentences or paragraphs, optionally filtered by a regex pattern.

        The method compiles the provided regex (or uses a wildcard regex if none is provided)
        and then filters the raw text of sentences or paragraphs based on whether they match
        the pattern.

        Args:
            regex_pattern (Optional[RegexPattern]): An optional regular expression pattern
                                                    to filter the text items. If None,
                                                    all items' raw text will be returned.
            reference_depth (ReferenceDepth): Specifies whether to retrieve sentences
                                              or paragraphs. Must be "sentences" or "paragraphs".
                                              Defaults to "sentences".

        Returns:
            list[str]: A list of raw text strings from the filtered sentences or paragraphs.
        """
        logger.info(
            f"Retrieving raw text for '{reference_depth}' with regex pattern: '{regex_pattern}'"
        )

        # Compile regex for performance if used many times
        if reference_depth == "sentences":
            text_items = self.sentences
        else:  # reference_depth == 'paragraphs'
            text_items = self.paragraphs

        # Use DOTALL to make '.' match newlines as well
        compiled_regex = (
            re.compile(regex_pattern, re.IGNORECASE | re.DOTALL)
            if regex_pattern
            else re.compile(r".", re.DOTALL)
        )

        # Filter the results using a single list comprehension
        filtered_result = [
            item.raw_text for item in text_items if compiled_regex.search(item.raw_text)
        ]
        logger.success(f"Found {len(filtered_result)} matching text items.")
        return filtered_result

    def _segment_paras_and_sents(self) -> None:
        """
        Segments the document's raw text into paragraphs and, subsequently,
        the paragraphs into sentences, if not already provided.

        Paragraph segmentation is determined by `paragraph_segmentation_mode`.
        Sentence segmentation within paragraphs always uses the SaT model.

        Actions performed:
        1. If `raw_text` is present but `paragraphs` are not, it extracts paragraphs
           based on `paragraph_segmentation_mode` ("newlines", "empty_line", or "sat").
           It then assigns these newly created `Paragraph` objects to `self.paragraphs`.
           A validation step ensures all segmented paragraphs are present in the original `raw_text`.
        2. If `paragraphs` exist but some do not have extracted sentences, it extracts
           sentences for those paragraphs using the SaT model identified by `sat_model_id`.
           Existing sentences in paragraphs are preserved.

        Raises:
            ValueError: If `raw_text` is provided but no valid paragraphs can be extracted.
            ValueError: If `paragraph_segmentation_mode` is an invalid value.
            AssertionError: If segmented sentences are not found within their respective paragraph texts.
        """
        if self.raw_text and not self.paragraphs:
            # Extract paragraphs from text, if text provided without paragraphs
            logger.info(
                "Text is being split into paragraphs, as no custom paragraphs were provided..."
            )
            if (
                self.paragraph_segmentation_mode == "newlines"
                or self.paragraph_segmentation_mode == "empty_line"
            ):
                paragraphs: list[str] = _split_text_into_paragraphs(
                    self.raw_text, self.paragraph_segmentation_mode
                )
            elif self.paragraph_segmentation_mode == "sat":
                logger.info(
                    f"Using SaT model '{self.sat_model_id}' for paragraph segmentation."
                )
                paragraphs_nested_list: list[list[str]] = _get_sat_model(
                    self.sat_model_id
                ).split(
                    self.raw_text,
                    do_paragraph_segmentation=True,
                )
                paragraphs = ["".join(i) for i in paragraphs_nested_list]
            else:
                logger.error(
                    f"Invalid paragraph segmentation mode specified: {self.paragraph_segmentation_mode}"
                )
                raise ValueError(
                    f"Invalid paragraph segmentation mode: {self.paragraph_segmentation_mode}"
                )

            if not paragraphs:
                raise ValueError("No valid paragraphs in text.")

            # Assign paragraphs on the document
            paragraphs_obj: list[Paragraph] = [
                Paragraph(raw_text=i) for i in paragraphs
            ]

            # Check that each paragraph is found in the document text
            # For duplicate paragraphs, verify each occurrence is matched in the document
            remaining_text = self.raw_text
            for paragraph in paragraphs_obj:
                if paragraph.raw_text not in remaining_text:
                    logger.critical(
                        f"Segmented paragraph '{paragraph.raw_text[:20]}...' not found in remaining raw text. Data inconsistency detected."
                    )
                    raise ValueError(
                        "Not all segmented paragraphs were matched in document text."
                    )
                # Remove the first occurrence to handle duplicates correctly
                remaining_text = remaining_text.replace(paragraph.raw_text, "", 1)
            self.paragraphs = paragraphs_obj

        if self.paragraphs:
            # Extract sentences for each paragraph without sentences provided
            if not all(i.sentences for i in self.paragraphs):
                logger.info("Paragraphs are being split into sentences...")
                if any(i.sentences for i in self.paragraphs):
                    logger.warning(
                        "Some paragraphs already have sentences. "
                        "These will be used `as is`."
                    )
                split_sents_for_paras = _get_sat_model(self.sat_model_id).split(
                    [p.raw_text for p in self.paragraphs]
                )
                for paragraph, sent_group in zip(
                    self.paragraphs, split_sents_for_paras
                ):
                    if not paragraph.sentences:
                        # Filter out empty sents, if any
                        sent_group = [i.strip() for i in sent_group]
                        sent_group = [i for i in sent_group if len(i)]
                        if not all(i in paragraph.raw_text for i in sent_group):
                            raise ValueError(
                                "Not all segmented sentences were matched in paragraph text."
                            )
                        paragraph.sentences = [
                            Sentence(
                                raw_text=i
                            )  # inherit custom data and additional context from paragraph object
                            for i in sent_group
                        ]
        logger.success(f"Generated paragraphs form the raw text.")
