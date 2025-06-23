"""
This module defines the `PdfConverter` Pydantic model, providing a robust
interface for converting PDF documents to text.

It encapsulates the `docling` library's `DocumentConverter` to offer
configurable options for OCR, table structure detection, and cell matching
during the conversion process. This class ensures consistent PDF processing
within the LLM ETL pipeline, with integrated logging for clarity and error handling.
"""

import logging
from pathlib import Path
from typing import Any, Union

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from pydantic import BaseModel, Field, PrivateAttr, validate_call

from llm_etl_pipeline.customized_logger import logger
from llm_etl_pipeline.extraction.internal import _SpecificWarningFilter

# Get the logger instance that Docling is using
# This is used to ignore a specific warning message from docling.
docling_specific_logger = logging.getLogger("docling_core.types.doc.document")
SPECIFIC_MESSAGE_TO_IGNORE = (
    "Parameter `strict_text` has been deprecated and will be ignored."
)
my_filter = _SpecificWarningFilter(SPECIFIC_MESSAGE_TO_IGNORE)
docling_specific_logger.addFilter(my_filter)


class PdfConverter(BaseModel):
    """
    A specialized class for the conversion of PDF documents, leveraging Pydantic
    for configuration options management and internally encapsulating a
    `DocumentConverter` instance.

    This class provides a streamlined interface for converting PDF documents
    into text, with configurable options for OCR, table structure detection,
    and cell matching during the conversion process.

    Attributes:
        do_ocr (bool):
            Indicates whether Optical Character Recognition (OCR) should be
            performed on the PDF document. Defaults to `False`.
            This field is `frozen=True`.
        do_table_structure (bool):
            Indicates whether to detect table structures within the PDF.
            Defaults to `True`. This field is `frozen=True`.
        do_cell_matching (bool):
            Indicates whether to perform cell matching for detected tables.
            Defaults to `False`. This field is `frozen=True`.
        _doc_converter (DocumentConverter):
            A private attribute holding the internal instance of `DocumentConverter`,
            configured based on the Pydantic fields.
    """

    # Pydantic fields for configuration options
    do_ocr: bool = Field(
        default=False,
        description="Indicates whether to perform OCR on the PDF document.",
        strict=True,
        frozen=True,
    )
    do_table_structure: bool = Field(
        default=True,
        description="Indicates whether to detect table structures.",
        strict=True,
        frozen=True,
    )
    do_cell_matching: bool = Field(
        default=False,
        description="Indicates whether to perform cell matching for tables.",
        strict=True,
        frozen=True,
    )
    # Private attribute for the DocumentConverter instance
    _doc_converter: DocumentConverter = PrivateAttr()

    def __init__(self, **data: Any):  # Added Any type hint for clarity
        """
        Constructor for the PdfConverter class.

        Initializes the Pydantic model with the provided data and then
        configures the internal `DocumentConverter` instance based on these
        settings.

        Args:
            **data (Any): Arbitrary keyword arguments corresponding to the
                          Pydantic fields (e.g., `do_ocr`, `do_table_structure`, `do_cell_matching`).
        """
        super().__init__(**data)  # Call to BaseModel constructor

        # Configure the internal DocumentConverter based on Pydantic fields
        self._configure_document_converter()

    def _configure_document_converter(self) -> None:
        """
        Configures the internal `DocumentConverter` instance based on the
        Pydantic fields of this model.

        This private method creates a `PdfPipelineOptions` object and populates
        it with the `do_ocr`, `do_table_structure`, and `do_cell_matching`
        settings defined in this `PdfConverter` instance. It then initializes
        the `_doc_converter` with these specific PDF format options.
        """
        pdf_pipeline_options = PdfPipelineOptions()
        pdf_pipeline_options.do_ocr = self.do_ocr
        pdf_pipeline_options.do_table_structure = self.do_table_structure
        pdf_pipeline_options.table_structure_options.do_cell_matching = (
            self.do_cell_matching
        )

        self._doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
            }
        )

    @validate_call
    def convert_to_text(self, input_pdf_path: Union[Path, str, DocumentStream]) -> str:
        """
        Converts a PDF document to plain text using the internal `DocumentConverter` instance.

        This method takes various forms of PDF input (file path, string path,
        or document stream) and utilizes the pre-configured `DocumentConverter`
        to perform the conversion.

        Args:
            input_pdf_path (Union[Path, str, DocumentStream]): The path to the input
                PDF document (as a `pathlib.Path` object or string), or a
                `DocumentStream` object representing the PDF content.

        Returns:
            str: The extracted plain text content from the converted PDF document.

        Raises:
            Exception: Re-raises any exception encountered during the conversion process
                       by the underlying `DocumentConverter`. An error message is also logged.
        """
        logger.info(f"Attempting to convert PDF to text from input: {input_pdf_path}")
        try:
            # Call the 'convert' method on the internal instance
            converted_document = self._doc_converter.convert(input_pdf_path)
            result_text = converted_document.document.export_to_text()
            logger.success("PDF successfully converted to text.")
            return result_text
        except Exception as e:
            logger.error(
                f"Error during PDF conversion for input '{input_pdf_path}': {e}"
            )
            raise
