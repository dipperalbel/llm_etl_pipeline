==========================
PdfConverter
==========================

A specialized class for the conversion of PDF documents, leveraging Pydantic
for configuration options management and internally encapsulating a ``DocumentConverter`` instance.
This class provides a streamlined interface for converting PDF documents
into text.

**Example Usage:**

Let's demonstrate how to convert a PDF document to plain text using ``PdfConverter``.

.. code-block:: python

    from llm_etl_pipeline import PdfConverter
    from pathlib import Path
    import os

    dummy_pdf_path = Path("example.pdf")
    converter = PdfConverter()
    extracted_text = converter.convert_to_text(dummy_pdf_path)
    print(extracted_text)

API Reference
--------------

.. autoclass:: llm_etl_pipeline.PdfConverter
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _doc_converter

