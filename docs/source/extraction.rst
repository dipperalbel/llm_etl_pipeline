Data Extraction Flow and Temporary Storage
=================

The core of this solution for information extraction relies on a multi-stage process leveraging local Large Language Models (LLMs) for specific data points. Our approach prioritizes accuracy and efficiency through a combination of heuristic text processing and targeted LLM inference.

* **PDF to Text Conversion:**
    The process begins with converting the PDF documents into raw text strings. This is handled by the `PdfConverter` class, which internally uses the `docling` package for robust text extraction from PDF files.

* **Text Segmentation - Paragraphs:**
    Following text conversion, the raw text undergoes segmentation into paragraphs using the `Document` class. Although various methods for defining paragraphs were explored—including single newlines (`\n`), empty lines (`\n\n`), and advanced models like `SaT (wtpsplit)` (https://github.com/segment-any-text/wtpsplit) — an heuristic approach based on empty lines (`\n\n`) was ultimately adopted. This method demonstrated superior performance in accurately identifying distinct paragraphs, a performance that is, of course, dependent on the initial text extraction quality from the PDFs.

* **Text Segmentation - Sentences:**
    After paragraph definition, sentences are extracted from each paragraph. For this granular segmentation, the set of `SaT (wtpsplit)` models (https://github.com/segment-any-text/wtpsplit) were employed due to their effectiveness in delineating individual sentences.

* **Information Filtering with Regular Expressions:**
    Before LLM processing, the segmented text (primarily paragraphs, though sentence-level filtering is also an option) undergoes a crucial filtering step using Regular Expressions. These regex patterns were custom-designed based on common characteristics observed in "call for proposal" PDFs to pre-select relevant sections. This includes identifying:
    * **Monetary Amounts:** Strings containing currency indicators (e.g., "EUR") coupled with digits.
    * **Consortium Details:** Sections typically related to consortium formation, specifically looking for the table that indicate minimum number of entities.

* **LLM-based Data Extraction:**
    Once filtered, the relevant paragraphs are fed to the pre-selected local LLMs.
    * **Granular Processing:** To maximize extraction accuracy, particularly for monetary information, paragraphs are inputted to the LLMs in batches rather than providing the entire document at once. This granular approach was observed to yield more precise results (at least for local LLM). For consortium entity extraction, the LLM receives the identified table as its input.
    * **Prompt Engineering:** User and system prompts for the LLMs are dynamically generated using a `Jinja2` template.
    * **Structured Output:** The LLM's raw output is then parsed using a `PydanticJsonParser`. This ensures that the extracted data conforms to a predefined schema, enabling robust validation and easy integration into subsequent processes. However, it is important to note that a well-defined fallback method is currently not in place to handle `ValidationError` instances raised by the parser.
    * **Iterative Accumulation:** This batch processing, prompting, and parsing cycle is repeated for all filtered paragraphs, and the results are accumulated to form the complete extracted dataset for the document.

The entire extraction process described above is repeated for **each individual call for proposal PDF document**. The extracted data from each PDF is then temporarily stored in two separate JSON files: one for monetary information and another for entity-related data.

.. note::

    **Local LLM Models:** We utilized `phi4:14b` primarily for extracting monetary information and `gemma3:27b` for processing consortium-related table data.