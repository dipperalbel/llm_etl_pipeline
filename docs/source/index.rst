==========================
ETL Project
==========================

This project operates under the following key assumptions regarding the input PDF files and project structure:

* **Textual PDFs:** All input PDF documents are assumed to be *text-searchable* (i.e., not scanned images). The project relies on the ability to extract raw text content directly from the PDFs. If scanned PDFs are provided, text extraction will fail.
* **English Language Content:** The textual content within all input PDFs is assumed to be primarily in the English language. Text processing and analysis steps (e.g., keyword extraction, natural language processing) may yield inaccurate or irrelevant results for content in other languages.
* **Consistent Document Structure:** Particularly for **"call for proposals"** PDFs, a very similar internal structure and layout are assumed. The project's parsing logic relies on this consistency to accurately locate and extract specific pieces of information. Deviations in structure of the call for proposal PDFs may lead to incomplete or incorrect data extraction.
* **Presence of Call Proposals:** For each EU project intended for processing, it's assumed that a corresponding 'call for proposal' PDF file exists within the designated input folder. This PDF must contain the string "call" in its filename.
* **Handling of Numbered Call Files:** In cases where multiple PDF files exist for the same call, identified by a common naming pattern like ``PROGRAMCODE-YYYY-TYPE-GRANT-CATEGORY-XX`` (e.g., ``AMIF-2025-TF2-AG-INTE-01``, ``AMIF-2025-TF2-AG-INTE-02``), the project will only process the file with the *lowest numerical suffix* (XX). This is due to the assumption that such sequentially numbered files contains identical core information. For example, the call for proposal PDFs for ``AMIF-2025-TF2-AG-INTE-01-WOMEN``, ``AMIF-2025-TF2-AG-INTE-02-HEALTH``,... contain identical core information despite their differing specific extensions.
* **Currency Denomination:** All monetary values (e.g., prices, budgets, grants) mentioned within the PDF documents are assumed to be denominated in **Euros (EUR)**.


.. toctree::
   :caption: Solution explanation
   :maxdepth: 2
   :hidden:

   solution_overview
   project_structure

.. toctree::
    :maxdepth: 2
    :caption: Design Choices and Approach
    :hidden:

    extraction
    transformation
    load

.. toctree::
    :maxdepth: 1
    :caption: Guide
    :hidden:
   
    installation
    usage
    output

.. toctree::
    :maxdepth: 2
    :caption: Api
    :hidden:
   
    extraction/pdfconverter
    extraction/document
    extraction/localllm
    transformation/pipeline
    




