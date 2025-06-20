[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/dipperalbel/adfcd8d043e30baf54876431640c2936/raw/coverage.json)](https://github.com/dipperalbel/llm_etl_pipeline/actions)
## Assumptions

This project operates under the following key assumptions regarding the input PDF files and project structure:

* **Textual PDFs:** All input PDF documents are assumed to be *text-searchable* (i.e., not scanned images). The project relies on the ability to extract raw text content directly from the PDFs. If scanned PDFs are provided, text extraction may fail or produce garbled output.
* **English Language Content:** The textual content within all input PDFs is assumed to be primarily in the English language. Text processing and analysis steps (e.g., keyword extraction, natural language processing) may yield inaccurate or irrelevant results for content in other languages.
* **Consistent Document Structure:** Particularly for "call for proposals" PDFs, a very similar internal structure and layout are assumed. The project's parsing logic relies on this consistency to accurately locate and extract specific pieces of information. Deviations in structure may lead to incomplete or incorrect data extraction.
* **Presence of Call Proposals:** For each EU project intended for processing, it's assumed that a corresponding PDF file exists within the designated input folder. This PDF must contain the string "call" in its filename or a prominent location within its text to correctly identify and process it as a call proposal document.
* **Handling of Numbered Call Files:** In cases where multiple PDF files exist for the same call, identified by a common naming pattern like `PROGRAMCODE-YYYY-TYPE-GRANT-CATEGORY-XX` (e.g., `AMIF-2025-TF2-AG-INTE-01`, `AMIF-2025-TF2-AG-INTE-02`), the project will only process the file with the *lowest numerical suffix* (XX). This is due to the assumption that such sequentially numbered files for the same call contain identical core information.
* **Currency Denomination:** All monetary values (e.g., prices, budgets, grants) mentioned within the PDF documents are assumed to be denominated in **Euros (EUR)**.

## Solution Overview

This project focuses on extracting key information from EU project-related PDF documents. During the data extraction process, it was identified that the provided PDFs, particularly those related to grant projects (e.g., AMIF), do not contain specific **Technology Readiness Level (TRL)** information. While TRL is a common concept in Horizon EU projects, the documents only offered generic definitions (TRL 1 to 9) without project-specific details. Attempts to extract TRL data, including leveraging LLM AI models (gemini), proved unsuccessful and led to hallucinations. For AMIF, it is not a practice to indicate TLR.

Consequently, the solution prioritizes the extraction of available and reliable data points:

* **Budget Information:** This includes detailed proposal budget and grant amounts per project, which are consistently and clearly documented within the **"call for proposal" PDFs**.
* **Organization Details:** Extraction of the number and type of organizations involved in grants was also targeted. However, due to ambiguity and lack of clear definitions within the document describing the task regarding what "number and type of organization" specifically entails in the grant context, this aspect could not be fully implemented or clarified through further inquiry.

Given that many of the provided PDFs were found to be templates or contained minimal additional data relevant to the extraction goals, the core focus of this solution was directed exclusively towards processing the "call for proposal" PDFs, as they proved to be the most valuable source of actionable information.

## Design Choices and Approach

The core of this solution for information extraction relies on a multi-stage process leveraging local Large Language Models (LLMs) for specific data points. Our approach prioritizes accuracy and efficiency through a combination of heuristic text processing and targeted LLM inference.

* **Local LLM Models:** We utilized `phi4:14b` primarily for extracting monetary information and `gemma3:27b` for processing consortium-related table data.

* **PDF to Text Conversion:**
    The process begins with converting the PDF documents into raw text strings. This is handled by the `PdfConverter` class, which internally uses the `docling` package for robust text extraction from PDF files.

* **Text Segmentation - Paragraphs:**
    Following text conversion, the raw text is segmented into paragraphs using the `Document` class. While various methods for paragraph definition were explored, including `\n` (single newline), `\n\n` (empty line), and models like `SaT (wtpsplit)` (https://github.com/segment-any-text/wtpsplit), an heuristic approach based on empty lines (`\n\n`) was adopted for its superior performance in accurately identifying distinct paragraph ( this perfomance depends of course on how it is extracted the text from the PDFs).

* **Text Segmentation - Sentences:**
    After paragraph definition, sentences are extracted from each paragraph. For this granular segmentation, the `SaT (wtpsplit)` model (https://github.com/segment-any-text/wtpsplit) was employed due to its effectiveness in delineating individual sentences.

* **Information Filtering with Regular Expressions:**
    Before LLM processing, the segmented text (primarily paragraphs, though sentence-level filtering is also an option) undergoes a crucial filtering step using Regular Expressions. These regex patterns were custom-designed based on common characteristics observed in "call for proposal" PDFs to pre-select relevant sections. This includes identifying:
    * **Monetary Amounts:** Strings containing currency indicators (e.g., "EUR") coupled with digits.
    * **Consortium Details:** Sections typically related to consortium formation, specifically looking for the table that indicate minimum number of entities. 

* **LLM-based Data Extraction:**
    Once filtered, the relevant paragraphs are fed to the pre-selected local LLMs.
    * **Granular Processing:** To maximize extraction accuracy, particularly for monetary information, paragraphs are inputted to the LLMs in batches rather than providing the entire document at once. This granular approach was observed to yield more precise results (at least for local LLM). For consortium entity extraction, the LLM receives the identified table as its input.
    * **Prompt Engineering:** User and system prompts for the LLMs are dynamically generated using a `Jinja2` template.
    * **Structured Output:** The LLM's raw output is then parsed using a `PydanticJsonParser`. This ensures that the extracted data conforms to a predefined schema, enabling robust validation and easy integration into subsequent processes. However, there is not a well defined fallback method in case of ValidationError caused by the parser.
    * **Iterative Accumulation:** This batch processing, prompting, and parsing cycle is repeated for all filtered paragraphs, and the results are accumulated to form the complete extracted dataset for the document.
 
### Data Extraction Flow and Temporary Storage

The entire extraction process described above is repeated for **each individual PDF document**. The extracted data from each PDF is then temporarily stored in two separate JSON files: one for monetary information and another for entity-related data.

### Data Transformation

Following the extraction phase, the temporarily stored JSON files are loaded into `pandas` DataFrames for subsequent transformation and consolidation. This stage is crucial for refining the extracted raw data:

* **Monetary Data Transformation:**
    Extracted monetary data undergoes various validation checks to ensure its quality and adherence to expected conditions. We then filter and retain only the information most relevant for analysis, such as the grant requested per project, available call budgets, or specific budget allocations per topic as mentioned in the call for proposal.
    A significant challenge identified was the **duplication of monetary values**, where identical amounts might or might not refer to the same underlying entity or concept (e.g., two mentions of the minimum EU grant request). To address this, a specific deduplication strategy is employed:
    1.  Sentences containing the duplicate monetary amounts are converted into embeddings using `Sentence-Transformers (S-BERT)`.
    2.  Hierarchical clustering is then performed on these embeddings using cosine distance.
    3.  If multiple sentences fall within the same cluster (indicating high semantic similarity for the same amount), the sentence with the *longest text* is selected as the representative for that cluster, simplifying the data while retaining context.

* **Entity Data Transformation:**
    For the extracted entity data, due to time constraints, the transformation primarily involves basic validation checks followed by a simple stacking of the dtaa extracted that contained the entity information.

* **Pipeline Orchestration:** All these transformation steps are orchestrated via a `Pipeline` class, which applies a series of pre-written functions sequentially to the `pandas` DataFrames, streamlining the data processing workflow.
 
### Data Load

Finally, the processed `pandas` DataFrames for monetary and organization type information are stored as CSV files: `etl_money_result.csv` for the monetary data, and `etl_entity_result.csv` for the entity data.

## Testing Strategy

For quality assurance, a suite of unit tests has been developed to validate individual components and functions of the codebase. While these tests provide foundational coverage, the current test coverage stands at approximately 50%, indicating areas for future expansion.

## Installation

Instructions on how to get your project up and running.

### Prerequisites

List any software, libraries, or tools that need to be installed before your project can run. Examples:
- Python 3.x
- pip (usually comes with Python)
- virtualenv 
- poetry (recommended)

### Install using PyPI

1.  **Install the package:**
    You can install the `llm_etl_pipeline` package directly using pip:
    ```bash
    pip install llm_etl_pipeline
    ```
### Install locally (recommended)

1. **Clone the repository:**
    Begin by cloning the project's Git repository to your local machine:
    ```bash
    git clone https://github.com/dipperalbel/llm_etl_pipeline
    cd your-project-name
    ```

2.  **Install dependencies with Poetry:**
    Navigate into the cloned project directory. Poetry will automatically create a virtual environment and install all project dependencies defined in `pyproject.toml` (and locked in `poetry.lock`):
    ```bash
    poetry install
    ```
    
3.  **Activate the virtual environment (optional, but good practice):**
    While you can run commands directly via `poetry run`, you can also activate the virtual environment managed by Poetry:
    ```bash
    poetry shell
    ```
    Once activated, your terminal prompt might change to indicate the active environment. To exit, simply type `exit`.

### Ollama Model Installation

**IMPORTANT:** This project requires specific LLM models to be available through your Ollama installation. After installing Ollama, you need to download `phi4:14b` and `gemma3:27b` using the Ollama command-line interface:

```bash
ollama pull phi4:14b
ollama pull gemma3:27b
