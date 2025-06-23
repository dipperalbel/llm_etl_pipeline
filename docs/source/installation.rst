==========================
Installation
==========================

Instructions on how to get your project up and running.

Prerequisites
-------------

* Python >=3.10,<3.14
* poetry

Install using PyPI
------------------

1.  **Install the package:**
    You can install the `llm_etl_pipeline` package directly using pip:

    .. code-block:: bash

        pip install llm_etl_pipeline

Install locally (recommended)
-----------------------------

1.  **Clone the repository:**
    Begin by cloning the project's Git repository to your local machine:

    .. code-block:: bash

        git clone https://github.com/dipperalbel/llm_etl_pipeline
        cd your-project-name

2.  **Install dependencies with Poetry:**
    Navigate into the cloned project directory. Poetry will automatically create a
    virtual environment and install all project dependencies defined in
    `pyproject.toml` (and locked in `poetry.lock`):

    .. code-block:: bash

        poetry install

3.  **Activate the virtual environment (optional, but good practice):**
    While you can run commands directly via `poetry run`, you can also activate
    the virtual environment managed by Poetry:

    .. code-block:: bash

        poetry shell

    Once activated, your terminal prompt might change to indicate the active
    environment. To exit, simply type `exit`.

Ollama Model Installation
-------------------------

**IMPORTANT:** This project requires specific LLM models to be available through
your Ollama installation. After installing Ollama, you need to download
`phi4:14b` and `gemma3:27b` using the Ollama command-line interface:

.. code-block:: bash

    ollama pull phi4:14b
    ollama pull gemma3:27b