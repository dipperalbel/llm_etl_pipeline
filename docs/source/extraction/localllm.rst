==========================
LocalLLM
==========================

A specialized LangChain `ChatOllama` model designed for local execution,
incorporating a default system prompt and a Pydantic output parser for
structured data extraction.

This class extends `ChatOllama` to provide predefined system instructions
and handle structured output parsing, streamlining interactions with the
local LLM for specific extraction tasks.

**Example Usage:**

This example demonstrates how to initialize ``LocalLLM`` and use its
``extract_information`` method to extract monetary information from text.

.. code-block:: python

    from llm_etl_pipeline.extraction import LocalLLM

    llm_extractor = LocalLLM(
        model="llama3",  # Replace with the name of your Ollama model
        temperature=0.3, # Keep temperature low for deterministic extraction
        default_system_prompt="You are a helpful assistant designed to extract information."
    )

    # Example text elements for extraction
    text_elements = [
        "The total cost was $150.75, with an additional fee of 20 USD.",
        "He paid 5 euros for coffee.",
        "The price increased by £10.",
        "She received 100 JPY from the exchange."
    ]

    print("\nAttempting to extract monetary information...")
    extracted_data = llm_extractor.extract_information(
        list_elem=text_elements,
        extraction_type='money',
        reference_depth='sentences'
    )

    print("\n--- Extracted Monetary Information ---")
    import json
    print(json.dumps(extracted_data, indent=2))
    print("--------------------------------------")



API Reference
--------------

.. autoclass:: llm_etl_pipeline.LocalLLM
   :members:
   :undoc-members:
   :show-inheritance:
