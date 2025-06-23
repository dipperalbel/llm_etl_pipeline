.. -*- mode: rst -*-

Document
==========================

A ``Document`` represents a document, capable of storing raw text and/or a structured collection of paragraphs, which can in turn contain sentences.

**Example Usage:**

Let's illustrate how to create a Document and use its methods.

.. code-block:: python

    from llm_etl_pipeline.extraction.public.documents import Document

    # Create a simple document with raw text
    doc_raw_text = Document(
        raw_text="This is the first sentence. This is the second sentence.\n\n"
                 "This is a new paragraph. Another sentence in the same paragraph."
    )

    # Example using get_paras_or_sents_raw_text
    # Get all sentences
    all_sentences = doc_raw_text.get_paras_or_sents_raw_text(reference_depth='sentences')

    # Get paragraphs containing "new paragraph"
    filtered_paragraphs = doc_raw_text.get_paras_or_sents_raw_text(
        regex_pattern="new paragraph",
        reference_depth='paragraphs'
    )

    # Get sentences containing "second"
    filtered_sentences = doc_raw_text.get_paras_or_sents_raw_text(
        regex_pattern="second",
        reference_depth='sentences'
    )


API Reference
-------------

.. autoclass:: llm_etl_pipeline.extraction.Document
   :members: get_paras_or_sents_raw_text, sentences
   :undoc-members: model_post_init
   :show-inheritance:

