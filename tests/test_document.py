import itertools
from typing import (  # Necessario per Literal nel mock_split_text_into_paragraphs
    Any,
    Literal,
)
from unittest.mock import (  # MagicMock potrebbe ancora essere usato per altri scopi se necessario
    create_autospec,
    patch,
)

import pytest
from pydantic import BaseModel, Field, ValidationError

# Import the Document class from its new assumed location
from llm_etl_pipeline.extraction import Document

# Import necessary types for type hinting and test data
from llm_etl_pipeline.typings import (
    NonEmptyStr,
    ReferenceDepth,
    RegexPattern,
    SaTModelId,
)

# --- Pytest Fixtures for Patching ---


class TestDocument:

    # Test __setattr__ behavior
    def test_setattr_raw_text_initial_assignment(self):
        """Allows initial assignment of raw_text."""
        doc = Document()
        doc.raw_text = "Some text."
        assert doc.raw_text == "Some text."

    def test_setattr_raw_text_reassignment_raises_error(self):
        """Prevents raw_text reassignment once populated."""
        doc = Document(raw_text="Initial text.")
        with pytest.raises(ValueError) as excinfo:
            doc.raw_text = "New text."
        assert "The attribute `raw_text` cannot be changed once populated." in str(
            excinfo.value
        )

    # Test _segment_paras_and_sents (model_post_init hook)
    def test_segment_paras_and_sents_from_raw_text_newlines(
        self,
    ):  # Removed mock_external_dependencies as arg
        """Tests paragraph and sentence segmentation from raw_text using 'newlines' mode."""
        raw_text = "Para 1.\nPara 2.\nPara 3."
        doc = Document(raw_text=raw_text, paragraph_segmentation_mode="newlines")

        # _split_text_into_paragraphs is mocked to split by '\n'
        assert len(doc.paragraphs) == 3
        assert doc.paragraphs[0].raw_text == "Para 1."
        assert doc.paragraphs[1].raw_text == "Para 2."
        assert doc.paragraphs[2].raw_text == "Para 3."

        # Le asserzioni sul logger sono state rimosse come richiesto

    def test_segment_paras_and_sents_from_raw_text_empty_line(
        self,
    ):  # Removed mock_external_dependencies as arg
        """Tests paragraph and sentence segmentation from raw_text using 'empty_line' mode."""
        raw_text = "Para 1.\n\nPara 2.\n\nPara 3."
        doc = Document(raw_text=raw_text, paragraph_segmentation_mode="empty_line")

        # _split_text_into_paragraphs is mocked to split by '\n\n' (empty_line)
        assert len(doc.paragraphs) == 3
        assert doc.paragraphs[0].raw_text == "Para 1."
        assert doc.paragraphs[1].raw_text == "Para 2."
        assert doc.paragraphs[2].raw_text == "Para 3."

    def test_segment_paras_and_sents_from_raw_text_sat_mode(
        self,
    ):  # Removed mock_external_dependencies as arg
        """Tests paragraph and sentence segmentation from raw_text using 'sat' mode."""
        raw_text = "Text for SAT paragraph split. Second paragraph here."
        doc = Document(raw_text=raw_text, paragraph_segmentation_mode="sat")

        # _get_sat_model.split should have been called with do_paragraph_segmentation=True
        assert len(doc.paragraphs) == 2
        assert doc.paragraphs[0].raw_text == "Text for SAT paragraph split."
        assert doc.paragraphs[1].raw_text == "Second paragraph here."

    def test_segment_paras_and_sents_no_raw_text_and_no_paragraphs(self):
        """Tests that no segmentation happens if no raw_text and no paragraphs provided."""
        doc = Document()
        assert doc.raw_text is None
        assert doc.paragraphs == []
