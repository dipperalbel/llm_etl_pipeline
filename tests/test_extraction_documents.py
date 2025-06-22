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
from llm_etl_pipeline.extraction import Document, Paragraph, Sentence


class TestDocument:
    """
    Test suite for the Document class.
    Organizes all Document-related tests into a single class.
    """

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
        raw_text = "Para 1.\nPara2\n\nPara 3.\n\nPara 4."
        doc = Document(raw_text=raw_text, paragraph_segmentation_mode="empty_line")

        # _split_text_into_paragraphs is mocked to split by '\n\n' (empty_line)
        assert len(doc.paragraphs) == 3
        assert doc.paragraphs[0].raw_text == "Para 1.\nPara2"
        assert doc.paragraphs[1].raw_text == "Para 3."
        assert doc.paragraphs[2].raw_text == "Para 4."

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

    def test_setattr_paragraphs_reassignment_raises_error(self):
        """Prevents paragraphs reassignment once populated."""
        doc = Document(
            raw_text="Test paragraphs reassignment once populated.",
            paragraph_segmentation_mode="newlines",
        )
        with pytest.raises(ValueError) as excinfo:
            doc.paragraphs = []
        assert "The attribute `paragraphs` cannot be changed once populated." in str(
            excinfo.value
        )


class TestParagraph:
    """
    Test suite for the Paragraph class, covering its attributes, immutability,
    and validation logic.
    """

    def test_paragraph_creation_valid_no_sentences(self):
        """
        Test paragraph creation with valid raw_text and no sentences.
        """
        text = "This is a simple paragraph."
        paragraph = Paragraph(raw_text=text)
        assert paragraph.raw_text == text
        assert paragraph.sentences == []

    def test_paragraph_creation_valid_with_sentences(self):
        """
        Test paragraph creation with valid raw_text and a list of sentences.
        """
        text = "First sentence. Second sentence. Third sentence."
        s1 = Sentence(raw_text="First sentence.")
        s2 = Sentence(raw_text="Second sentence.")
        s3 = Sentence(raw_text="Third sentence.")
        sentences = [s1, s2, s3]

        paragraph = Paragraph(raw_text=text, sentences=sentences)
        assert paragraph.raw_text == text
        assert len(paragraph.sentences) == 3
        assert paragraph.sentences[0].raw_text == s1.raw_text
        assert paragraph.sentences[1].raw_text == s2.raw_text
        assert paragraph.sentences[2].raw_text == s3.raw_text

    def test_paragraph_raw_text_immutability(self):
        """
        Test that raw_text is frozen and cannot be modified after creation.
        """
        paragraph = Paragraph(
            raw_text="Original paragraph text.",
            sentences=[Sentence(raw_text="Original paragraph text.")],
        )
        with pytest.raises(
            ValidationError
        ) as excinfo:  # Pydantic raises ValidationError for frozen fields
            paragraph.raw_text = "New text."
        assert "Field is frozen" in str(excinfo.value)
        assert "type=frozen_field" in str(excinfo.value)

    def test_paragraph_sentences_immutability_after_population(self):
        """
        Test that the sentences list cannot be reassigned once populated.
        Appending to the list should still be allowed, but reassignment is not.
        """
        s1 = Sentence(raw_text="Initial sentence.")
        paragraph = Paragraph(raw_text="Initial sentence.", sentences=[s1])

        with pytest.raises(ValueError) as excinfo:
            paragraph.sentences = [Sentence(raw_text="Another sentence.")]
        assert "The attribute `sentences` cannot be changed once populated." in str(
            excinfo.value
        )

    def test_paragraph_creation_empty_raw_text_raises_validation_error(self):
        """
        Test that creating a Paragraph with an empty string for `raw_text` raises ValidationError.
        """
        with pytest.raises(ValidationError) as excinfo:
            Paragraph(raw_text="")
        assert "String should have at least 1 character" in str(excinfo.value)

    def test_paragraph_validation_sentences_not_in_raw_text(self):
        """
        Test that model_validator raises ValueError if a sentence's raw_text is not
        found in the paragraph's raw_text.
        """
        paragraph_text = "This is the actual paragraph."
        # Sentence text not present in paragraph_text
        s1 = Sentence(raw_text="This sentence is not in the paragraph.")
        s2 = Sentence(raw_text="This is the actual paragraph.")  # This one is present

        with pytest.raises(ValueError) as excinfo:
            Paragraph(raw_text=paragraph_text, sentences=[s1, s2])
        assert "Not all sentences were matched in paragraph text." in str(excinfo.value)

    def test_paragraph_validation_sentences_partially_in_raw_text(self):
        """
        Test that model_validator raises ValueError if a sentence's raw_text is
        only partially found or not an exact match, even if parts are present.
        """
        paragraph_text = "The quick brown fox jumps over the lazy dog."
        s1 = Sentence(
            raw_text="quick brown cat"
        )  # 'quick brown' is there, but not 'cat'
        with pytest.raises(ValueError) as excinfo:
            Paragraph(raw_text=paragraph_text, sentences=[s1])
        assert "Not all sentences were matched in paragraph text." in str(excinfo.value)

    def test_paragraph_validation_sentences_exact_match(self):
        """
        Test that model_validator passes when all sentences' raw_text are exact
        substrings of the paragraph's raw_text.
        """
        paragraph_text = "This is sentence one. This is sentence two."
        s1 = Sentence(raw_text="This is sentence one.")
        s2 = Sentence(raw_text="This is sentence two.")
        paragraph = Paragraph(raw_text=paragraph_text, sentences=[s1, s2])
        assert paragraph.raw_text == paragraph_text
        assert len(paragraph.sentences) == 2

    def test_paragraph_string_representation(self):
        """
        Test the string representation (__str__) of the Paragraph object.
        It should return the raw_text.
        """
        text = "Initial sentence."
        s1 = Sentence(raw_text=text)
        paragraph = Paragraph(raw_text=text, sentences=[s1])
        assert text in str(paragraph)

    def test_paragraph_sentences_list_clearing_not_allowed_if_populated(self):
        """
        Test that clearing a populated sentences list by reassigning an empty list
        is *not* allowed by __setattr__.
        """
        s1 = Sentence(raw_text="A sentence.")
        paragraph = Paragraph(raw_text="A sentence.", sentences=[s1])

        with pytest.raises(ValueError) as excinfo:
            paragraph.sentences = (
                []
            )  # Reassigning an empty list to a previously populated list
        assert "The attribute `sentences` cannot be changed once populated." in str(
            excinfo.value
        )


class TestSentence:
    """
    Test suite for the Sentence class.
    Organizes all Sentence-related tests into a single class.
    """

    def test_sentence_creation_valid(self):
        """
        Test that a Sentence instance can be created successfully with valid text.
        """
        text = "This is a valid sentence."
        sentence = Sentence(raw_text=text)
        assert sentence.raw_text == text
        assert isinstance(sentence.raw_text, str)

    def test_sentence_creation_with_special_characters(self):
        """
        Test that a Sentence instance can be created with text containing special characters.
        """
        text = "Hello, world! How are you doing today? 😊"
        sentence = Sentence(raw_text=text)
        assert sentence.raw_text == text

    def test_sentence_immutability(self):
        """
        Test that attempting to modify `raw_text` after initialization raises a ValidationError
        specifically because the field is frozen.
        """
        sentence = Sentence(raw_text="Original text.")
        with pytest.raises(
            ValidationError
        ) as excinfo:  # Changed from TypeError to ValidationError
            sentence.raw_text = "New text."
        assert "Field is frozen" in str(excinfo.value)
        assert "type=frozen_field" in str(excinfo.value)

    def test_sentence_creation_empty_string_raises_validation_error(self):
        """
        Test that creating a Sentence with an empty string for `raw_text` raises ValidationError.
        This checks the `NonEmptyStr` constraint.
        """
        with pytest.raises(ValidationError) as excinfo:
            Sentence(raw_text="")
        # Check that the error message indicates a string length issue
        assert "String should have at least 1 character" in str(excinfo.value)

    def test_sentence_creation_none_raises_validation_error(self):
        """
        Test that creating a Sentence with None for `raw_text` raises ValidationError.
        """
        with pytest.raises(ValidationError) as excinfo:
            # Pydantic will try to coerce, but None cannot be a NonEmptyStr
            Sentence(raw_text=None)
        # Updated assertion to match Pydantic's current error message for None input
        assert "Input should be a valid string" in str(excinfo.value)
        assert "type=string_type" in str(excinfo.value)

    def test_sentence_creation_non_string_raises_validation_error(self):
        """
        Test that creating a Sentence with a non-string type for `raw_text` raises ValidationError.
        """
        with pytest.raises(ValidationError) as excinfo:
            Sentence(raw_text=123)
        assert "value is not a valid string" in str(
            excinfo.value
        ) or "type=string_type" in str(excinfo.value)

    def test_sentence_string_representation(self):
        """
        Test the string representation (__str__) of the Sentence object.
        """
        text = "This is a test sentence for __str__."
        sentence = Sentence(raw_text=text)
        assert text in str(sentence)

    def test_sentence_repr_representation(self):
        """
        Test the developer-friendly representation (__repr__) of the Sentence object.
        Pydantic provides a default __repr__.
        """
        text = "Another sentence."
        sentence = Sentence(raw_text=text)
        # Pydantic's default repr is usually like "Sentence(raw_text='Another sentence.')"
        assert f"Sentence(raw_text='{text}')" in repr(sentence)
