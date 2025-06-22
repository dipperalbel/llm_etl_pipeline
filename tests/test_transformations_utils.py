import pandas as pd
import pytest
from pydantic import ValidationError

from llm_etl_pipeline.transformation import load_df_from_json


class TestLoadDfFromJson:

    def test_money_schema_json_content(self):
        df = load_df_from_json("json/test_money_schema.json")

        assert not df.empty
        assert len(df) == 12
        assert set(df.columns) == {
            "value",
            "currency",
            "context",
            "original_sentence",
            "document_id",
        }

        # Test AMIF-2024-TF2-AG-INFO-01 document
        doc1_df = df[df["document_id"] == "AMIF-2024-TF2-AG-INFO-01"].reset_index(
            drop=True
        )
        assert len(doc1_df) == 6
        assert doc1_df["value"].iloc[0] == 10000000.0
        assert doc1_df["currency"].iloc[0] == "EUR"
        assert (
            "min_entities" not in doc1_df.columns
        )  # This format doesn't include min_entities

        # Test AMIF-2024-TF2-AG-THB-01 document
        doc2_df = df[df["document_id"] == "AMIF-2024-TF2-AG-THB-01"].reset_index(
            drop=True
        )
        assert len(doc2_df) == 6
        assert doc2_df["value"].iloc[0] == 6000000.0
        assert doc2_df["currency"].iloc[0] == "EUR"
        assert (
            "min_entities" not in doc2_df.columns
        )  # This format doesn't include min_entities

        # Ensure no 'min_entities' column exists in the final concatenated DataFrame
        assert "min_entities" not in df.columns

    def test_not_str_as_input(self):

        df = load_df_from_json("json/test_entity_schema.json")

        assert not df.empty
        # Total rows: 3 from AMIF-2024-TF2-AG-INFO-01 + 3 from AMIF-2024-TF2-AG-THB-01 = 6
        assert len(df) == 6
        assert set(df.columns) == {"organization_type", "document_id", "min_entities"}

        # Test AMIF-2024-TF2-AG-INFO-01 document
        doc1_df = df[df["document_id"] == "AMIF-2024-TF2-AG-INFO-01"].reset_index(
            drop=True
        )
        assert len(doc1_df) == 3
        assert doc1_df["organization_type"].tolist() == [
            "non-profit",
            "international organisation",
            "for-profit",
        ]
        assert doc1_df["min_entities"].tolist() == [[3], [3], [3]]

        # Test AMIF-2024-TF2-AG-THB-01 document
        doc2_df = df[df["document_id"] == "AMIF-2024-TF2-AG-THB-01"].reset_index(
            drop=True
        )
        assert len(doc2_df) == 3
        assert doc2_df["organization_type"].tolist() == [
            "non-profit",
            "international organisation",
            "for-profit",
        ]
        assert doc2_df["min_entities"].tolist() == [[3], [3], [3]]

    def test_entity_schema_json_content(self):
        df = load_df_from_json("json/test_entity_schema.json")

        assert not df.empty
        # Total rows: 3 from AMIF-2024-TF2-AG-INFO-01 + 3 from AMIF-2024-TF2-AG-THB-01 = 6
        assert len(df) == 6
        assert set(df.columns) == {"organization_type", "document_id", "min_entities"}

        # Test AMIF-2024-TF2-AG-INFO-01 document
        doc1_df = df[df["document_id"] == "AMIF-2024-TF2-AG-INFO-01"].reset_index(
            drop=True
        )
        assert len(doc1_df) == 3
        assert doc1_df["organization_type"].tolist() == [
            "non-profit",
            "international organisation",
            "for-profit",
        ]
        assert doc1_df["min_entities"].tolist() == [[3], [3], [3]]

        # Test AMIF-2024-TF2-AG-THB-01 document
        doc2_df = df[df["document_id"] == "AMIF-2024-TF2-AG-THB-01"].reset_index(
            drop=True
        )
        assert len(doc2_df) == 3
        assert doc2_df["organization_type"].tolist() == [
            "non-profit",
            "international organisation",
            "for-profit",
        ]
        assert doc2_df["min_entities"].tolist() == [[3], [3], [3]]

    def test_not_str_as_input_raises_error(self):
        with pytest.raises(ValidationError) as excinfo:
            df = load_df_from_json(-1)

    def test_none_as_input_raises_error(self):
        with pytest.raises(ValidationError) as excinfo:
            df = load_df_from_json(None)
