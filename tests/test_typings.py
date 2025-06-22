from pathlib import Path
from unittest.mock import patch  # <-- Add this import

import pandas as pd
import pytest
from pydantic import BaseModel, Field, ValidationError

from llm_etl_pipeline.typings import (
    ExtractionType,
    LanguageRequirement,
    NonEmptyDataFrame,
    NonEmptyListStr,
    NonEmptyStr,
    NonZeroInt,
    ReferenceDepth,
    RegexPattern,
    SaTModelId,
)


class MyConfig(BaseModel):
    id: NonZeroInt
    name: NonEmptyStr
    tags: NonEmptyListStr
    data: NonEmptyDataFrame
    pattern: RegexPattern
    ref_depth: ReferenceDepth
    ext_type: ExtractionType
    model_id: SaTModelId
    lang: LanguageRequirement = "en"


class TestCustomTypes:

    def test_non_zero_int(self):
        # Valid cases
        assert (
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            ).id
            == 1
        )
        assert (
            MyConfig(
                id=100,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            ).id
            == 100
        )

        # Invalid cases
        with pytest.raises(ValidationError):
            MyConfig(
                id=0,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            )
        with pytest.raises(ValidationError):
            MyConfig(
                id=-5,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            )
        with pytest.raises(ValidationError):
            MyConfig(
                id="abc",
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            )

    def test_non_empty_str(self):
        # Valid cases
        assert (
            MyConfig(
                id=1,
                name="hello",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            ).name
            == "hello"
        )
        assert (
            MyConfig(
                id=1,
                name="  abc  ",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            ).name
            == "abc"
        )  # Should be stripped

        # Invalid cases
        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            )
        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="   ",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            )
        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name=None,
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            )

    def test_non_empty_list_str(self):
        # Valid cases
        assert MyConfig(
            id=1,
            name="a",
            tags=["item1"],
            data=pd.DataFrame([1]),
            pattern=".",
            ref_depth="paragraphs",
            ext_type="money",
            model_id="sat-1l",
        ).tags == ["item1"]
        assert MyConfig(
            id=1,
            name="a",
            tags=[" item2 ", "item3"],
            data=pd.DataFrame([1]),
            pattern=".",
            ref_depth="paragraphs",
            ext_type="money",
            model_id="sat-1l",
        ).tags == ["item2", "item3"]

        # Invalid cases (list empty)
        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="a",
                tags=[],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            )
        # Invalid cases (list contains empty string)
        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="a",
                tags=[""],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            )
        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="a",
                tags=["  "],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            )
        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="a",
                tags=[1],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            )
        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="a",
                tags=None,
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            )

    def test_non_empty_dataframe(self):
        # Valid cases
        df_non_empty = pd.DataFrame({"col1": [1, 2]})
        assert MyConfig(
            id=1,
            name="a",
            tags=["a"],
            data=df_non_empty,
            pattern=".",
            ref_depth="paragraphs",
            ext_type="money",
            model_id="sat-1l",
        ).data.equals(df_non_empty)

        # Invalid cases (empty DataFrame)
        df_empty = pd.DataFrame()
        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=df_empty,
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            )
        # Invalid cases (not a DataFrame)
        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=[1, 2, 3],
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            )
        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=None,
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            )

    def test_regex_pattern(self):
        # Valid cases
        assert (
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern="^abc$",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            ).pattern
            == "^abc$"
        )
        assert (
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern="\d+",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            ).pattern
            == "\d+"
        )

        # Invalid cases (invalid regex syntax)
        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern="[",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            )
        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern="*",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            )
        # Invalid cases (empty string, handled by NonEmptyStr)
        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern="",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            )

        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=None,
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            )

    def test_reference_depth(self):
        # Valid cases
        assert (
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            ).ref_depth
            == "paragraphs"
        )
        assert (
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="sentences",
                ext_type="money",
                model_id="sat-1l",
            ).ref_depth
            == "sentences"
        )

        # Invalid cases
        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="words",
                ext_type="money",
                model_id="sat-1l",
            )
        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth=123,
                ext_type="money",
                model_id="sat-1l",
            )
        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth=None,
                ext_type="money",
                model_id="sat-1l",
            )

    def test_extraction_type(self):
        # Valid cases
        assert (
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            ).ext_type
            == "money"
        )
        assert (
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="entity",
                model_id="sat-1l",
            ).ext_type
            == "entity"
        )

        # Invalid cases
        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="other",
                model_id="sat-1l",
            )

        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type=None,
                model_id="sat-1l",
            )
        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type=12,
                model_id="sat-1l",
            )

    def test_sat_model_id(self):
        # Valid StandardSaTModelId
        assert (
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
            ).model_id
            == "sat-1l"
        )
        assert (
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-12l-sm",
            ).model_id
            == "sat-12l-sm"
        )

    def test_language_requirement(self):
        # Valid case
        assert (
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
                lang="en",
            ).lang
            == "en"
        )

        # Invalid case
        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
                lang="fr",
            )
        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
                lang=None,
            )
        with pytest.raises(ValidationError):
            MyConfig(
                id=1,
                name="a",
                tags=["a"],
                data=pd.DataFrame([1]),
                pattern=".",
                ref_depth="paragraphs",
                ext_type="money",
                model_id="sat-1l",
                lang=12,
            )
