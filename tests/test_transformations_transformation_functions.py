import re  # Needed for regex testing

import numpy as np  # Needed for np.nan
import pandas as pd
import pytest
from pydantic import ValidationError  # For testing Pydantic validations

# Import necessary functions and types
from llm_etl_pipeline.transformation import (
    drop_rows_if_no_column_matches_regex,
    group_by_document_and_stack_types,
    reduce_list_ints_to_unique,
    verify_list_column_contains_only_ints,
)
from llm_etl_pipeline.typings import (
    NonEmptyDataFrame,
    NonEmptyListStr,
    NonEmptyStr,
    RegexPattern,
)


# --- Tests for drop_rows_if_no_column_matches_regex ---
# Fixture for a non-empty sample DataFrame
@pytest.fixture
def sample_dataframe_general() -> NonEmptyDataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "text_col": ["apple", "banana", "orange", "grape", "kiwi"],
            "num_col": [10, 20, 30, 40, 50],
            "list_col": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
            "mixed_list_col": [[1, "a"], [2, 3], [4], [None], [5]],
            "nullable_col": ["A", None, "C", "D", "E"],
            "non_string_col": ["str", 123, "another_str", 4.5, True],
            "document_id": ["doc1", "doc1", "doc2", "doc2", "doc3"],
            "entity_type": ["PERSON", "ORG", "LOC", "PERSON", "ORG"],
            "min_entities": [[1, 2, 1], [1, 2, 1], [3, 4, 3], [3, 4, 3], [5, 6]],
        }
    )


# --- Tests for drop_rows_if_no_column_matches_regex ---
class TestDropRowsIfNoColumnMatchesRegex:

    def test_keeps_all_matching_rows(self, sample_dataframe_general: NonEmptyDataFrame):
        """All rows have at least one match, no row is dropped."""
        # Modified df to ensure all rows genuinely contain 'a' in at least one column
        df = pd.DataFrame(
            {
                "colA": ["apple", "banana", "grape"],  # Changed 'cherry' to 'grape'
                "colB": ["fruit", "yellow", "data"],  # Changed 'red' to 'data'
            }
        )
        columns = ["colA", "colB"]
        regex = r"a"  # 'a' is present in all rows in colA or colB

        result_df = drop_rows_if_no_column_matches_regex(df, columns, regex)

        # DEBUG PRINT: What is result_df actually? (Keeping for user's own debug if needed)
        import builtins  # Use builtins.print to ensure it's not mocked by pytest's capture

        builtins.print("\n--- DEBUG: result_df in test_keeps_all_matching_rows ---")
        builtins.print(result_df)
        builtins.print("--- END DEBUG ---\n")

        # Explicitly define expected_df to rule out any re-indexing issues
        expected_df = pd.DataFrame(
            {"colA": ["apple", "banana", "grape"], "colB": ["fruit", "yellow", "data"]}
        )
        # Ensure the index is identical, as df.loc preserves original index
        expected_df.index = df.index

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_drops_non_matching_rows(self, sample_dataframe_general: NonEmptyDataFrame):
        """Some rows have no match and are dropped."""
        df = pd.DataFrame(
            {"col1": ["apple", "orange", "grape"], "col2": ["banana", "kiwi", "pear"]}
        )
        columns = ["col1", "col2"]
        regex = r"orange|kiwi"  # Only row 1 (orange, kiwi) should remain

        result_df = drop_rows_if_no_column_matches_regex(df, columns, regex)
        expected_df = pd.DataFrame(
            {"col1": ["orange"], "col2": ["kiwi"]}, index=[1]
        )  # Original index is 1

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_case_insensitivity_and_dotall(
        self, sample_dataframe_general: NonEmptyDataFrame
    ):
        """Tests that regex is case-insensitive and DOTALL."""
        df = pd.DataFrame(
            {"description": ["Hello world", "another Line", "TEST dotall"]}
        )
        columns = ["description"]
        regex_case_insensitive = r"hello"  # 'Hello'
        regex_dotall = r"TEST.dotall"  # 'TEST dotall'

        result_df_case = drop_rows_if_no_column_matches_regex(
            df, columns, regex_case_insensitive
        )
        pd.testing.assert_frame_equal(
            result_df_case, df.iloc[[0]]
        )  # Only 'Hello world' should remain

        result_df_dotall = drop_rows_if_no_column_matches_regex(
            df, columns, regex_dotall
        )
        pd.testing.assert_frame_equal(
            result_df_dotall, df.iloc[[2]]
        )  # Only 'TEST dotall' should remain

    def test_multiple_columns_and_mixed_matches(
        self, sample_dataframe_general: NonEmptyDataFrame
    ):
        """Test with multiple columns and mixed matches."""
        df = pd.DataFrame(
            {"colA": ["abc", "def", "ghi", "jkl"], "colB": ["xyz", "uvw", "abc", "mno"]}
        )
        columns = ["colA", "colB"]
        regex = r"abc"

        # Rows 0 (colA), 2 (colB) should have matches. Row 1 has no match, row 3 has no match.
        result_df = drop_rows_if_no_column_matches_regex(df, columns, regex)
        expected_df = pd.DataFrame(
            {"colA": ["abc", "ghi"], "colB": ["xyz", "abc"]}, index=[0, 2]
        )
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_column_not_found_raises_error(
        self, sample_dataframe_general: NonEmptyDataFrame
    ):
        """Raises ValueError if a specified column is not found."""
        with pytest.raises(ValueError) as excinfo:
            drop_rows_if_no_column_matches_regex(
                sample_dataframe_general, ["non_existent_col"], r"abc"
            )
        assert "Column 'non_existent_col' not found" in str(excinfo.value)

    def test_column_contains_null_values_raises_error(
        self, sample_dataframe_general: NonEmptyDataFrame
    ):
        """Raises ValueError if a column contains null values."""
        with pytest.raises(ValueError) as excinfo:
            drop_rows_if_no_column_matches_regex(
                sample_dataframe_general, ["nullable_col"], r"A"
            )
        assert "contains 'None' or missing values" in str(excinfo.value)

    def test_column_contains_non_string_elements_raises_error(
        self, sample_dataframe_general: NonEmptyDataFrame
    ):
        """Raises ValueError if a column contains non-string elements."""
        with pytest.raises(ValueError) as excinfo:
            drop_rows_if_no_column_matches_regex(
                sample_dataframe_general, ["non_string_col"], r"str"
            )
        assert "contains non-string elements" in str(excinfo.value)

    def test_column_containing_nan_raises_error(
        self, sample_dataframe_general: NonEmptyDataFrame
    ):
        """Tests that a ValueError is raised if a target column contains NaN values,
        as the function does not skip such columns but requires non-null strings."""
        df_test_nan_col = pd.DataFrame(
            {
                "col_with_data": ["A", "B", "C"],
                "target_col_with_nan": [np.nan, "test", "data"],  # Contains NaN
            }
        )
        columns = ["target_col_with_nan", "col_with_data"]
        regex = r"A"

        with pytest.raises(ValueError) as excinfo:
            drop_rows_if_no_column_matches_regex(df_test_nan_col, columns, regex)
        assert "contains 'None' or missing values" in str(excinfo.value)
        # Verify the specific index where NaN is found
        assert "[0]" in str(excinfo.value)


# --- Tests for verify_list_column_contains_only_ints ---
class TestVerifyListColumnContainsOnlyInts:

    def test_valid_list_of_ints_column(
        self, sample_dataframe_general: NonEmptyDataFrame
    ):
        """Tests that a column with lists of only integers passes validation."""
        df = pd.DataFrame({"data": [[1, 2], [3], [4, 5, 6]]})
        result_df = verify_list_column_contains_only_ints(df, ["data"])
        pd.testing.assert_frame_equal(result_df, df)

    def test_valid_multiple_list_of_ints_columns(
        self, sample_dataframe_general: NonEmptyDataFrame
    ):
        """Tests multiple columns with lists of only integers."""
        df = pd.DataFrame({"list1": [[1, 2], [3]], "list2": [[10], [20, 30]]})
        result_df = verify_list_column_contains_only_ints(df, ["list1", "list2"])
        pd.testing.assert_frame_equal(result_df, df)

    def test_column_not_found_raises_error(
        self, sample_dataframe_general: NonEmptyDataFrame
    ):
        """Raises ValueError if a column is not found."""
        with pytest.raises(ValueError) as excinfo:
            verify_list_column_contains_only_ints(
                sample_dataframe_general, ["non_existent_list_col"]
            )
        assert "Column 'non_existent_list_col' not found" in str(excinfo.value)

    def test_column_contains_none_raises_error(
        self, sample_dataframe_general: NonEmptyDataFrame
    ):
        """Raises ValueError if a column contains None."""
        df = pd.DataFrame({"data": [[1], None, [2]]})
        with pytest.raises(ValueError) as excinfo:
            verify_list_column_contains_only_ints(df, ["data"])
        assert "contains a missing value (NaN/None) at index 1" in str(excinfo.value)

    def test_column_contains_nan_raises_error(
        self, sample_dataframe_general: NonEmptyDataFrame
    ):
        """Raises ValueError if a column contains np.nan."""
        df = pd.DataFrame({"data": [[1], np.nan, [2]]})
        with pytest.raises(ValueError) as excinfo:
            verify_list_column_contains_only_ints(df, ["data"])
        assert "contains a missing value (NaN/None) at index 1" in str(excinfo.value)

    def test_column_contains_non_list_elements_raises_error(
        self, sample_dataframe_general: NonEmptyDataFrame
    ):
        """Raises ValueError if a column contains non-list elements."""
        df = pd.DataFrame({"data": [[1], "not a list", [2]]})
        with pytest.raises(ValueError) as excinfo:
            verify_list_column_contains_only_ints(df, ["data"])
        assert "is not a list" in str(excinfo.value)

    def test_list_contains_non_integer_elements_raises_error(
        self, sample_dataframe_general: NonEmptyDataFrame
    ):
        """Raises ValueError if a list contains non-integer elements."""
        df = pd.DataFrame({"data": [[1, 2], [3, "a"], [4]]})
        with pytest.raises(ValueError) as excinfo:
            verify_list_column_contains_only_ints(df, ["data"])
        assert "is not an integer. Found value: a" in str(excinfo.value)

    # Removed test_empty_column_is_skipped as it was attempting to test unreachable code
    # given the NonEmptyDataFrame input type.


# --- Tests for reduce_list_ints_to_unique ---
class TestReduceListIntsToUnique:

    def test_removes_duplicates_preserves_order(
        self, sample_dataframe_general: NonEmptyDataFrame
    ):
        """Tests that duplicates are removed while preserving order."""
        df = pd.DataFrame({"col": [[1, 2, 1, 3], [4, 5, 4], [6, 6, 6, 7]]})
        expected_df = pd.DataFrame({"col": [[1, 2, 3], [4, 5], [6, 7]]})
        result_df = reduce_list_ints_to_unique(df, "col")
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_handles_none_and_nan(self, sample_dataframe_general: NonEmptyDataFrame):
        """Tests handling of None and NaN values."""
        df = pd.DataFrame({"col": [[1, 1], None, [2, 3, 2], np.nan]})
        expected_df = pd.DataFrame({"col": [[1], None, [2, 3], np.nan]})
        result_df = reduce_list_ints_to_unique(df, "col")
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_empty_lists(self, sample_dataframe_general: NonEmptyDataFrame):
        """Tests with empty lists."""
        df = pd.DataFrame({"col": [[], [1, 1], []]})
        expected_df = pd.DataFrame({"col": [[], [1], []]})
        result_df = reduce_list_ints_to_unique(df, "col")
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_column_not_found_raises_error(
        self, sample_dataframe_general: NonEmptyDataFrame
    ):
        """Raises ValueError if the target column is not found."""
        with pytest.raises(ValueError) as excinfo:
            reduce_list_ints_to_unique(sample_dataframe_general, "non_existent_col")
        assert "Column 'non_existent_col' not found" in str(excinfo.value)

    def test_non_list_element_raises_type_error(
        self, sample_dataframe_general: NonEmptyDataFrame
    ):
        """Raises TypeError if an element is not a list, None, or NaN."""
        df = pd.DataFrame({"col": [[1], "not a list", [2]]})
        with pytest.raises(TypeError) as excinfo:
            reduce_list_ints_to_unique(df, "col")
        assert (
            "Value in column 'col' must be a list, None, or NaN. Found: <class 'str'>"
            in str(excinfo.value)
        )


# --- Tests for group_by_document_and_stack_types ---
class TestGroupByDocumentAndStackTypes:

    def test_basic_grouping_and_stacking(
        self, sample_dataframe_general: NonEmptyDataFrame
    ):
        """Tests basic grouping and stacking."""
        df = pd.DataFrame(
            {
                "document_id": ["docA", "docA", "docB", "docB", "docC"],
                "target": ["tag1", "tag2", "tag3", "tag1", "tag4"],
                "min_entities": [[1], [1], [2], [2], [3]],
            }
        )

        result_df = group_by_document_and_stack_types(df, "target")

        expected_df = pd.DataFrame(
            {
                "document_id": ["docA", "docB", "docC"],
                "target": [["tag1", "tag2"], ["tag3", "tag1"], ["tag4"]],
                "min_entities": [[1], [2], [3]],
            }
        )
        # For comparing lists of lists, pd.testing.assert_frame_equal is robust
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_unique_target_values_stacked(
        self, sample_dataframe_general: NonEmptyDataFrame
    ):
        """Ensures that stacked target values are unique and preserve order."""
        df = pd.DataFrame(
            {
                "doc_id": ["d1", "d1", "d1"],
                "type": ["A", "B", "A"],
                "min_ents": [[1], [1], [1]],
            }
        )
        result_df = group_by_document_and_stack_types(
            df,
            target_column="type",
            document_id_column="doc_id",
            min_entities_column="min_ents",
        )
        expected_df = pd.DataFrame(
            {
                "doc_id": ["d1"],
                "type": [["A", "B"]],  # A should appear only once
                "min_ents": [[1]],
            }
        )
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_configurable_column_names(
        self, sample_dataframe_general: NonEmptyDataFrame
    ):
        """Tests that column names are configurable."""
        df = pd.DataFrame(
            {
                "my_doc_id": ["d1", "d1", "d2"],
                "my_extracted_type": ["E1", "E2", "E3"],
                "my_min_vals": [[10], [10], [20]],
            }
        )

        result_df = group_by_document_and_stack_types(
            df,
            target_column="my_extracted_type",
            document_id_column="my_doc_id",
            min_entities_column="my_min_vals",
        )

        expected_df = pd.DataFrame(
            {
                "my_doc_id": ["d1", "d2"],
                "my_extracted_type": [["E1", "E2"], ["E3"]],
                "my_min_vals": [[10], [20]],
            }
        )
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_required_column_not_found_raises_error(
        self, sample_dataframe_general: NonEmptyDataFrame
    ):
        """Raises ValueError if a required column is not found."""
        with pytest.raises(ValueError) as excinfo:
            group_by_document_and_stack_types(
                sample_dataframe_general,
                target_column="non_existent_target",
                document_id_column="document_id",
                min_entities_column="min_entities",
            )
        assert "Required column 'non_existent_target' not found" in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            group_by_document_and_stack_types(
                sample_dataframe_general,
                target_column="entity_type",
                document_id_column="non_existent_doc_id",
                min_entities_column="min_entities",
            )
        assert "Required column 'non_existent_doc_id' not found" in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            group_by_document_and_stack_types(
                sample_dataframe_general,
                target_column="entity_type",
                document_id_column="document_id",
                min_entities_column="non_existent_min_entities",
            )
        assert "Required column 'non_existent_min_entities' not found" in str(
            excinfo.value
        )

    def test_empty_dataframe_input(self):
        """Tests behavior with an empty input DataFrame (should fail due to NonEmptyDataFrame)."""
        empty_df = pd.DataFrame(columns=["document_id", "target", "min_entities"])
        with pytest.raises(
            ValidationError
        ) as excinfo:  # NonEmptyDataFrame should catch it
            group_by_document_and_stack_types(empty_df, "target")
        # Corrected assertion message to match Pydantic's NonEmptyDataFrame validator
        assert "DataFrame must contain at least one row (i.e., not be empty)." in str(
            excinfo.value
        )
