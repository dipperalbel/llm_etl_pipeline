import re  # Needed for regex testing

import numpy as np  # Needed for np.nan
import pandas as pd
import pytest
from pydantic import ValidationError  # For testing Pydantic validations

# Import necessary functions and types
from llm_etl_pipeline.transformation import (
    check_columns_satisfy_regex,
    check_numeric_columns,
    check_string_columns,
    drop_rows_if_no_column_matches_regex,
    drop_rows_not_satisfying_regex,
    drop_rows_with_non_positive_values,
    group_by_document_and_stack_types,
    reduce_list_ints_to_unique,
    verify_list_column_contains_only_ints,
    verify_no_empty_strings,
    verify_no_missing_data,
    verify_no_negatives,
)
from llm_etl_pipeline.typings import NonEmptyDataFrame


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


@pytest.fixture
def sample_non_empty() -> NonEmptyDataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "text_col": ["apple", "banana", "orange", "grape", "kiwi"],
            "num_col": [10, 20, 30, 40, 50],
            "list_col": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
            "document_id": ["doc1", "doc1", "doc2", "doc2", "doc3"],
            "entity_type": ["PERSON", "ORG", "LOC", "PERSON", "ORG"],
            "min_entities": [[1, 2, 1], [1, 2, 1], [3, 4, 3], [3, 4, 3], [5, 6]],
        }
    )


class TestDropRowsWithNonPositiveValues:
    """
    Test suite for the 'drop_rows_with_non_positive_values' function.
    """

    def test_no_non_positive_values(self):
        """
        Test case to verify that no rows are dropped when the specified columns
        contain only positive values.
        """
        input_df = NonEmptyDataFrame(
            {"A": [1, 2, 3], "B": [10, 20, 30], "C": [100, 200, 300]}
        )
        columns_to_check = ["A", "B"]
        expected_df = input_df.copy()
        result_df = drop_rows_with_non_positive_values(input_df, columns_to_check)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_single_non_positive_value(self):
        """
        Test case to ensure that a single row with a non-positive value (zero)
        in a specified column is correctly dropped.
        """
        input_df = NonEmptyDataFrame(
            {"A": [1, 0, 3], "B": [10, 20, 30], "C": [100, 200, 300]}
        )
        columns_to_check = ["A"]
        expected_df = NonEmptyDataFrame(
            {"A": [1, 3], "B": [10, 30], "C": [100, 300]}, index=[0, 2]
        )
        result_df = drop_rows_with_non_positive_values(input_df, columns_to_check)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_multiple_non_positive_values_single_column(self):
        """
        Test case to check if multiple rows containing non-positive values
        (negative and zero) in a single specified column are all dropped.
        """
        input_df = NonEmptyDataFrame({"A": [1, -5, 3, 0, 7], "B": [10, 20, 30, 40, 50]})
        columns_to_check = ["A"]
        expected_df = NonEmptyDataFrame(
            {"A": [1, 3, 7], "B": [10, 30, 50]}, index=[0, 2, 4]
        )
        result_df = drop_rows_with_non_positive_values(input_df, columns_to_check)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_column_not_found(self):
        """
        Test case to verify that a ValueError is raised when a specified column
        in 'columns_to_check' does not exist in the DataFrame.
        """
        input_df = NonEmptyDataFrame({"A": [1, 2], "B": [3, 4]})
        columns_to_check = ["C"]
        with pytest.raises(ValueError, match="not found"):
            drop_rows_with_non_positive_values(input_df, columns_to_check)

    def test_column_with_null_values(self):
        """
        Test case to confirm that a ValueError is raised when a specified column
        contains null (None or NaN) values.
        """
        input_df = NonEmptyDataFrame({"A": [1, None, 3], "B": [10, 20, 30]})
        columns_to_check = ["A"]
        with pytest.raises(ValueError, match="'None'"):
            drop_rows_with_non_positive_values(input_df, columns_to_check)

    def test_column_with_non_numeric_values(self):
        """
        Test case to ensure that a ValueError is raised when a specified column
        contains non-numeric elements (e.g., strings) that cannot be checked for positivity.
        """
        input_df = NonEmptyDataFrame({"A": [1, "two", 3], "B": [10, 20, 30]})
        columns_to_check = ["A"]
        with pytest.raises(ValueError, match="non-numeric elements"):
            drop_rows_with_non_positive_values(input_df, columns_to_check)

    def test_dataframe_with_float_values(self):
        """
        Test case to verify the correct behavior when dealing with float values,
        including negative floats and zero, ensuring corresponding rows are dropped.
        """
        input_df = NonEmptyDataFrame(
            {"A": [1.5, -0.1, 3.0, 0.0, 7.2], "B": [10.1, 20.2, 30.3, 40.4, 50.5]}
        )
        columns_to_check = ["A"]
        expected_df = NonEmptyDataFrame(
            {"A": [1.5, 3.0, 7.2], "B": [10.1, 30.3, 50.5]}, index=[0, 2, 4]
        )
        result_df = drop_rows_with_non_positive_values(input_df, columns_to_check)
        pd.testing.assert_frame_equal(result_df, expected_df)


class TestDropRowsNotSatisfyingRegex:
    """
    Test suite for the drop_rows_not_satisfying_regex function.
    """

    def test_drop_rows_not_satisfying_regex_drops_rows(self):
        """
        Test that rows not satisfying the regex are correctly dropped.
        """
        df = pd.DataFrame(
            {
                "col_id": [1, 2, 3, 4],
                "value": ["apple_123", "banana_xyz", "cherry_456", "date_abc"],
            }
        )
        columns_to_check = ["value"]
        regex_pattern = r"^\w+_(\d{3})$"  # Ends with _ and 3 digits
        expected_df = pd.DataFrame(
            {"col_id": [1, 3], "value": ["apple_123", "cherry_456"]}, index=[0, 2]
        )  # Original indices maintained

        result_df = drop_rows_not_satisfying_regex(df, columns_to_check, regex_pattern)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_drop_rows_not_satisfying_regex_no_rows_dropped(self):
        """
        Test that no rows are dropped when all values satisfy the regex.
        """
        df = pd.DataFrame(
            {"col_id": [1, 2, 3], "code": ["CODE_001", "CODE_002", "CODE_003"]}
        )
        columns_to_check = ["code"]
        regex_pattern = r"^CODE_\d{3}$"
        expected_df = df.copy()  # All rows should be kept

        result_df = drop_rows_not_satisfying_regex(df, columns_to_check, regex_pattern)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_drop_rows_not_satisfying_regex_multiple_columns_and_mixed_matches(self):
        """
        Test dropping rows based on multiple columns with mixed match results.
        A row is dropped if *any* of the specified columns fail the regex.
        """
        df = pd.DataFrame(
            {
                "id": [10, 11, 12, 13, 14],
                "email": [
                    "a@b.com",
                    "c@d.com",
                    "e.f.g@h.net",
                    "invalid-email",
                    "valid@domain.org",
                ],
                "phone": [
                    "123-456-7890",
                    "abc-def-ghij",
                    "987-654-3210",
                    "111-222-3333",
                    "555-555-5555",
                ],
            }
        )
        columns_to_check = ["email", "phone"]
        email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        phone_regex = r"^\d{3}-\d{3}-\d{4}$"

        combined_regex_pattern = f"({email_regex})|({phone_regex})"

        expected_df = pd.DataFrame(
            {
                "id": [
                    10,
                    12,
                    13,
                    14,
                ],  # Row 11 (index 1) 'abc-def-ghij' and 'c@d.com' (not matching phone/email)
                # Row 13 (index 3) 'invalid-email'
                "email": [
                    "a@b.com",
                    "e.f.g@h.net",
                    "invalid-email",
                    "valid@domain.org",
                ],
                "phone": [
                    "123-456-7890",
                    "987-654-3210",
                    "111-222-3333",
                    "555-555-5555",
                ],
            },
            index=[0, 2, 3, 4],
        )  # Row at index 1 is dropped because 'phone' fails.

        expected_df = pd.DataFrame(
            {
                "id": [10, 12, 14],
                "email": ["a@b.com", "e.f.g@h.net", "valid@domain.org"],
                "phone": ["123-456-7890", "987-654-3210", "555-555-5555"],
            },
            index=[0, 2, 4],
        )

        result_df = drop_rows_not_satisfying_regex(
            df, columns_to_check, combined_regex_pattern
        )
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_drop_rows_not_satisfying_regex_empty_dataframe_raises_validation_error(
        self,
    ):
        """
        Test that an empty DataFrame raises a ValidationError due to NonEmptyDataFrame typing.
        """
        df = pd.DataFrame()
        columns_to_check = ["any_col"]
        regex_pattern = r".*"
        with pytest.raises(ValidationError) as excinfo:
            drop_rows_not_satisfying_regex(df, columns_to_check, regex_pattern)
        assert "empty" in str(excinfo.value)

    def test_drop_rows_not_satisfying_regex_non_existent_column_raises_value_error(
        self,
    ):
        """
        Test that a ValueError is raised if a specified column does not exist.
        """
        df = pd.DataFrame({"col1": ["data"]})
        columns_to_check = ["non_existent"]
        regex_pattern = r".*"
        with pytest.raises(ValueError) as excinfo:
            drop_rows_not_satisfying_regex(df, columns_to_check, regex_pattern)
        assert (
            "Column 'non_existent' not found in the DataFrame. Cannot check regex."
            in str(excinfo.value)
        )

    def test_drop_rows_not_satisfying_regex_column_with_nulls_raises_value_error(self):
        """
        Test that a ValueError is raised if a specified column contains null values.
        """
        df = pd.DataFrame({"col1": ["data1", None, "data3"]})
        columns_to_check = ["col1"]
        regex_pattern = r".*"
        with pytest.raises(ValueError) as excinfo:
            drop_rows_not_satisfying_regex(df, columns_to_check, regex_pattern)
        assert (
            "Column 'col1' contains 'None' or missing values at indices: [1]. All values must be non-null for regex check."
            in str(excinfo.value)
        )

    def test_drop_rows_not_satisfying_regex_column_with_non_string_data_raises_value_error(
        self,
    ):
        """
        Test that a ValueError is raised if a specified column contains non-string elements.
        """
        df = pd.DataFrame({"col1": ["str1", 123, "str3"]})
        columns_to_check = ["col1"]
        regex_pattern = r".*"
        with pytest.raises(ValueError) as excinfo:
            drop_rows_not_satisfying_regex(df, columns_to_check, regex_pattern)
        assert "Column 'col1' contains non-string elements" in str(excinfo.value)

    def test_drop_rows_not_satisfying_regex_empty_columns_to_check_raises_validation_error(
        self,
    ):
        """
        Test that an empty list for columns_to_check raises a ValidationError.
        """
        df = pd.DataFrame({"col1": ["a"]})
        columns_to_check = []
        regex_pattern = r".*"
        with pytest.raises(ValidationError) as excinfo:
            drop_rows_not_satisfying_regex(df, columns_to_check, regex_pattern)
        assert "input_value=" in str(excinfo.value)

    def test_drop_rows_not_satisfying_regex_empty_regex_pattern_raises_validation_error(
        self,
    ):
        """
        Test that an empty regex pattern string raises a ValidationError.
        """
        df = pd.DataFrame({"col1": ["a"]})
        columns_to_check = ["col1"]
        regex_pattern = ""
        with pytest.raises(ValidationError) as excinfo:
            drop_rows_not_satisfying_regex(df, columns_to_check, regex_pattern)
        assert "String should have at least 1 character" in str(excinfo.value)


class TestCheckColumnsSatisfyRegex:
    """
    Test suite for the check_columns_satisfy_regex function.
    """

    def test_check_columns_satisfy_regex_success(self):
        """
        Test that the function returns the DataFrame when all specified string columns
        have values that fully satisfy the regex pattern.
        """
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "codes": ["ABC123", "XYZ789"],
                "names": ["NameOne", "NameTwo"],
                "mixed": ["ABC123", 456],  # This column should not be checked by regex
            }
        )
        columns_to_check = ["codes", "names"]
        # Regex for alphanumeric codes, and names starting with Name
        regex_pattern = r"^[A-Z]{3}\d{3}$|^Name[A-Za-z]+$"
        result_df = check_columns_satisfy_regex(df, columns_to_check, regex_pattern)
        pd.testing.assert_frame_equal(result_df, df)

    def test_check_columns_satisfy_regex_column_not_found_raises_value_error(self):
        """
        Test that a ValueError is raised if a specified column is not in the DataFrame.
        """
        df = pd.DataFrame({"colA": ["val1"]})
        columns_to_check = ["non_existent_col"]
        regex_pattern = r".*"
        with pytest.raises(ValueError) as excinfo:
            check_columns_satisfy_regex(df, columns_to_check, regex_pattern)
        assert "not found in the DataFrame. Cannot check regex." in str(excinfo.value)

    def test_check_columns_satisfy_regex_empty_column_raises_value_error(self):
        """
        Test that a ValueError is raised if a specified column contains None.
        """
        df = pd.DataFrame(
            {
                "col1": ["a", "b"],
                "empty_col": [None, None],  # Effectively empty
            }
        )
        columns_to_check = ["empty_col"]
        regex_pattern = r".*"
        with pytest.raises(ValueError) as excinfo:
            check_columns_satisfy_regex(df, columns_to_check, regex_pattern)
        assert "'None'" in str(excinfo.value)

    def test_check_columns_satisfy_regex_null_values_raises_value_error(self):
        """
        Test that a ValueError is raised if a specified column contains null values.
        """
        df = pd.DataFrame(
            {
                "data_col": ["item1", None, "item3"],  # Contains None
            }
        )
        columns_to_check = ["data_col"]
        regex_pattern = r".*"
        with pytest.raises(ValueError) as excinfo:
            check_columns_satisfy_regex(df, columns_to_check, regex_pattern)
        assert "'None'" in str(excinfo.value)

    def test_check_columns_satisfy_regex_non_string_data_raises_value_error(self):
        """
        Test that a ValueError is raised if a column contains non-string data.
        """
        df = pd.DataFrame(
            {
                "mixed_col": ["text", 123, "more_text"],  # Contains an integer
            }
        )
        columns_to_check = ["mixed_col"]
        regex_pattern = r".*"
        with pytest.raises(ValueError) as excinfo:
            check_columns_satisfy_regex(df, columns_to_check, regex_pattern)
        assert "contains non-string elements" in str(excinfo.value)

    def test_check_columns_satisfy_regex_no_match_raises_value_error(self):
        """
        Test that a ValueError is raised if a string value does not fully satisfy the regex.
        """
        df = pd.DataFrame(
            {
                "product_codes": ["P123", "A-456", "P789"],  # 'A-456' does not match
            }
        )
        columns_to_check = ["product_codes"]
        regex_pattern = r"^P\d{3}$"  # Starts with P, followed by 3 digits
        with pytest.raises(ValueError) as excinfo:
            check_columns_satisfy_regex(df, columns_to_check, regex_pattern)
        assert "does NOT fully satisfy the regex" in str(excinfo.value)

    def test_empty_dataframe_for_regex_check_raises_validation_error(self):
        """
        Test that an empty DataFrame raises a ValidationError due to NonEmptyDataFrame typing.
        """
        df = pd.DataFrame()
        columns_to_check = ["any_col"]
        regex_pattern = r".*"
        with pytest.raises(ValidationError) as excinfo:
            check_columns_satisfy_regex(df, columns_to_check, regex_pattern)
        assert "empty" in str(excinfo.value)

    def test_empty_columns_to_check_raises_validation_error_regex(self):
        """
        Test that an empty list for columns_to_check raises a ValidationError.
        """
        df = pd.DataFrame({"col1": ["a"]})
        columns_to_check = []  # Empty list
        regex_pattern = r".*"
        with pytest.raises(ValidationError) as excinfo:
            check_columns_satisfy_regex(df, columns_to_check, regex_pattern)
        assert "input_value=[]" in str(excinfo.value)

    def test_empty_regex_pattern_raises_validation_error(self):
        """
        Test that an empty regex pattern string raises a ValidationError due to NonEmptyStr.
        """
        df = pd.DataFrame({"col1": ["a"]})
        columns_to_check = ["col1"]
        regex_pattern = ""  # Empty string
        with pytest.raises(ValidationError) as excinfo:
            check_columns_satisfy_regex(df, columns_to_check, regex_pattern)
        assert "String should have at least 1 character" in str(excinfo.value)


class TestCheckStringColumns:
    """
    Test suite for the check_string_columns function.
    """

    def test_check_string_columns_success(self):
        """
        Test that the function returns the DataFrame when specified columns contain only strings and no nulls.
        """
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "description": ["Info 1", "Info 2", "Info 3"],
                "value": [10.5, 20.0, 30.1],
            }
        )
        columns_to_check = ["name", "description"]
        result_df = check_string_columns(df, columns_to_check)
        pd.testing.assert_frame_equal(result_df, df)

    def test_check_string_columns_column_not_found_raises_value_error(self):
        """
        Test that a ValueError is raised if a specified column is not in the DataFrame.
        """
        df = pd.DataFrame({"colA": ["val1"], "colB": ["val2"]})
        columns_to_check = ["colA", "non_existent_col"]
        with pytest.raises(ValueError) as excinfo:
            check_string_columns(df, columns_to_check)
        assert "not found" in str(excinfo.value)

    def test_check_string_columns_empty_column_raises_value_error(self):
        """
        Test that a ValueError is raised if a specified string column is empty (contains only nulls).
        """
        df = pd.DataFrame(
            {
                "col1": ["a", "b"],
                "empty_str_col": [None, None],  # Effectively empty for string check
                "col3": ["x", "y"],
            }
        )
        columns_to_check = ["empty_str_col"]
        with pytest.raises(ValueError) as excinfo:
            check_string_columns(df, columns_to_check)
        assert "empty" in str(excinfo.value)

    def test_check_string_columns_null_values_raises_value_error(self):
        """
        Test that a ValueError is raised if a string column contains null values.
        """
        df = pd.DataFrame(
            {
                "col1": ["str1", "str2", None],  # Contains None
                "col2": ["strA", "strB", "strC"],
            }
        )
        columns_to_check = ["col1"]
        with pytest.raises(ValueError) as excinfo:
            check_string_columns(df, columns_to_check)
        assert "'None'" in str(excinfo.value)

    def test_check_string_columns_non_string_data_raises_value_error(self):
        """
        Test that a ValueError is raised if a column contains non-string data (e.g., numbers, booleans).
        """
        df = pd.DataFrame(
            {
                "str_col": ["text", "more_text", "more_and_more_text"],
                "mixed_type_col": ["item1", 123, "item3"],  # Contains an integer
                "another_str_col": ["last", "first", "middle"],
            }
        )
        columns_to_check = ["mixed_type_col"]
        with pytest.raises(ValueError) as excinfo:
            check_string_columns(df, columns_to_check)
        assert "contains non-string elements" in str(excinfo.value)

    def test_empty_dataframe_for_string_check_raises_validation_error(self):
        """
        Test that an empty DataFrame raises a ValidationError due to NonEmptyDataFrame typing.
        """
        df = pd.DataFrame()
        columns_to_check = ["any_col"]
        with pytest.raises(ValidationError) as excinfo:
            check_string_columns(df, columns_to_check)
        assert "empty" in str(excinfo.value)

    def test_empty_columns_to_check_raises_validation_error(self):
        """
        Test that an empty list for columns_to_check raises a ValidationError.
        """
        df = pd.DataFrame({"col1": ["a", "b"]})
        columns_to_check = []  # Empty list
        with pytest.raises(ValidationError) as excinfo:
            check_string_columns(df, columns_to_check)
        assert "input_value=[]" in str(excinfo.value)


class TestCheckNumericColumns:
    """
    Test suite for the check_numeric_columns function.
    These tests explicitly avoid logging assertions.
    """

    def test_check_numeric_columns_success(self, sample_non_empty):
        """
        Test that the function returns the DataFrame when specified columns are numeric and valid.
        """
        columns_to_check = ["id", "num_col"]
        result_df = check_numeric_columns(sample_non_empty, columns_to_check)
        pd.testing.assert_frame_equal(result_df, sample_non_empty)

    def test_check_numeric_columns_column_not_found_raises_value_error(
        self, sample_non_empty
    ):
        """
        Test that a ValueError is raised if a specified column is not in the DataFrame.
        """

        columns_to_check = ["id", "non_existent_col"]
        with pytest.raises(ValueError) as excinfo:
            check_numeric_columns(sample_non_empty, columns_to_check)
        assert "not found" in str(excinfo.value)

    def test_check_numeric_columns_empty_column_raises_value_error(self):
        """
        Test that a ValueError is raised if a specified column is empty.
        (e.g., if a column contains only NaNs, which effectively makes it empty for numeric content)
        """
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "empty_col": [
                    float("nan"),
                    float("nan"),
                    float("nan"),
                ],  # Effectively empty
                "col3": [4, 5, 6],
            }
        )
        columns_to_check = ["empty_col"]
        with pytest.raises(ValueError) as excinfo:
            check_numeric_columns(df, columns_to_check)
        assert "None" in str(excinfo.value)

    def test_check_numeric_columns_null_values_raises_value_error(self):
        """
        Test that a ValueError is raised if a numeric column contains null values.
        """
        df = pd.DataFrame(
            {"col1": [1, 2, None], "col2": [3.0, 4.0, 5.0]}  # Contains None
        )
        columns_to_check = ["col1"]
        with pytest.raises(ValueError) as excinfo:
            check_numeric_columns(df, columns_to_check)
        assert "'None' or missing values" in str(excinfo.value)

    def test_check_numeric_columns_non_numeric_data_raises_value_error(self):
        """
        Test that a ValueError is raised if a column contains non-numeric data.
        """
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "mixed_col": [10, "abc", 20],  # Non-numeric string
                "col3": [30, 40, 50],
            }
        )
        columns_to_check = ["mixed_col"]
        with pytest.raises(ValueError) as excinfo:
            check_numeric_columns(df, columns_to_check)
        assert "contains non-numeric" in str(excinfo.value)

    def test_empty_dataframe_for_numeric_check_raises_validation_error(self):
        """
        Test that an empty DataFrame raises a ValidationError for check_numeric_columns.
        """
        df = pd.DataFrame()
        columns_to_check = ["any_col"]
        with pytest.raises(ValidationError) as excinfo:
            check_numeric_columns(df, columns_to_check)
        assert "empty" in str(excinfo.value)

    def test_empty_columns_to_check_raises_validation_error(self):
        """
        Test that an empty list for columns_to_check raises a ValidationError.
        """
        df = pd.DataFrame({"col1": [1, 2]})
        columns_to_check = []  # Empty list
        with pytest.raises(ValidationError) as excinfo:
            check_numeric_columns(df, columns_to_check)
        assert "input_value=[]" in str(excinfo.value)


class TestVerifyNoEmptyStrings:
    """
    Test suite for the verify_no_empty_strings function.
    """

    def test_no_empty_strings_success(self, sample_non_empty):
        """
        Test that the function returns the DataFrame when no empty strings are present.
        """
        result_df = verify_no_empty_strings(sample_non_empty)
        pd.testing.assert_frame_equal(result_df, sample_non_empty)

    def test_empty_string_raises_value_error(self):
        """
        Test that a ValueError is raised when an object column contains an empty string.
        """
        df = pd.DataFrame(
            {
                "col1": ["value1", "", "value3"],  # Empty string here
                "col2": [10, 20, 30],
            }
        )
        with pytest.raises(ValueError) as excinfo:
            verify_no_empty_strings(df)

        expected_error_message = (
            "empty strings ('') at indices: [1]. Empty strings are not allowed."
        )
        assert expected_error_message in str(excinfo.value)

    def test_empty_dataframe_raises_validation_error_empty_strings(self):
        """
        Test that an empty DataFrame raises a ValidationError for verify_no_empty_strings
        due to NonEmptyDataFrame typing.
        """
        df = pd.DataFrame()
        with pytest.raises(ValidationError) as excinfo:
            verify_no_empty_strings(df)

        assert "empty" in str(excinfo.value)

    def test_none_dataframe_raises_validation_error_empty_strings(self):
        """
        Test that an empty DataFrame raises a ValidationError for verify_no_empty_strings
        due to NonEmptyDataFrame typing.
        """
        df = None
        with pytest.raises(ValidationError) as excinfo:
            verify_no_empty_strings(df)

        assert "None" in str(excinfo.value)


class TestVerifyNoMissingData:
    """
    Test suite for the verify_no_missing_data function.
    """

    def test_no_missing_data_success(self, sample_non_empty):  # Use caplog fixture
        """
        Test that the function returns the DataFrame
        """
        # Reset caplog to ensure only logs from this test are captured
        result_df = verify_no_missing_data(sample_non_empty)
        pd.testing.assert_frame_equal(result_df, sample_non_empty)

    def test_missing_none_raises_value_error(
        self, sample_dataframe_general
    ):  # Use caplog fixture
        """
        Test that a ValueError is raised when the DataFrame contains NaN values.
        """
        with pytest.raises(ValueError) as excinfo:
            verify_no_missing_data(sample_dataframe_general)

        # Check for error message containing the column with NaN
        expected_error_message = "Found 'None' or missing values in column 'nullable_col' at indices: [1]. No missing data is allowed."
        assert expected_error_message in str(excinfo.value)

    def test_empty_df_as_input(self):  # Use caplog fixture
        """
        Test that a ValueError is raised when the input DataFrame is empty
        """
        with pytest.raises(ValueError) as excinfo:
            verify_no_missing_data(pd.DataFrame())

        # Check for error message containing the column with NaN
        expected_error_message = "empty"
        assert expected_error_message in str(excinfo.value)

    def test_none_df_as_input(self):  # Use caplog fixture
        """
        Test that a ValueError is raised when the input is None
        """
        with pytest.raises(ValueError) as excinfo:
            verify_no_missing_data(None)

        # Check for error message containing the column with NaN
        expected_error_message = "None"
        assert expected_error_message in str(excinfo.value)


class TestVerifyNoNegatives:
    """
    Test suite for the verify_no_negatives function.
    These tests explicitly avoid logging assertions as requested.
    """

    def test_no_negatives_success(self, sample_non_empty):
        """
        Test that the function returns the DataFrame when no negative values are present.
        """
        result_df = verify_no_negatives(sample_non_empty)
        pd.testing.assert_frame_equal(result_df, sample_non_empty)

    def test_negatives_raises_value_error(self):
        """
        Test that a ValueError is raised when a numeric column contains negative values.
        """
        df = pd.DataFrame(
            {
                "col1": [1, -2, 3],  # Negative value here
                "col2": [0.0, 1.5, 10.0],
            }
        )
        with pytest.raises(ValueError) as excinfo:
            verify_no_negatives(df)

        expected_error_message = "negative values in numeric column 'col1' at indices: [1]. All numeric values must be non-negative."
        assert expected_error_message in str(excinfo.value)

    def test_negatives_in_float_column_raises_value_error(self):
        """
        Test that a ValueError is raised when a float column contains negative values.
        """
        df = pd.DataFrame(
            {"col_a": [1.0, 2.0, -5.5], "col_b": ["x", "y", "z"]}  # Negative float here
        )
        with pytest.raises(ValueError) as excinfo:
            verify_no_negatives(df)

        expected_error_message = "negative values in numeric column 'col_a' at indices: [2]. All numeric values must be non-negative."
        assert expected_error_message in str(excinfo.value)

    def test_empty_dataframe_raises_validation_error_negatives(self):
        """
        Test that an empty DataFrame raises a ValidationError for verify_no_negatives due to NonEmptyDataFrame typing.
        """
        df = pd.DataFrame()
        with pytest.raises(ValidationError) as excinfo:
            verify_no_negatives(df)

        assert "empty" in str(excinfo.value)

    def test_dataframe_with_nan_but_no_negatives(self):
        """
        Test with a DataFrame containing NaNs but no negative values in numeric columns.
        NaNs should be ignored for negative checks.
        """
        df = pd.DataFrame(
            {
                "col1": [1, float("nan"), 3],
                "col2": [0.0, None, 10.0],  # None in object column, NaN in numeric
                "col3": [4, 5, 6],
            }
        )
        result_df = verify_no_negatives(df)
        pd.testing.assert_frame_equal(result_df, df)

    def test_non_numeric_columns_are_ignored(self):
        """
        Test that non-numeric columns with values that might look negative (e.g., strings)
        do not trigger an error.
        """
        df = pd.DataFrame(
            {
                "text_col": ["positive", "negative_str", "100", "-50_string"],
                "num_col": [1, 2, 3, 4],
            }
        )
        result_df = verify_no_negatives(df)
        pd.testing.assert_frame_equal(result_df, df)


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
