import inspect
from functools import partial
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import ValidationError

# Import the Pipeline class and NonEmptyDataFrame from their respective locations
from llm_etl_pipeline.transformation import Pipeline
from llm_etl_pipeline.typings import NonEmptyDataFrame


# Define some helper functions for testing the pipeline
def func_add_column(
    df: NonEmptyDataFrame, col_name: str, value: int
) -> NonEmptyDataFrame:
    """Adds a new column to the DataFrame."""
    df[col_name] = value
    return df


def func_filter_rows(
    df: NonEmptyDataFrame, column: str, threshold: int
) -> NonEmptyDataFrame:
    """Filters rows based on a column value."""
    return df[df[column] > threshold]


def func_return_non_df(
    df: NonEmptyDataFrame,
) -> NonEmptyDataFrame:  # Incorrect return type
    """Returns a dict, which should cause a runtime error."""
    return {"data": df.to_dict()}


def func_return_empty_df(df: NonEmptyDataFrame) -> NonEmptyDataFrame:
    """Returns an empty DataFrame."""
    return df[
        df["value"] < 0
    ]  # Assumes 'value' column exists; will likely return empty


def func_raises_error(df: NonEmptyDataFrame) -> NonEmptyDataFrame:
    """A function that always raises an error."""
    raise RuntimeError("Simulated error in pipeline function.")


# Functions with incorrect signatures for validator testing
def func_no_args() -> NonEmptyDataFrame:  # No args
    return pd.DataFrame()


def func_wrong_return_type(df: NonEmptyDataFrame) -> int:  # Wrong return type
    return 1


def func_wrong_first_arg_type(data: list) -> NonEmptyDataFrame:  # Wrong first arg type
    return pd.DataFrame()


# Create a fixture for a non-empty DataFrame
@pytest.fixture
def sample_dataframe() -> NonEmptyDataFrame:
    return pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})


# Mock the logger to prevent console output during tests and potentially assert calls
@pytest.fixture(autouse=True)  # autouse=True means this fixture runs for every test
def mock_logger():
    with patch("llm_etl_pipeline.customized_logger.logger") as mock_log:
        yield mock_log


class TestPipeline:

    # --- Test Pipeline Initialization and _check_function_signature ---

    def test_pipeline_init_empty_functions(self):
        """Test initializing a Pipeline with an empty list of functions."""
        pipeline = Pipeline(functions=[])
        assert pipeline.functions == []

    def test_pipeline_init_valid_functions(self):
        """Test initializing a Pipeline with functions having correct signatures."""

        def valid_func1(df: pd.DataFrame) -> pd.DataFrame:
            return df

        def valid_func2(df: NonEmptyDataFrame, x: int) -> NonEmptyDataFrame:
            return df

        pipeline = Pipeline(functions=[valid_func1, valid_func2])
        assert len(pipeline.functions) == 2
        assert pipeline.functions[0] is valid_func1
        assert pipeline.functions[1] is valid_func2

    def test_pipeline_init_invalid_no_args_function(self):
        """Test initialization fails if a function has no arguments."""
        with pytest.raises(ValidationError) as excinfo:
            Pipeline(functions=[func_no_args])
        assert "must accept at least one argument" in str(excinfo.value)

    def test_pipeline_init_invalid_return_type_function(self):
        """Test initialization fails if a function has an incorrect return type annotation."""
        with pytest.raises(ValidationError) as excinfo:
            Pipeline(functions=[func_wrong_return_type])
        assert "has return type annotation" in str(excinfo.value)
        assert "expected 'pd.DataFrame' or 'NonEmptyDataFrame'" in str(excinfo.value)

    def test_pipeline_init_invalid_first_arg_type_function(self):
        """Test initialization fails if a function's first argument is not DataFrame-typed."""
        with pytest.raises(ValidationError) as excinfo:
            Pipeline(functions=[func_wrong_first_arg_type])
        assert (
            "The first argument must be type-annotated as 'pd.DataFrame' or 'NonEmptyDataFrame'"
            in str(excinfo.value)
        )

    def test_pipeline_init_with_partial_function(self):
        """Test initialization with a partial function."""

        def func_with_extra_args(
            df: NonEmptyDataFrame, arg1: int, arg2: str
        ) -> NonEmptyDataFrame:
            df[arg2] = arg1
            return df

        # This partial ensures that when the pipeline calls func, it only needs the DataFrame.
        # The inspect.signature for func_partial still reflects the original function's signature
        # but handles the binding of 'arg1' and 'arg2'. The first argument remains 'df'.
        func_partial = partial(func_with_extra_args, arg1=100, arg2="new_col")

        pipeline = Pipeline(functions=[func_partial])
        assert len(pipeline.functions) == 1
        assert pipeline.functions[0] is func_partial

    # --- Test Pipeline.run method ---

    def test_run_empty_pipeline(
        self, sample_dataframe: NonEmptyDataFrame, mock_logger: MagicMock
    ):
        """Test running an empty pipeline returns the input DataFrame."""
        pipeline = Pipeline()
        result_df = pipeline.run(sample_dataframe.copy())
        pd.testing.assert_frame_equal(result_df, sample_dataframe)

    def test_run_single_function(
        self, sample_dataframe: NonEmptyDataFrame, mock_logger: MagicMock
    ):
        """Test running a pipeline with a single function."""
        pipeline = Pipeline(
            functions=[partial(func_add_column, col_name="test_col", value=100)]
        )
        expected_df = sample_dataframe.copy()
        expected_df["test_col"] = 100

        result_df = pipeline.run(sample_dataframe.copy())
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_run_multiple_functions(
        self, sample_dataframe: NonEmptyDataFrame, mock_logger: MagicMock
    ):
        """Test running a pipeline with multiple functions."""
        # Function to add a column
        add_col_func = partial(func_add_column, col_name="new_val", value=5)

        # Function to filter, operating on the newly added column
        filter_func = partial(func_filter_rows, column="new_val", threshold=4)

        pipeline = Pipeline(functions=[add_col_func, filter_func])

        initial_df = sample_dataframe.copy()
        expected_df_after_add = initial_df.copy()
        expected_df_after_add["new_val"] = 5

        expected_df_final = expected_df_after_add[
            expected_df_after_add["new_val"] > 4
        ].copy()  # Should be all rows

        result_df = pipeline.run(initial_df)
        pd.testing.assert_frame_equal(result_df, expected_df_final)

    def test_run_function_returns_non_dataframe(
        self, sample_dataframe: NonEmptyDataFrame, mock_logger: MagicMock
    ):
        """Testa l'esecuzione di una pipeline dove una funzione restituisce un oggetto non-DataFrame."""
        pipeline = Pipeline(
            functions=[func_return_non_df]
        )  # Questo ora dovrebbe passare la validazione di init
        with pytest.raises(
            TypeError
        ) as excinfo:  # Questo TypeError dovrebbe essere sollevato dal metodo .run
            pipeline.run(sample_dataframe)
        assert "returned type 'dict', but expected 'DataFrame'." in str(excinfo.value)

    def test_run_function_returns_empty_dataframe(
        self, sample_dataframe: NonEmptyDataFrame, mock_logger: MagicMock
    ):
        """Test running a pipeline where a function returns an empty DataFrame."""
        # Ensure sample_dataframe has a 'value' column for func_return_empty_df
        pipeline = Pipeline(functions=[func_return_empty_df])

        # The pipeline should stop and return an empty DataFrame
        result_df = pipeline.run(sample_dataframe)

        assert result_df.empty

    def test_run_function_raises_exception(
        self, sample_dataframe: NonEmptyDataFrame, mock_logger: MagicMock
    ):
        """Test running a pipeline where a function raises an arbitrary exception."""
        pipeline = Pipeline(functions=[func_raises_error])
        with pytest.raises(RuntimeError) as excinfo:
            pipeline.run(sample_dataframe)
        assert "Simulated error in pipeline function." in str(excinfo.value)

    def test_validate_assignment_on_functions(self):
        """Test that assigning invalid functions after initialization re-runs validation."""
        pipeline = Pipeline()  # Initially empty, valid
        assert pipeline.functions == []

        # Assign an invalid function (no arguments)
        with pytest.raises(ValidationError) as excinfo:
            pipeline.functions = [func_no_args]
        assert "must accept at least one argument" in str(excinfo.value)

        # After failure, the attribute should NOT have been updated (it should remain empty)
        assert pipeline.functions == []

        # Assign a valid function list
        def valid_func(df: pd.DataFrame) -> pd.DataFrame:
            return df

        pipeline.functions = [valid_func]
        assert pipeline.functions == [valid_func]
