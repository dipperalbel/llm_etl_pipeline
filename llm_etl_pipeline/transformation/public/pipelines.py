"""
This module defines the `Pipeline` class, which enables the creation and
execution of data processing workflows for pandas DataFrames.

The `Pipeline` class facilitates applying a sequence of transformation
functions to data, ensuring type consistency and providing robust runtime
validation throughout the process.
"""

import inspect
from typing import Any, Callable, List

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    validate_call,
)

from llm_etl_pipeline.customized_logger import logger
from llm_etl_pipeline.typings import NonEmptyDataFrame


class Pipeline(BaseModel):
    """
    A class to create and execute a data processing pipeline for pandas DataFrames.

    This pipeline applies a sequence of functions to a DataFrame, where the output
    of one function serves as the input for the next. It ensures type consistency
    and provides runtime validation for the function chain.

    Attributes:
        functions (List[Callable[[NonEmptyDataFrame, Any], NonEmptyDataFrame]]):
            A list of callable functions that represent the steps in the pipeline.
            Each function is expected to accept a pandas DataFrame (specifically
            `NonEmptyDataFrame` or `pd.DataFrame`) as its first argument and
            return a pandas DataFrame (`NonEmptyDataFrame` or `pd.DataFrame`).
            Functions can optionally accept additional `Any` arguments, but
            at least one DataFrame-typed argument is required.
            Defaults to an empty list.
    """

    model_config = ConfigDict(validate_assignment=True)
    functions: List[Callable[[NonEmptyDataFrame, Any], NonEmptyDataFrame]] = Field(
        default_factory=list
    )

    @field_validator("functions")
    # pylint: disable=no-self-argument
    def _check_function_signature(
        cls, v: List[Callable[[NonEmptyDataFrame, Any], NonEmptyDataFrame]]
    ) -> List[Callable[[NonEmptyDataFrame, Any], NonEmptyDataFrame]]:
        """
        Validates the signature of each function added to the pipeline.

        This validator ensures that:
        1. Each function accepts at least one argument.
        2. The function's return type annotation is either `pd.DataFrame` or `NonEmptyDataFrame`.
        3. At least one of the function's arguments is type-annotated as `pd.DataFrame`
           or `NonEmptyDataFrame`. This argument is expected to be the DataFrame passed
           between pipeline steps.

        Args:
            v (List[Callable[[NonEmptyDataFrame, Any], NonEmptyDataFrame]]):
                The list of functions provided to the `functions` attribute.

        Returns:
            List[Callable[[NonEmptyDataFrame, Any], NonEmptyDataFrame]]:
                The validated list of functions if all checks pass.

        Raises:
            ValueError:
                - If any function in the list does not accept at least one argument.
                - If any function's return type annotation is not `pd.DataFrame`
                  or `NonEmptyDataFrame`.
                - If no argument in a function is type-annotated as `pd.DataFrame`
                  or `NonEmptyDataFrame`.
        """
        if len(v) == 0:
            return v
        for i, func in enumerate(v):
            func_name = (
                func.__name__ if hasattr(func, "__name__") else f"function at index {i}"
            )
            signature = inspect.signature(func)
            parameters = list(signature.parameters.values())
            # DEBUG: Stampa i dettagli di ogni funzione controllata
            if not parameters:
                logger.error(
                    f"Function '{func_name}' must accept at least one argument."
                )
                raise ValueError(
                    f"Function '{func_name}' must accept at least one argument."
                )
            return_annotation = signature.return_annotation
            if not (
                return_annotation is pd.DataFrame
                or return_annotation is NonEmptyDataFrame
            ):
                logger.error(
                    f"Function '{func_name}': has return type annotation '{getattr(return_annotation, '__name__', str(return_annotation))}', but expected 'pd.DataFrame' or 'NonEmptyDataFrame'."
                )
                raise ValueError(
                    f"Function '{func_name}': has return type annotation '{getattr(return_annotation, '__name__', str(return_annotation))}', but expected 'pd.DataFrame' or 'NonEmptyDataFrame'."
                )
            check_for_annotations = False
            annotation = parameters[0].annotation
            if annotation is pd.DataFrame or annotation is NonEmptyDataFrame:
                check_for_annotations = True
            if not check_for_annotations:
                logger.error(
                    f"Function '{func_name}': The first argument must be type-annotated as 'pd.DataFrame' or 'NonEmptyDataFrame'."
                )
                raise ValueError(
                    f"Function '{func_name}': The first argument must be type-annotated as 'pd.DataFrame' or 'NonEmptyDataFrame'."
                )
        return v

    @validate_call
    def run(self, input_df: NonEmptyDataFrame) -> pd.DataFrame:
        """
        Executes the defined pipeline of functions on the provided DataFrame.

        The `input_df` is passed as the initial input to the first function.
        The output DataFrame from each function is then passed as the input to
        the subsequent function in the `functions` list.
        A runtime check ensures that each function indeed returns a `pandas.DataFrame`.

        Args:
            input_df (NonEmptyDataFrame): The initial pandas DataFrame to start
                                          the pipeline processing.

        Returns:
            pd.DataFrame: The final DataFrame after all functions in the pipeline have been successfully executed.

        Raises:
            TypeError: If any function in the pipeline returns a value that is
                       not an instance of `pandas.DataFrame`.
            Exception: Re-raises any other exceptions that occur during the
                       execution of individual functions within the pipeline.
        """
        df = input_df.copy()
        logger.info(f"Running Pipeline with {len(self.functions)} functions.")
        for i, func in enumerate(self.functions):
            func_name = (
                func.__name__ if hasattr(func, "__name__") else f"anonymous_func_{i}"
            )
            logger.info(f"Executing step {i+1}: {func_name}")

            try:
                result_df = func(df)

                # *** RUNTIME ACTUAL RETURN TYPE VERIFICATION ***
                if not isinstance(result_df, pd.DataFrame):

                    logger.error(
                        f"Function '{func_name}' at step {i+1} returned type "
                        f"'{type(result_df).__name__}', but expected 'DataFrame'."
                    )
                    raise TypeError(
                        f"Function '{func_name}' at step {i+1} returned type "
                        f"'{type(result_df).__name__}', but expected 'DataFrame'."
                    )

                df = result_df.copy()

                if df.empty:
                    logger.warning(
                        f"DataFrame is empty after function '{func_name}' at step {i+1}. Stopping pipeline execution."
                    )
                    break

            except Exception:
                logger.error(
                    f"Error during execution of function '{func_name}' at step {i+1}"
                )
                raise

        logger.success("Pipeline execution complete.")
        return df
