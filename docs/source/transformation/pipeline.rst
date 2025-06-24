==========================
Pipeline
==========================

A class to create and execute a data processing pipeline for pandas DataFrames.
This pipeline applies a sequence of functions to a DataFrame, where the output
of one function serves as the input for the next. It ensures type consistency
and provides runtime validation for the function chain.


**Example Usage:**

Let's illustrate how to define a data processing pipeline using the ``Pipeline`` class
and execute it on a pandas DataFrame.


.. code-block:: python

   import pandas as pd
   import logging
   from llm_etl_pipeline.transformation import Pipeline


    # Define some sample transformation functions
   def add_one_column(df: pd.DataFrame) -> pd.DataFrame:
        """Adds a 'value_plus_one' column."""
        logger.info(f"Step: add_one_column received {len(df)} rows.")
        df['value_plus_one'] = df['value'] + 1
        print(f"After add_one_column:\n{df}\n")
        return df

   def multiply_by_two_column(df: pd.DataFrame) -> pd.DataFrame:
        """Multiplies the 'value_plus_one' column by 2."""
        logger.info(f"Step: multiply_by_two_column received {len(df)} rows.")
        df['value_times_two'] = df['value_plus_one'] * 2
        print(f"After multiply_by_two_column:\n{df}\n")
        return df

   def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Filters rows where 'value' is greater than 1."""
        logger.info(f"Step: filter_rows received {len(df)} rows.")
        filtered_df = df[df['value'] > 1]
        print(f"After filter_rows:\n{filtered_df}\n")
        return filtered_df

    # Create an initial DataFrame
   initial_df = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'value': [10, 20, 5, 15]
   })

   pipeline = Pipeline(
        functions=[
            add_one_column,
            filter_rows,
            multiply_by_two_column # Note: filter_rows might remove data needed by subsequent steps
                                   # This ordering will make multiply_by_two_column operate on a potentially smaller DF
        ]
   )

   final_df = pipeline.run(initial_df.copy())


API Reference
--------------

.. autoclass:: llm_etl_pipeline.Pipeline
   :members:
   :undoc-members:
   :show-inheritance: