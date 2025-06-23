Output Files
==========================

Upon successful completion of the extraction and processing, two CSV files will be generated in the root directory:

* `etl_money_result.csv`: This file contains the extracted and processed monetary information. It includes the following columns:
    * `document_id`: The EU grant project identifier (e.g., `AMIF-2024-TF2-AG-THB`).
    * `value`: The extracted monetary amount.
    * `currency`: The currency associated with the `value`.
    * `context`: The motivation or context for the extracted amount.
    * `original_sentence`: The original sentence from the input text where the amount was found.

* `etl_entity_result.csv`: This file contains the extracted and validated entity data. It includes the following columns:
    * `document_id`: The EU grant project identifier.
    * `organization_type`: A list of organization types found in the consortium table of the proposal PDF.
    * `min_entities`: The minimum number of entities, as indicated in the entities row of the consortium table.