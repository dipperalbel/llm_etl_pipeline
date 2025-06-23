Data Transformation
==========================

Following the extraction phase, the stored JSON files are loaded into `pandas` DataFrames for subsequent transformation and consolidation. This stage is crucial for refining the extracted raw data:

* **Monetary Data Transformation:**
    Extracted monetary data undergoes various validation checks to ensure its quality and adherence to expected conditions. We then filter and retain only the information most relevant for analysis, such as the grant requested per project, available call budgets, or specific budget allocations per topic as mentioned in the call for proposal.
    A significant challenge identified was the **duplication of monetary values**, where identical amounts might or might not refer to the same underlying entity or concept (e.g., two mentions of the minimum EU grant request). To address this, a specific deduplication strategy is employed:
    1.  Sentences containing the duplicate monetary amounts are converted into embeddings using `Sentence-Transformers (S-BERT)`(https://github.com/UKPLab/sentence-transformers).
    2.  Hierarchical clustering is then performed on these embeddings using cosine distance.
    3.  If multiple sentences fall within the same cluster (indicating high semantic similarity for the same amount), the sentence with the *longest text* is selected as the representative for that cluster, simplifying the data while retaining context.

* **Entity Data Transformation:**
    For the extracted entity data, due to time constraints, the transformation primarily involves basic validation checks followed by a simple stacking of the type of organization data extracted.

* **Pipeline Orchestration:** All these transformation steps are orchestrated via a `Pipeline` class, which applies a series of pre-written functions sequentially to the `pandas` DataFrames, streamlining the data processing workflow.