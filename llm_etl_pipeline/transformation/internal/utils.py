import pandas as pd
from pydantic import StrictFloat
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from llm_etl_pipeline.customized_logger import logger
from llm_etl_pipeline.typings import NonEmptyListStr


def _cluster_list_sents(
    input_list: NonEmptyListStr = None,
    threshold: StrictFloat = 0.8,
    model_st: SentenceTransformer = None,
    hierarchical_clustering: AgglomerativeClustering = None,
) -> list[str]:
    """
    Clusters a list of sentences semantically and returns the longest sentence from each cluster.

    This function utilizes Sentence-BERT to generate embeddings for the input sentences, followed by
    hierarchical agglomerative clustering. Sentences are grouped into clusters if their cosine
    distance is below the specified `threshold`. For each resulting cluster, the longest sentence
    is chosen as its representative. If the input list contains only one sentence, clustering is
    bypassed, and the original list is returned directly.

    Args:
        input_list (NonEmptyListStr): A list of strings (sentences) to be clustered. Must not be empty.
        threshold (StrictFloat): The maximum cosine distance for sentences to be considered
                                 part of the same cluster. Defaults to 0.8.
        model_st (SentenceTransformer): An initialized SentenceTransformer model for generating
                                        sentence embeddings. If None, it implies an external
                                        caller handles model loading.
        hierarchical_clustering (AgglomerativeClustering): An initialized AgglomerativeClustering
                                                           model configured for hierarchical clustering.
                                                           If None, it implies an external caller
                                                           handles model initialization.

    Returns:
        list[str]: A list of representative sentences, where each sentence is the longest
                   from its respective cluster.

    Raises:
        ValueError: If there's an issue generating sentence embeddings or performing
                    hierarchical clustering.
    """
    sentences = input_list.copy()
    result_list = sentences
    logger.info(f"Received {len(sentences)} sentences for clustering.")

    if sentences and len(sentences) > 1 and model_st and hierarchical_clustering:
        logger.info(
            "Proceeding with clustering as conditions are met (more than one sentence, models provided)."
        )

        try:
            logger.info("Generating sentence embeddings...")
            embeddings = model_st.encode(sentences)
            logger.info("Sentence embeddings generated successfully.")
        except Exception as e:
            logger.error(f"Failed to generate sentence embeddings. Error: {e}")
            raise ValueError(
                f"Failed to generate sentence embeddings. Please check the SentenceTransformer model or input data: {e}"
            )

        try:
            logger.info("Performing hierarchical clustering...")
            clusters = hierarchical_clustering.fit_predict(embeddings)
            logger.info(
                f"Hierarchical clustering completed. Found {len(set(clusters))} clusters."
            )
        except Exception as e:
            logger.error(f"Failed to perform hierarchical clustering. Error: {e}")
            raise ValueError(
                f"Failed to perform hierarchical clustering. Ensure the clustering model is correctly initialized and the embeddings are valid: {e}"
            )

        logger.info("Selecting the longest sentence from each cluster...")
        df_clusters = (
            pd.DataFrame({"sentences": sentences, "clusters": clusters})
            .groupby("clusters")["sentences"]
            .apply(lambda x: max(x, key=len))
            .reset_index()
        )

        result_list = df_clusters["sentences"].to_list()
        logger.success(
            f"Clustering complete. Reduced {len(sentences)} sentences to {len(result_list)} representative sentences."
        )

    elif sentences and len(sentences) <= 1:
        logger.success(
            "Input list contains 0 or 1 sentence. Skipping clustering and returning the original list."
        )
    else:
        logger.warning(
            "Clustering skipped: Either input list or SentenceTransformer model or AgglomerativeClustering model was not provided."
        )

    return result_list
