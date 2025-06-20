# DISABLE FOR THE TEST THE LOGGERS
from llm_etl_pipeline.customized_logger.loggers import (
    _configure_logger_from_env,
    logger,
)

logger.disable("")
