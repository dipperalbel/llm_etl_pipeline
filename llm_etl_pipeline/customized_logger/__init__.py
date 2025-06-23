"""
This __init__.py file exports the configured logger instance from the customized_logger package.

It provides a centralized and pre-configured logger for consistent logging
across the entire llm_etl_pipeline project, using the Loguru library.
"""

from llm_etl_pipeline.customized_logger.loggers import logger

__all__ = ["logger"]
