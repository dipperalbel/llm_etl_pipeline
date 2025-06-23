"""
Provides a centralized and customizable logging solution for the project using Loguru.

This module sets up a robust logging system that can be configured via environment
variables, directing log messages to both standard output (console) and a file.
It supports dynamic log level control and provides custom color schemes for better readability.
"""

import os
import sys

from loguru import logger

# --- 1. Definire la directory dei log ---
LOG_DIRECTORY = "logs"
LOG_FILE_NAME = "file_log.log"
log_file_path = os.path.join(LOG_DIRECTORY, LOG_FILE_NAME)

DEFAULT_LOGGER_LEVEL = "INFO"
# Dynamically control logging state with env vars
DISABLE_LOGGER_ENV_VAR_NAME = "PROJECT_DISABLE_LOGGER"
LOGGER_LEVEL_ENV_VAR_NAME = "PROJECT_LOGGER_LEVEL"


class _DedicatedStream:
    """
    A dedicated stream wrapper for formatting and directing messages to
    a base stream.
    This class intercepts messages and adds a consistent prefix before
    writing them to an underlying stream, typically standard output or a file.
    """

    def __init__(self, base):
        """
        Initializes the dedicated stream wrapper.

        Args:
            base: The underlying stream object (e.g., sys.stdout, an open file object)
                  to which formatted messages will be written.
        """
        self.base = base

    def write(self, message):
        """
        Writes a formatted message to the base stream.

        The message will be prefixed with "[project] " before being written.

        Args:
            message (str): The string message to be written.
        """
        # You can add a prefix or other formatting if you wish
        self.base.write(f"[project] {message}")

    def flush(self):
        """
        Flushes the base stream's buffer.

        Ensures that all buffered output is immediately written to the base stream.
        """
        self.base.flush()


dedicated_stream = _DedicatedStream(sys.stdout)


# Helper to read environment config at import time
def _read_env_vars() -> tuple[bool, str]:
    """
    Returns the (disabled_status, level) read from environment variables.
    """
    disable_str = os.getenv(DISABLE_LOGGER_ENV_VAR_NAME, "False").lower()
    disable_logger = disable_str in ["true", "1", "yes"]
    # Default to DEFAULT_LOGGER_LEVEL if no variable is set or invalid
    level_str = os.getenv(LOGGER_LEVEL_ENV_VAR_NAME, DEFAULT_LOGGER_LEVEL).upper()
    valid_levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
    if level_str not in valid_levels:
        level_str = DEFAULT_LOGGER_LEVEL
    return disable_logger, level_str


def _apply_color_scheme():
    """
    Defines custom colors for each log level (mimicking colorlog style)
    """
    logger.level("DEBUG", color="<cyan>")
    logger.level("INFO", color="<blue>")
    logger.level("SUCCESS", color="<green>")
    logger.level("WARNING", color="<yellow>")
    logger.level("ERROR", color="<red>")
    logger.level("CRITICAL", color="<red><bold>")


# Main configuration function
def _configure_logger_from_env():
    """
    Configures the Loguru logger based on environment variables.
    This can be called at import time (once) or re-called any time.

    (Loguru does not require `getLogger(name)`; we just import `logger` and use it.)
    """
    disable_logger, level_str = _read_env_vars()

    # If the library name is used, we can selectively enable/disable it.
    # But Loguru doesn't use named loggers the same way stdlib does,
    # so we can do a global enable/disable or apply filter functions instead.
    if disable_logger:
        logger.disable("")
    else:
        logger.enable("")

    # Remove default handlers
    logger.remove()

    # Apply custom level color scheme
    _apply_color_scheme()

    output_format = (
        "<white>{time:YYYY-MM-DD HH:mm:ss.SSS}</white> | "
        "<level>{level: <7}</level> | "
        "{message}"
    )

    logger.add(
        dedicated_stream,
        level=level_str,
        colorize=True,
        format=output_format,
    )

    logger.add(
        log_file_path,
        level=level_str,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        colorize=True,
        format=output_format,
    )


_configure_logger_from_env()
