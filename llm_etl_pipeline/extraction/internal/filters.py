import logging


class _SpecificWarningFilter(logging.Filter):
    """
    A custom logging filter to suppress specific warning messages.

    This filter allows you to define a particular message string. Any log record
    whose message contains this string will be ignored and not processed by the
    logger. All other log records will pass through the filter.

    Attributes:
        message_to_ignore (str): The specific substring that, if found in a log
                                 record's message, will cause the record to be
                                 filtered out.
    """

    def __init__(self, message_to_ignore: str):
        super().__init__()
        self.message_to_ignore = message_to_ignore

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Determines whether the given log record should be processed or ignored.

        This method is called for each log record. It returns `False` if the
        record's message contains `message_to_ignore`, effectively suppressing
        that record. Otherwise, it returns `True`, allowing the record to pass.

        Args:
            record: The log record to evaluate.

        Returns:
            True if the record should be processed (shown), False if it should
            be ignored.
        """
        if self.message_to_ignore in record.getMessage():
            return False
        return True
