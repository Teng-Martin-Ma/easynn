import logging


class Logger:
    """Logger class to log the training process.

    Args:
        logname (str): log file name
    """

    def __init__(self, logname):
        self.logger = logging.getLogger(__name__)
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)
        fh = logging.FileHandler(logname)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(fh)

    def info(self, content):
        """Log the content."""
        self.logger.info(content)

    def close(self):
        """Close the logger."""
        self.log('')
        logging.shutdown()
