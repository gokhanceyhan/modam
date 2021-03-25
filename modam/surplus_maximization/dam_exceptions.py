"""Implements exception classes to define exceptions to be thrown during solution process"""


class InvalidBidException(Exception):

    """Implements invalid bid exception"""

    def __init__(self, message):
        self.message = message


class UnsupportedProblemException(Exception):

    """Implements unsupported problem exception"""

    def __init__(self, message):
        self.message = message
