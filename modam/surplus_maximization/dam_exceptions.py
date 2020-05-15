"""Implements exception classes to define exceptions to be thrown during solution process"""


class UnsupportedProblemException(Exception):

    """Implements unsupported problem exception"""

    def __init__(self, message):
        self.message = message
