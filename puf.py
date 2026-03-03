from abc import ABC, abstractmethod

class Puf:

    def __init__(self):
        """
        creates a new instance of the PUF class
        """
        pass

    @abstractmethod
    def evaluate(self, challenge):
        """
        @return the response of the PUF instance to @challenge
        might incorporate a noisy measurement of the response
        """
        pass