from abc import abstractmethod
from typing import Iterable

from coba.environments.primitives import Environment, Interaction

class WarmStartEnvironment(Environment):
    """The interface for an environment made with logged bandit data and simulated interactions."""
       
    @abstractmethod
    def read(self) -> Iterable[Interaction]:
        """The sequence of interactions in a simulation.

        Remarks:
            This function should always be "re-iterable".
        """
        ...
