import collections.abc

from abc import abstractmethod
from typing import Sequence, Dict, Any, Iterable

from coba.exceptions import CobaException
from coba.environments.primitives import Context, Action, Environment, Interaction

class SimulatedInteraction(Interaction):
    """Simulated data that describes an interaction where the choice is up to you."""

    def __init__(self,
        context : Context,
        actions : Sequence[Action],
        rewards : Sequence[float],
        **kwargs) -> None:
        ...
        """Instantiate SimulatedInteraction.

        Args
            context : Features describing the interaction's context.
            actions : Features describing available actions during the interaction.
            rewards : The reward for each action in the interaction.
            **kwargs: Additional information that should be recorded in the interactions table of an experiment result. If any
                data is a sequence with length equal to actions only the data at the selected action index will be recorded.
        """

        if rewards and len(rewards) != len(actions):
            raise CobaException("Interaction reward counts must equal action counts.")

        self._actions  = actions
        self._rewards = rewards

        super().__init__(context, **kwargs)

    @property
    def actions(self) -> Sequence[Action]:
        """The interaction's available actions."""

        if not isinstance(self._actions[0], collections.abc.Hashable):
            self._actions = list(map(self._make_hashable,self._actions))

        return self._actions

    @property
    def rewards(self) -> Sequence[float]:
        """The reward for each action in the interaction."""
        return self._rewards


class SimulatedEnvironment(Environment):
    """An environment made from SimulatedInteractions."""
    
    @abstractmethod
    def read(self) -> Iterable[SimulatedInteraction]:
        """The sequence of interactions in the environment.

        Remarks:
            This function should always be "re-iterable".
        """
        ...

class MemorySimulation(SimulatedEnvironment):
    """A simulation implementation created from in memory sequences of contexts, actions and rewards."""

    def __init__(self, interactions: Sequence[SimulatedInteraction], params: Dict[str, Any] = {}) -> None:
        """Instantiate a MemorySimulation.

        Args:
            interactions: The sequence of interactions in this simulation.
        """
        self._interactions = interactions
        self._params = params

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    def read(self) -> Iterable[SimulatedInteraction]:
        return self._interactions
