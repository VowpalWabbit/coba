from abc import abstractmethod
from typing import Sequence, Any, Iterable, Optional

from coba.environments.primitives import Context, Action, Environment, Interaction

class LoggedInteraction(Interaction):
    """Logged data that describes an interaction where the choice was already made."""

    def __init__(self,
        context: Context,
        action : Action,
        reward : float,
        probability: Optional[float] = None,
        actions: Optional[Sequence[Action]] = None,
        **kwargs) -> None:
        ...
        """Instantiate LoggedInteraction.

        Args
            context    : Features describing the context that was logged. This should be `None` for multi-armed bandit simulations.
            action     : Features describing the taken action for logging purposes.
            reward     : The reward that was revealed when the logged action was taken.
            probability: The probability that the logged action was taken.
            actions    : All actions that were availble to be taken when the logged action was taken.
            **kwargs   : Additional information that should be recorded in the interactions table of an experiment result. If
                any data is a sequence with length equal to actions only the data at the selected action index will be recorded.
        """

        self._context = self._make_hashable(context)
        self._action  = self._make_hashable(action)
        self._reward  = reward
        self._prob    = probability
        self._actions = actions

        super().__init__(self._context, **kwargs)

    @property
    def action(self) -> Action:
        """The action that was taken."""
        return self._action

    @property
    def reward(self) -> float:
        """The reward that was observed after taking the given action."""
        return self._reward

    @property
    def probability(self) -> Optional[float]:
        """The probability the action was taken."""
        return self._prob

    @property
    def actions(self) -> Optional[Sequence[Action]]:
        """The altnerative actions that were available."""
        return self._actions

class LoggedEnvironment(Environment):
    """An Environment made from LoggedInteractions."""

    @abstractmethod
    def read(self) -> Iterable[LoggedInteraction]:
        """The sequence of interactions in a simulation.

        Remarks:
            This function should always be "re-iterable".
        """
        ...
