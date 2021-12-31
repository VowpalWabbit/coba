from abc import abstractmethod
from typing import Sequence, Dict, Any, Iterable, overload, Optional

from coba.environments.primitives import Context, Action, Environment, Interaction

class LoggedInteraction(Interaction):

    @overload
    def __init__(self,
        context: Context,
        action : Action,
        *,
        reward: float,
        probability: Optional[float] = None,
        actions: Optional[Sequence[Action]] = None,
        **kwargs) -> None:
        ...
        """Instantiate LoggedInteraction.

        Args
            context : Features describing the context that was logged. This should be `None` for multi-armed bandit simulations.
            action  : Features describing the taken action for logging purposes.
            reward  : The reward that was revealed when the logged action was taken.
            probability: The probability that the logged action was taken.
            actions    : All actions that were availble to be taken when the logged action was taken.
            **kwargs   : Additional information that should be recorded in the interactions table of an experiment result. If 
                any data is a sequence with length equal to actions only the data at the selected action index will be recorded.
        """

    @overload
    def __init__(self,
        context: Context,
        action : Action,
        *,
        reveal     : Any,
        probability: Optional[float] = None,
        actions    : Optional[Sequence[Action]] = None,
        **kwargs) -> None:
        ...
        """Instantiate LoggedInteraction.

        Args
            context    : Features describing the context that was logged. This should be `None` for multi-armed bandit simulations.
            action     : Features describing the taken action for logging purposes.
            reveal     : The information that was revealed when the logged action was taken.
            probability: The probability that the logged action was taken.
            actions    : All actions that were availble to be taken when the logged action was taken.
            **kwargs   : Additional information that should be recorded in the interactions table of an experiment result. If 
                any data is a sequence with length equal to actions only the data at the selected action index will be recorded.
        """

    def __init__(self, context: Context, action: Action, **kwargs) -> None:

        self._context     = self._hashable(context)
        self._action      = self._hashable(action)
        self._kwargs      = kwargs

        if "actions" in self._kwargs:
            self._kwargs["actions"] = list(map(self._hashable,self._kwargs["actions"]))

        super().__init__(self._context)

    @property
    def action(self) -> Action:
        """The action that was taken."""
        return self._action

    @property
    def kwargs(self) -> Dict[str,Any]:
        return self._kwargs

class LoggedEnvironment(Environment):
    """The interface for an environment made with logged bandit data."""
    
    @abstractmethod
    def read(self) -> Iterable[LoggedInteraction]:
        """The sequence of interactions in a simulation.

        Remarks:
            This function should always be "re-iterable".
        """
        ...

