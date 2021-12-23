from abc import abstractmethod, ABC

from coba.typing import Optional, Sequence, Hashable, Any, Union, Iterable, overload, Dict
from coba.utilities import HashableDict
from coba.pipes import Source, Filter

Action  = Union[Hashable, HashableDict]
Context = Union[None, Hashable, HashableDict]

class Interaction:
    def __init__(self, context: Context) -> None:
        self._context = context

    @property
    def context(self) -> Context:
        """The context in which an action was taken."""
        return self._context

    def _hashable(self, feats):

        if isinstance(feats, dict):
            return HashableDict(feats)

        if isinstance(feats,list):
            return tuple(feats)

        return feats

class SimulatedInteraction(Interaction):
    """A class to contain all data needed to represent an interaction in a simulated bandit interaction."""

    @overload
    def __init__(self,
        context: Context,
        actions: Sequence[Action],
        *,
        rewards: Sequence[float],
        **kwargs) -> None:
        ...
        """Instantiate SimulatedInteraction.

        Args
            context : Features describing the interaction's context. This should be `None` for multi-armed bandit simulations.
            actions : Features describing available actions in the interaction.
            rewards : The reward that will be revealed to learners based on the taken action. We require len(rewards) == len(actions).
            **kwargs: Additional information that should be recorded in the interactions table of an experiment result. If any
                data is a sequence with length equal to actions only the data at the selected action index will be recorded.
        """

    @overload
    def __init__(self,
        context: Context,
        actions: Sequence[Action], 
        *,
        reveals: Sequence[Any],
        **kwargs) -> None:
        ...
        """Instantiate SimulatedInteraction.

        Args
            context : Features describing the interaction's context. Will be `None` for multi-armed bandit simulations.
            actions : Features describing available actions in the interaction.
            reveals : The data that will be revealed to learners based on the selected action. We require len(reveals) == len(actions).
                When working with non-scalar data use "reveals" instead of "rewards" to make it clear to Coba the data is non-scalar.
            **kwargs: Additional information that should be recorded in the interactions table of an experiment result. If any
                data is a sequence with length equal to actions only the data at the selected action index will be recorded.
        """

    @overload
    def __init__(self, 
        context: Context, 
        actions: Sequence[Action], 
        *,
        rewards : Sequence[float],
        reveals : Sequence[Any],
        **kwargs) -> None:
        ...
        """Instantiate SimulatedInteraction.

        Args
            context : Features describing the interaction's context. Will be `None` for multi-armed bandit simulations.
            actions : Features describing available actions in the interaction.
            rewards : A sequence of scalar values representing reward. When both rewards and reveals are provided only 
                reveals will be shown to the learner when an action is selected. The reward values will only be used 
                by Coba when plotting experimental results. We require that len(rewards) == len(actions).
            reveals : The data that will be revealed to learners based on the selected action. We require len(reveals) == len(actions).
                When working with non-scalar data use "reveals" instead of "rewards" to make it clear to Coba the data is non-scalar.
            **kwargs: Additional information that should be recorded in the interactions table of an experiment result. If any
                data is a sequence with length equal to actions only the data at the selected action index will be recorded.
        """

    def __init__(self, context: Context, actions: Sequence[Action], **kwargs) -> None:

        assert kwargs.keys() & {"rewards", "reveals"}, "Interaction requires either a rewards or reveals keyword warg."

        assert "rewards" not in kwargs or len(actions) == len(kwargs["rewards"]), "Interaction rewards must match action length."
        assert "reveals" not in kwargs or len(actions) == len(kwargs["reveals"]), "Interaction reveals must match action length."

        self._context = self._hashable(context)
        self._actions = list(map(self._hashable,actions))

        self._kwargs  = kwargs

        super().__init__(self._context)

    @property
    def context(self) -> Context:
        """The interaction's context description."""

        return self._context

    @property
    def actions(self) -> Sequence[Action]:
        """The interaction's available actions."""

        return self._actions

    @property
    def kwargs(self) -> Dict[str,Any]:
        return self._kwargs

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

class Environment(Source[Iterable[Interaction]], ABC):

    @property
    @abstractmethod
    def params(self) -> Dict[str,Any]:
        """Paramaters describing the simulation.

        Remarks:
            These will become columns in the environments data of an experiment result.
        """
        ...
    
    @abstractmethod
    def read(self) -> Iterable[Interaction]:
        """The sequence of interactions in a simulation.

        Remarks:
            This function should always be "re-iterable".
        """
        ...

    def __str__(self) -> str:
        return str(self.params) if self.params else self.__class__.__name__

class SimulatedEnvironment(Environment):
    """The interface for a simulated environment."""
    
    @abstractmethod
    def read(self) -> Iterable[SimulatedInteraction]:
        """The sequence of interactions in a simulation.

        Remarks:
            This function should always be "re-iterable".
        """
        ...

class LoggedEnvironment(Environment):
    """The interface for an environment made with logged bandit data."""
    
    @abstractmethod
    def read(self) -> Iterable[SimulatedInteraction]:
        """The sequence of interactions in a simulation.

        Remarks:
            This function should always be "re-iterable".
        """
        ...

class WarmStartEnvironment(Environment):
    """The interface for an environment made with logged bandit data and simulated interactions."""
       
    @abstractmethod
    def read(self) -> Iterable[Interaction]:
        """The sequence of interactions in a simulation.

        Remarks:
            This function should always be "re-iterable".
        """
        ...

class EnvironmentFilter(Filter[Iterable[Interaction],Iterable[Interaction]], ABC):

    @property
    @abstractmethod
    def params(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[SimulatedInteraction]:
        """Apply a filter to a Simulation's interactions."""
        ...

    def __str__(self) -> str:
        return str(self.params) if self.params else self.__class__.__name__
