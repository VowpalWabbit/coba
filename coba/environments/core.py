from abc import abstractmethod, ABC
from typing import Optional, Sequence, Hashable, Any, Union, Iterable, overload, Dict

from coba.utilities import HashableDict
from coba.pipes import Source

Action  = Union[Hashable, HashableDict]
Context = Union[None, Hashable, HashableDict]

class Interaction:
    pass

class SimulatedInteraction(Interaction):
    """A class to contain all data needed to represent an interaction in a simulated bandit interaction."""

    @overload
    def __init__(self,
        context: Context,
        actions: Sequence[Action],
        *,
        rewards: Sequence[float],
        **kwargs: Sequence[Any]) -> None:
        ...
        """Instantiate Interaction.

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
        **kwargs: Sequence[Any]) -> None:
        ...
        """Instantiate Interaction.

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
        **kwargs: Sequence[Any]) -> None:
        ...
        """Instantiate Interaction.

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

    def __init__(self, *args, **kwargs) -> None:

        assert len(args) == 2, "An unexpected number of positional arguments was supplied to Interaction."
        assert kwargs.keys() & {"rewards", "reveals"}, "Interaction requires either a rewards or reveals keyword warg."
  
        assert "rewards" not in kwargs or len(args[1]) == len(kwargs["rewards"]), "Interaction rewards must match action length."
        assert "reveals" not in kwargs or len(args[1]) == len(kwargs["reveals"]), "Interaction reveals must match action length."

        self._context = self._hashable(args[0])
        self._actions = [ self._hashable(action) for action in args[1] ]

        self._kwargs  = kwargs

    def _hashable(self, feats):

        if isinstance(feats, dict):
            return HashableDict(feats)

        if isinstance(feats,list):
            return tuple(feats)

        return feats

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

    def __init__(self, 
        context: Context, 
        action: Action, 
        reward: float, 
        probability: Optional[float] = None,
        actions: Optional[Sequence[Action]] = None) -> None:

        self._context     = context
        self._action      = action
        self._reward      = reward
        self._actions     = actions
        self._probability = probability

    @property
    def context(self) -> Context:
        """The context in which an action was taken."""
        return self._context

    @property
    def action(self) -> Action:
        """The action that was taken."""
        return self._context

    @property
    def reward(self) -> float:
        """The reward that was observed when the action was taken."""
        return self._reward

    @property
    def actions(self) -> Optional[Sequence[Action]]:
        """The actions that were available to take."""
        return self._actions

    @property
    def probability(self) -> Optional[float]:
        """The probability that the given action was taken."""
        return self._probability

class Environment:
    pass

class SimulatedEnvironment(Environment, Source[Iterable[SimulatedInteraction]], ABC):
    """The interface for a simulated environment."""

    @property
    @abstractmethod
    def params(self) -> Dict[str,Any]:
        """Paramaters describing the simulation.

        Remarks:
            These will become columns in the environments data of an experiment result.
        """
        ...
    
    @abstractmethod
    def read(self) -> Iterable[SimulatedInteraction]:
        """The sequence of interactions in a simulation.

        Remarks:
            This function should always be "re-iterable".
        """
        ...

class LoggedEnvironment(Environment, Source[Iterable[LoggedInteraction]], ABC):
    """The interface for an environment made with logged bandit data."""

    @property
    @abstractmethod
    def params(self) -> Dict[str,Any]:
        """Paramaters describing the simulation.

        Remarks:
            These will become columns in the environments data of an experiment result.
        """
        ...
    
    @abstractmethod
    def read(self) -> Iterable[SimulatedInteraction]:
        """The sequence of interactions in a simulation.

        Remarks:
            This function should always be "re-iterable".
        """
        ...

class WarmStartEnvironment(Environment, Source[Iterable[Interaction]], ABC):
    """The interface for an environment made with logged bandit data and simulated interactions."""
   
    @property
    @abstractmethod
    def params(self) -> Dict[str,Any]:
        """Paramaters describing the simulation.

        Remarks:
            These will become columns in the environments data of an experiment result.
        """
        ...
    
    @abstractmethod
    def read(self) -> Iterable[SimulatedInteraction]:
        """The sequence of interactions in a simulation.

        Remarks:
            This function should always be "re-iterable".
        """
        ...
