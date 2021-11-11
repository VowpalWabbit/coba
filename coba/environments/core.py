import collections

from abc import abstractmethod, ABC
from typing import Optional, Sequence, Hashable, Any, Union, Iterable, overload, Dict

from coba.utilities import HashableDict
from coba.pipes import Source

Action  = Union[Hashable, HashableDict]
Context = Union[None, Hashable, HashableDict]

class SimulatedInteraction:
    """A class to contain all data needed to represent an interaction in a simulated bandit interaction."""

    @overload
    def __init__(self, 
        context: Context, 
        actions: Sequence[Action],
        *
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
        actions: Sequence[Action], *, 
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

        context = self._flatten(args[0])
        actions = [ self._flatten(action) for action in args[1] ]

        self._context =  context if not isinstance(context,dict) else HashableDict(context)
        self._actions = [ action if not isinstance(action ,dict) else HashableDict(action) for action in actions ]
        self._kwargs  = kwargs

    def _is_sparse(self, feats):

        if isinstance(feats,dict):
            return True

        if not isinstance(feats, collections.Sequence):
            return False

        if len(feats) != 2:
            return False

        if not isinstance(feats[0], collections.Sequence) or not isinstance(feats[1],collections.Sequence):
            return False

        if len(feats[0]) != len(feats[1]):
            return False

        if isinstance(feats[0],str) or isinstance(feats[1],str):
            return False

        return True

    def _flatten(self, feats):

        if not isinstance(feats, collections.Sequence) or isinstance(feats, (str,dict)):
            return feats

        if not self._is_sparse(feats):

            flattened_dense_values = []

            for val in feats:
                if isinstance(val,(list,tuple,bytes)):
                    flattened_dense_values.extend(val)
                else:
                    flattened_dense_values.append(val)
            
            return tuple(flattened_dense_values)
        else:
            keys = []
            vals = []

            for key,val in zip(*feats):

                if isinstance(val, (list,tuple,bytes)):
                    for sub_key,sub_val in enumerate(val):
                        keys.append(f"{key}_{sub_key}")
                        vals.append(sub_val)
                else:
                    keys.append(key)
                    vals.append(val)

            return HashableDict(zip(keys,vals))

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


class LoggedInteraction:
    """
    TODO: docs
    """
    def __init__(self, context: Context, action: Action, reward: float, actions: Sequence[Action] = None, probability: float = None) -> None:

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
    def actions(self) -> Sequence[Action]:
        """The actions that were available to take."""
        return self._actions

    @property
    def probability(self) -> Optional[Sequence[float]]:
        """The probability that the given action was taken."""
        return self._probability

class Environment:
    pass

class Simulation(Source[Iterable[SimulatedInteraction]], ABC):
    """The simulation interface."""

    @property
    @abstractmethod
    def params(self) -> Dict[str,Any]:
        """Paramaters describing the simulation.

        Remarks:
            These will be simulation columns in coba.benchmark.Result.
        """
        ...
    
    @abstractmethod
    def read(self) -> Iterable[SimulatedInteraction]:
        """The sequence of interactions in a simulation.

        Remarks:
            This function should always be "re-iterable".
        """
        ...

class WarmStart(Source[Iterable[Union[LoggedInteraction,SimulatedInteraction]]], ABC):
    """
    TODO: docs
    """
    @property
    @abstractmethod
    def params(self) -> Dict[str,Any]:
        """Paramaters describing the simulation.

        Remarks:
            These will be simulation columns in coba.benchmark.Result.
        """
        ...
    
    @abstractmethod
    def read(self) -> Iterable[SimulatedInteraction]:
        """The sequence of interactions in a simulation.

        Remarks:
            This function should always be "re-iterable".
        """
        ...
