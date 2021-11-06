import collections

from abc import abstractmethod, ABC
from typing import Optional, Sequence, Hashable, Any, Union, Iterable, overload, Dict

from coba.utilities import HashableDict
from coba.pipes import Source

Action  = Union[Hashable, HashableDict]
Context = Union[None, Hashable, HashableDict]

class SimulatedInteraction:
    """A class to contain all data needed to represent an interaction in a bandit simulation."""

    @overload
    def __init__(self, 
        context: Context, 
        actions: Sequence[Action], *, 
        rewards: Sequence[float], 
        **kwargs: Sequence[Any]) -> None:
        ...
        """Instantiate Interaction.

        Args
            context : Features describing the interaction's context. Will be `None` for multi-armed bandit simulations.
            actions : Features describing available actions in the interaction.
            rewards : The rewards that will be revealed to learners based on the action that is taken.
            **kwargs: Additional information beyond reward that will be recorded in the results of a benchmark.
        """

    @overload
    def __init__(self, 
        context: Context, 
        actions: Sequence[Action], *, 
        reveals: Sequence[Any], 
        rewards: Optional[Sequence[float]] = None, 
        **kwargs: Sequence[Any]) -> None:
        ...
        """Instantiate Interaction.

        Args
            context : Features describing the interaction's context. Will be `None` for multi-armed bandit simulations.
            actions : Features describing available actions in the interaction.
            reveals : This will be revealed to learners based on the action that is taken. This can be any kind of information.
            rewards : An optional scalar value. This won't be revealed to learners. This will only be used for plotting.
            **kwargs: Additional information beyond whit is revealed that will be recorded in the results of a benchmark.
        """
    
    def __init__(self, *args, **kwargs) -> None:

        assert len(args) == 2, "An unexpected number of positional arguments was supplied to Interaction."
        assert kwargs.keys() & {"rewards", "reveals"}, "Interaction requires either a rewards or reveals kwarg."

        self._only_rewards = "rewards" in kwargs and "reveals" not in kwargs

        context = args[0]
        actions = args[1]
 
        rewards = kwargs.pop("rewards", None)
        reveals = kwargs.pop("reveals", None)
        extras  = kwargs

        assert not rewards or len(actions) == len(rewards), "Interaction rewards must match action length."
        assert not reveals or len(actions) == len(reveals), "Interaction reveals must match action length."

        context = self._flatten(context)
        actions = [ self._flatten(action) for action in actions ]

        self._context =  context if not isinstance(context,dict) else HashableDict(context)
        self._actions = [ action if not isinstance(action ,dict) else HashableDict(action) for action in actions ]

        self._rewards = rewards
        self._reveals = reveals
        self._extras  = extras

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
    def rewards(self) -> Optional[Sequence[float]]:
        """The reward associated with each action."""
        return self._rewards

    @property
    def reveals(self) -> Sequence[Any]:
        """The information revealed to learners based on the selected action."""
        return self._reveals if self._reveals else self._rewards

    @property
    def extras(self) -> Dict[str,Any]:
        """Additional information regarding this interaction."""
        return self._extras    

    @property
    def results(self) -> Dict[str,Any]:
        
        results_dict = dict(self._extras)

        if self._rewards: results_dict["rewards"] = self._rewards
        if self._reveals: results_dict["reveals"] = self._reveals

        return results_dict

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
