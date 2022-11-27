from operator import eq
from collections import abc
from numbers import Number
from abc import abstractmethod, ABC
from typing import Any, Union, Iterable, Sequence, Mapping, TypeVar, Iterator
from coba.backports import Literal

from coba.pipes import Source, SourceFilters, Filter
from coba.exceptions import CobaException

Context   = Union[None, str, Number, 'HashableSeq', 'HashableMap']
Action    = Union[str, Number, 'HashableSeq', 'HashableMap']
Actions   = Sequence[Action]

T = TypeVar('T')

class HashableMap(Mapping):
    def __init__(self, item: Mapping) -> None:
        self._item = item

    def __getitem__(self, key):
        return self._item[key]

    def __iter__(self):
        return iter(self._item)

    def __len__(self):
        return len(self._item)

    def _get_hash(self):
        _hash = hash(tuple(self._item.items()))
        self._hash = _hash
        self._get_hash = lambda: _hash
        return _hash

    def __hash__(self) -> int:
        return self._get_hash()

    def __repr__(self) -> str:
        return repr(self._item)

    def __str__(self) -> str:
        return str(self._item)

class HashableSeq(Sequence):

    def __init__(self, item: Mapping) -> None:
        self._item = item

    def __getitem__(self, index):
        return self._item[index]

    def __len__(self) -> int:
        return len(self._item)

    def __eq__(self, o: object) -> bool:
        try:
            return len(self._item) == len(o) and all(map(eq, self._item, o))
        except:
            return False
    
    def _get_hash(self):
        _hash = hash(tuple(self._item))
        self._hash = _hash
        self._get_hash = lambda:_hash
        return _hash

    def __hash__(self) -> int:
        return self._get_hash()

    def __repr__(self) -> str:
        return repr(self._item)

    def __str__(self) -> str:
        return str(self._item)

class Feedback(ABC):
    @abstractmethod
    def eval(self, arg: Action) -> Any:
        ...

class SequenceFeedback(Feedback):
    def __init__(self, actions: Sequence[Action], values: Sequence[Any]) -> None:
        self._actions = actions
        self._values  = list(values)

    def eval(self, arg: Any) -> Any:
        return self._values[self._actions.index(arg)]

    def __getitem__(self, index: int) -> Any:
        return self._values[index]
    
    def __len__(self) -> int:
        return len(self._values)

    def __eq__(self, o: object) -> bool:
        return isinstance(o,abc.Sequence) and list(o) == list(self._values)

class Reward(ABC):
    
    @abstractmethod
    def eval(self, arg: Action) -> float:
        ...

    @abstractmethod
    def argmax(self) -> Action:
        ...
    
    def max(self) -> float:
        return self.eval(self.argmax())

class L1Reward(Reward):

    def __init__(self, label: float) -> None:
        self._label = label
        self.eval = lambda arg: arg-label if label > arg else label-arg 

    def argmax(self) -> float:
        return self._label

    def eval(self, arg: Any) -> float:#pragma: no cover
        #defined in __init__ for performance
        raise NotImplementedError("This should have been defined in __init__.")

class HammingReward(Reward):
    def __init__(self, label: Sequence[Any]) -> None:
        self._is_seq = isinstance(label,abc.Sequence) and not isinstance(label,(str,tuple))
        self._label  = set(label) if self._is_seq else label

    def argmax(self) -> Union[Any,Sequence[Any]]:
        return self._label

    def eval(self, arg: Union[Any, Sequence[Any]]) -> float:
        if self._is_seq:
            return len(self._label.intersection(arg))/len(self._label)
        else:
            return int(self._label==arg)

class BinaryReward(Reward):

    def __init__(self, value: Action):
        self._argmax = value

    def argmax(self) -> Action:
        return self._argmax

    def eval(self, arg: Action) -> float:
        return float(self._argmax==arg)

class ScaleReward(Reward):

    def __init__(self, reward: Reward, shift: float, scale: float, target: Literal["argmax","value"]) -> None:

        if target not in ["argmax","value"]:
            raise CobaException("An unrecognized scaling target was requested for a Reward.")

        self._shift  = shift
        self._scale  = scale
        self._target = target
        self._reward = reward

        old_argmax = reward.argmax()
        new_argmax = (old_argmax+shift)*scale if target == "argmax" else old_argmax

        self._old_argmax = old_argmax
        self._new_argmax = new_argmax

        old_eval = reward.eval

        if self._target == "argmax":
            self.eval = lambda arg: old_eval(old_argmax + (arg-new_argmax))
        if self._target == "value":
            self.eval = lambda arg: (old_eval(arg)+shift)*scale

    def argmax(self) -> float:
        return self._new_argmax

    def eval(self, arg: Any) -> float: #pragma: no cover
        #defined in __init__ for performance
        raise NotImplementedError("This should have been defined in __init__.")

class SequenceReward(Reward):
    def __init__(self, actions: Sequence[Action], values: Sequence[T]) -> None:
        if len(actions) != len(values):
            raise CobaException("Interaction reward counts must equal action counts.")
        self._actions = actions
        self._values  = values

    def eval(self, arg: Any) -> T:
        return self._values[self._actions.index(arg)]

    def argmax(self) -> Action:
        max_r = self._values[0]
        max_a = self._actions[0]
        for a,r in zip(self._actions,self._values):
            if r > max_r: 
                max_a = a
                max_r = r
        return max_a

    def __getitem__(self, index: int) -> T:
        return self._values[index]
    
    def __len__(self) -> int:
        return len(self._actions)

    def __iter__(self) -> Iterator[float]:
        return iter(map(self.__getitem__,range(len(self._values))))

    def __eq__(self, o: object) -> bool:
        try:
            return list(o) == list(self)
        except:#pragma: no cover
            return False

class MulticlassReward(Reward):
    def __init__(self, actions: Sequence[Action], label: Action) -> None:
        self._actions = actions
        self._label   = label

    def eval(self, action: Any) -> T:
        return int(self._label == action)

    def argmax(self) -> Action:
        return self._label

    def __getitem__(self, index: int) -> T:
        return self._actions[index] == self._label
    
    def __len__(self) -> int:
        return len(self._actions)

    def __iter__(self) -> Iterator[float]:
        return iter(map(self.__getitem__,range(len(self._actions))))

    def __eq__(self, o: object) -> bool:
        return isinstance(o,abc.Sequence) and list(o) == list(self)

class Interaction(dict):
    """An individual interaction that occurs in an Environment."""
    def __init__(self,
        type    : str,
        context : Context,
        actions : Actions,
        rewards : Union[Reward, Sequence[float]],
        **kwargs) -> None:
        ...
        """Instantiate SimulatedInteraction.

        Args
            context : Features describing the interaction's context.
            actions : Features describing available actions during the interaction.
            rewards : The reward for each action in the interaction.
            **kwargs: Any additional information.
        """

        self['type']    = type
        self['context'] = context
        self['actions'] = actions

        if isinstance(rewards,(list,tuple)):
            self['rewards'] = SequenceReward(actions,rewards)
            self['is_discrete'] = True
        else:
            self['rewards'] = rewards
            self['is_discrete'] = actions and len(actions)>0

        if kwargs: self.update(kwargs)

    @property
    def context(self):
        return self['context']

    @property
    def actions(self):
        return self['actions']

    @property
    def rewards(self):
        return self['rewards']

    @property
    def is_discrete(self):
        return self['is_discrete']

    @property
    def kwargs(self):
        return {k:self[k] for k in self.keys()-{'context','actions','rewards','is_discrete','type'} }

    def copy(self): # don't delegate w/ super - dict.copy() -> dict :(
        return type(self)(**self)

class SimulatedInteraction(Interaction):
    """Simulated data that describes an interaction where the choice is up to you."""

    def __init__(self,
        context : Context,
        actions : Actions,
        rewards : Union[Reward, Sequence[float]],
        **kwargs) -> None:

        """Instantiate SimulatedInteraction.

        Args
            context : Features describing the interaction's context.
            actions : Features describing available actions during the interaction.
            rewards : The reward for each action in the interaction.
            **kwargs: Any additional information.
        """
        kwargs.pop('type',None)
        super().__init__('simulated',context,actions,rewards,**kwargs)
        if actions is None: kwargs.pop('actions')

class GroundedInteraction(Interaction):
    """Logged data that describes an interaction where the choice was already made."""
    def __init__(self,
        context: Context,
        actions: Actions,
        rewards: Union[Reward, Sequence[float]],
        feedbacks: Union[Feedback, Sequence[Any]],
        **kwargs) -> None:
        ...
        """Instantiate GroundedInteraction.

        Args
            context: Features describing the interaction's context.
            actions: Features describing available actions during the interaction.
            rewards: The reward for each action in the interaction.
            feedbacks: The feedback for each action in the interaction.
            **kwargs: Additional information that should be recorded in the interactions table of an experiment result.
        """
        kwargs.pop('type',None)
        if isinstance(feedbacks,(list,tuple)):
            self['feedbacks'] = SequenceFeedback(actions,feedbacks)
        else:
            self['feedbacks'] = feedbacks
        super().__init__('grounded',context, actions, rewards, **kwargs)

    @property
    def feedbacks(self):
        return self['feedbacks']

    @property
    def kwargs(self):
        return {k:self[k] for k in super().kwargs.keys()-{'feedbacks'} }

class LoggedInteraction(Interaction):
    """Logged data that describes an interaction where the choice was already made."""

    def __init__(self,
        context: Context,
        action: Action,
        reward: float,
        probability: float = None,
        actions: Actions = None,
        rewards: Union[Reward, Sequence[float]] = None,
        **kwargs) -> None:
        """Instantiate LoggedInteraction.

        Args
            context: Features describing the logged context.
            action: Features describing the action taken by the logging policy.
            reward: The reward that was revealed when the logged action was taken.
            probability: The probability that the logged action was taken. That is P(action|context,actions,logging policy).
            actions: All actions that were availble to be taken when the logged action was taken. Necessary for OPE.
            rewards: The rewards to use for off policy evaluation. These rewards will not be shown to any learners. They will
                only be recorded in experimental results. If probability and actions is provided and rewards is None then 
                rewards will be initialized using the IPS estimator.
            **kwargs : Any additional information.
        """
        kwargs.pop('type',None)
        if probability and actions and rewards is None:
            rewards = [ int(a==action)*reward/probability for a in actions ]

        self['action']      = action
        self['probability'] = probability
        self['reward']      = reward

        super().__init__('logged',context, actions, rewards,**kwargs)
        #if actions is None: self.pop('actions')

    @property
    def action(self):
        return self['action']

    @property
    def probability(self):
        return self['probability']

    @property
    def reward(self):
        return self['reward']

    @property
    def kwargs(self):
        return {k:self[k] for k in super().kwargs.keys()-{'action','probability','reward'} }


class EnvironmentFilter(Filter[Iterable[Interaction],Iterable[Interaction]], ABC):
    """A filter that can be applied to an Environment."""

    @abstractmethod
    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        """Apply a filter to an Environment's interactions."""
        ...

class Environment(Source[Iterable[Interaction]], ABC):
    """An Environment that produces Contextual Bandit data"""

    @property
    def params(self) -> Mapping[str,Any]: # pragma: no cover
        """Paramaters describing the simulation.

        Remarks:
            These will become columns in the environments table of experiment results.
        """
        return {}

    @abstractmethod
    def read(self) -> Iterable[Interaction]:
        """The sequence of interactions in the simulation.

        Remarks:
            This function should always be "re-iterable".
        """
        ...

    def __str__(self) -> str:
        return str(self.params) if self.params else self.__class__.__name__

class SimulatedEnvironment(Environment):
    """An environment made from SimulatedInteractions."""

    @abstractmethod
    def read(self) -> Iterable[SimulatedInteraction]:
        """The sequence of interactions in the environment.

        Remarks:
            This function should always be "re-iterable".
        """
        ...

class SafeEnvironment(Environment):
    """A wrapper for environment-likes that guarantees interface consistency."""

    def __init__(self, environment: Environment) -> None:
        """Instantiate a SafeEnvironment.

        Args:
            environment: The environment we wish to make sure has the expected interface
        """

        self._environment = environment if not isinstance(environment, SafeEnvironment) else environment._environment

    @property
    def params(self) -> Mapping[str, Any]:
        try:
            params = self._environment.params
        except AttributeError:
            params = {}

        if "type" not in params:

            if isinstance(self._environment, SourceFilters):
                params["type"] = self._environment._source.__class__.__name__
            else:
                params["type"] = self._environment.__class__.__name__

        return params

    def read(self) -> Iterable[Interaction]:
        return self._environment.read()

    def __str__(self) -> str:
        params = dict(self.params)
        tipe   = params.pop("type")

        if len(params) > 0:
            return f"{tipe}({','.join(f'{k}={v}' for k,v in params.items())})"
        else:
            return tipe
