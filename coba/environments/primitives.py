from collections import abc
from numbers import Number
from abc import abstractmethod, ABC
from typing import Any, Union, Iterable, Sequence, Mapping, Optional, TypeVar, Generic, overload
from coba.backports import Literal

from coba.utilities import HashableDict
from coba.pipes import Source, SourceFilters
from coba.exceptions import CobaException

Context   = Union[None, str, Number, tuple, HashableDict]
Action    = Union[str, Number, tuple, HashableDict]
Actions   = Sequence[Action]

T = TypeVar('T')

class Feedback(ABC, Generic[T]):
    @abstractmethod
    def eval(self, arg: Action) -> T:
        ...

class DiscreteFeedback(Sequence, Feedback[T]):
    def __init__(self, actions: Sequence[Action], values: Sequence[T]) -> None:
        self._actions = actions
        self._values  = values
        self._lookup  = dict(zip(actions,values))

    def eval(self, arg: Any) -> T:
        return self._lookup[arg]

    def __getitem__(self, index: int) -> T:
        return self._values[index]
    
    def __len__(self) -> int:
        return len(self._values)

    def __eq__(self, o: object) -> bool:
        return isinstance(o,abc.Sequence) and list(o) == list(self._values)

class Reward(Feedback[float]):
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

class DiscreteReward(Reward, DiscreteFeedback[float]):

    def argmax(self) -> Action:
        max_r = -float('inf')
        max_a = None
        for a,r in zip(self._actions,self._values):
            if r > max_r: 
                max_a = a
                max_r = r
        return max_a

class Interaction:
    """An individual interaction that occurs in an Environment."""

    def __init__(self, 
        context: Context, 
        actions: Actions,
        rewards: Union[Reward, Sequence[float]], 
        **kwargs) -> None:
        """Instantiate an Interaction.

        Args:
            context: The context in which the interaction occured.
        """
        self._context = context
        self._actions = actions
        self._rewards = rewards
        self._kwargs  = kwargs
        self._hashed  = set()

        try:
            if self.is_discrete and len(rewards) != len(actions):
                raise CobaException("An interaction's reward count must equal its action count.")
        except CobaException:
            raise
        except Exception:
            pass

    @property
    def is_discrete(self) -> bool:
        try:
            return len(self.actions) > 0
        except:
            return False

    @property
    def context(self) -> Context:
        """The context in which the interaction occured."""
        if "context" not in self._hashed:
            self._context = self._make_hashable(self._context)
            self._hashed.add("context")
        return self._context

    @property
    def actions(self) -> Actions:
        """The actions available in the interaction."""
        if "actions" not in self._hashed:
            if self._actions is not None:
                try:
                    self._actions = list(map(self._make_hashable,self._actions))
                except:
                    pass
            self._hashed.add("actions")
        return self._actions

    @property
    def rewards(self) -> Reward:
        
        if "rewards" not in self._hashed:
            if self._rewards and self.actions:
                try:
                    self._rewards = DiscreteReward(self.actions,self._rewards)
                except (TypeError, AttributeError):
                    pass            
            self._hashed.add("rewards")

        return self._rewards

    @property
    def kwargs(self) -> Mapping[str,Any]:
        """Additional information associatd with the Interaction."""
        return self._kwargs

    def _make_hashable(self, feats):
        if isinstance(feats,abc.Hashable): 
            return feats
        try:
            return feats.to_builtin()
        except Exception:
            if isinstance(feats, abc.Sequence):
                return tuple(feats)
            if isinstance(feats, abc.Mapping):
                return HashableDict(feats)
            return feats

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

        super().__init__(context, actions, rewards, **kwargs)
        self._feedbacks = feedbacks

    @property
    def feedbacks(self) -> Feedback:
        """The feedback for each action in the interaction."""
        if "feedback" not in self._hashed:
            try:
                self._feedbacks = DiscreteFeedback(self.actions,self._feedbacks)
            except:
                pass
            self._hashed.add("feedback")

        return self._feedbacks

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
            **kwargs : Additional information that should be recorded in the interactions table of an experiment result. If
                any data is a sequence with length equal to actions only the data at the selected action index will be recorded.
        """
        self._action      = action
        self._reward      = reward
        self._probability = probability

        if probability and actions and rewards is None:
            try:
                rewards = [ int(a==action)*reward/probability for a in actions ]
            except:
                rewards = lambda a: int(a==action)*reward/probability

        super().__init__(context, actions, rewards, **kwargs)

    @property
    def action(self) -> Action:
        """The action that was taken."""
        if "action" not in self._hashed:
            self._action = self._make_hashable(self._action)
            self._hashed.add("action")
        return self._action

    @property
    def reward(self) -> float:
        """The reward that was observed after taking the given action."""
        return self._reward

    @property
    def probability(self) -> Optional[float]:
        """The probability the action was taken."""
        return self._probability

class SimulatedInteraction(Interaction):
    """Simulated data that describes an interaction where the choice is up to you."""

    def __init__(self,
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
            **kwargs: Additional information that should be recorded in the interactions table of an experiment result. If any
                data is a sequence with length equal to actions only the data at the selected action index will be recorded.
        """

        super().__init__(context, actions, rewards, **kwargs)

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
