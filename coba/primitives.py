"""Basic building blocks shared across modules."""

from ast import literal_eval
from abc import ABC, abstractmethod
from numbers import Number
from operator import eq
from collections import abc
from typing import Union, Tuple, Sequence, Mapping, Iterable, overload
from typing import TypeVar, Generic, Optional, Iterator, Any

from coba.exceptions import CobaException
from coba.utilities import try_else, minimize

Context = Union[None, str, Number, Sequence, Mapping]
Action  = Union[str, Number, Sequence, Mapping]
Actions = Union[None, Sequence['Action']]
Reward  = float
Prob    = float
Kwargs  = Mapping[str,Any]

Pred = Union[
    'Action',
    Tuple['Action','Prob'],
    Tuple['Action'       , 'Kwargs'],
    Tuple['Action','Prob', 'Kwargs'],
]

_T_out = TypeVar("_T_out", bound=Any, covariant    =True)
_T_in  = TypeVar("_T_in" , bound=Any, contravariant=True)

class Namespaces(dict):
    pass

class Pipe:

    @property
    def params(self) -> Mapping[str,Any]:
        """Parameters describing the pipe."""
        return { }

    def __str__(self) -> str:
        params = str(self.params)[1:-1] if self.params else ''
        return f"{type(self).__name__}({params})"

class Source(ABC, Pipe, Generic[_T_out]):
    """A pipe that can be read."""

    @abstractmethod
    def read(self) -> _T_out:
        """Read the item."""
        ...

class Filter(ABC, Pipe, Generic[_T_in, _T_out]):
    """A pipe that can modify an item."""

    @abstractmethod
    def filter(self, item: _T_in) -> _T_out:
        """Filter the item."""
        ...

class Sink(ABC, Pipe, Generic[_T_in]):
    """A pipe that writes item."""

    @abstractmethod
    def write(self, item: _T_in) -> None:
        """Write the item."""
        ...

class Line(ABC, Pipe):
    """A pipe that can be run."""

    @abstractmethod
    def run(self):
        """Run the pipe."""
        ...

class Rewards(ABC):
    """A function rewarding actions."""

    @abstractmethod
    def __call__(self, action: 'Action') -> 'Reward':
        """Get reward for action.

        Args:
            action: An action taken.

        Returns:
            Reward received for the action.

        :meta public:
        """
        ...

class Interaction(dict):
    """An interaction in an Environment.

    The only assumption made by Coba is that interactions are a `dict`.
    To distinguish between Interaction types we look at the interaction's
    key values rather than type.
    """
    __slots__=()

class SimulatedInteraction(Interaction):
    """An interaction with reward information for all actions."""
    __slots__=()

    def __init__(self,
        context: 'Context',
        actions: 'Actions',
        rewards: Union[Rewards, Sequence['Reward']],
        **kwargs) -> None:
        """Instantiate SimulatedInteraction.

        Args:
            context : Features describing the interaction's context.
            actions : Features describing available actions during the interaction.
            rewards : The reward for each action in the interaction.
            kwargs : Any additional information.
        """

        self['context'] = context
        self['actions'] = actions
        self['rewards'] = rewards

        if kwargs: self.update(kwargs)

class GroundedInteraction(Interaction):
    """An interaction with feedbacks for Interaction Grounded Learning."""
    __slots__=()

    def __init__(self,
        context  : 'Context',
        actions  : 'Actions',
        rewards  : Union[Rewards, Sequence['Reward']],
        feedbacks: Union[Rewards, Sequence['Reward']],
        **kwargs) -> None:
        """Instantiate GroundedInteraction.

        Args:
            context: Features describing the interaction's context.
            actions: Features describing available actions during the interaction.
            rewards: The reward for each action in the interaction.
            feedbacks: The feedback for each action in the interaction.
            **kwargs: Additional information that should be recorded in the interactions table of an experiment result.
        """

        self['context']   = context
        self['actions']   = actions
        self['rewards']   = rewards
        self['feedbacks'] = feedbacks

        if kwargs: self.update(kwargs)

class LoggedInteraction(Interaction):
    """An interaction with a reward and propensity score for an action."""
    __slots__ = ()

    def __init__(self,
        context    : 'Context',
        action     : 'Action',
        reward     : 'Reward',
        probability: 'Prob' = None,
        **kwargs) -> None:
        """Instantiate LoggedInteraction.

        Args:
            context: Features describing the logged context.
            action: Features describing the action taken by the logging policy.
            reward: The reward that was revealed when the logged action was taken.
            probability: The probability that the logged action was taken. That is P(action|context,actions,logging policy).
            **kwargs: Any additional information.
        """

        self['context'] = context
        self['action']  = action
        self['reward']  = reward

        if probability is not None:
            self['probability'] = probability

        if kwargs: self.update(kwargs)

class EnvironmentFilter(Filter[Iterable[Interaction],Iterable[Interaction]], ABC):
    """Modify an Environment."""

    @property
    def params(self) -> Mapping[str,Any]:
        """Parameters describing the modification."""
        return { }

    @abstractmethod
    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        """Apply filter to an Environment's interactions."""
        ...

class Environment(Source[Iterable[Interaction]], ABC):
    """A source of Interactions."""

    @property
    def params(self) -> Mapping[str,Any]: #pragma: no cover
        """Paramaters describing the Environment.

        Remarks:
            These will become columns in the environments table of experiment results.
        """
        return {}

    @abstractmethod
    def read(self) -> Iterable[Interaction]:
        """A sequence of interactions.

        Remarks:
            This function should always be "re-iterable".
        """
        ...

    def __str__(self) -> str:
        return str(self.params) if self.params else self.__class__.__name__

class Learner(ABC):
    """An agent that acts and learns."""

    @property
    def params(self) -> Mapping[str,Any]:
        """Parameters describing the learner (used for descriptive purposes only).

        Remarks:
            These will become columns in the learners table of experiment results.
        """
        return {}

    def score(self, context: 'Context', actions: 'Actions', action: 'Action') -> 'Prob':
        """Propensity score an action.

        Args:
            context: The current context.
            actions: The current set of actions that can be chosen.
            action: The action to propensity score.

        Returns:
            The propensity score of the given action. That is, P(action|context,actions).
        """
        raise NotImplementedError((
            "The `score` interface has not been implemented for this learner."
        ))

    def predict(self, context: 'Context', actions: 'Actions') -> 'Pred':
        """Predict which action to take in the context.

        Args:
            context: The current context. It will either be None (multi-armed bandit),
                a value (a single feature), a sequence of values (dense features), or a
                dictionary (sparse features).
            actions: The current set of actions to choose from in the given context.
                Each action will either be a value (a single feature), a sequence of values
                (dense features), or a dictionary (sparse features).

        Returns:
            A Prediction. Several prediction formats are supported. See the type-hint for these.
        """
        raise NotImplementedError((
            "The `predict` interface has not been implemented for this learner."
        ))

    def learn(self, context: 'Context', action: 'Action', reward: 'Reward', probability: 'Prob', **kwargs) -> None:
        """Learn about the action taken in the context.

        Args:
            context: The context in which the action was taken.
            action: The action that was taken.
            reward: The reward for the given context and action (feedback for IGL problems).
            probability: The probability the given action was taken.
            **kwargs: Optional information returned during prediction.
        """
        raise NotImplementedError((
            "The `learn` interface has not been implemented for this learner."
        ))

class Evaluator(ABC):
    """An evaluator for learners in environments."""

    @property
    def params(self) -> Mapping[str,Any]:
        """Parameters describing the evaluator (used for descriptive purposes only).

        Remarks:
            These will become columns in the evaluators table of experiment results.
        """
        return {}

    @abstractmethod
    def evaluate(self, environment: Optional[Environment], learner: Optional[Learner]) -> Union[Mapping[Any,Any],Iterable[Mapping[Any,Any]]]:
        """Evaluate the learner on the given interactions.

        Args:
            environment: The Environment we want to evaluate against.
            learner: The Learner that we wish to evaluate.

        Returns:
            Evaluation results
        """
        ...

def is_batch(item):
    return hasattr(item,'is_batch')

class Categorical(str):
    __slots__ = ('levels','as_int','as_onehot','as_onehot_l')

    def __new__(cls, value:str, levels: Sequence[str]) -> str:
        return str.__new__(Categorical,value)

    def __init__(self, value:str, levels: Sequence[str]) -> None:
        self.levels = levels
        self.as_int = levels.index(value)
        onehot = [0]*len(levels)
        onehot[self.as_int] = 1
        self.as_onehot = tuple(onehot)

    def __repr__(self) -> str:
        return f"Categorical('{self}',{self.levels})"

    def __reduce__(self):
        return Categorical, (str(self),list(map(str,self.levels)))

class Dense(ABC):
    __slots__=()

    @abstractmethod
    def __getitem__(self, key) -> Any:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator:
        ...

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._row, attr)

    def __eq__(self, o) -> bool:
        try:
            return len(self) == len(o) and all(map(eq, self, o))
        except:
            return False

    def copy(self) -> list:
        return list(iter(self))

class Dense_:
    ##Instantiating classes which inherit from Dense is moderately expensive due to the ABC checks.
    ##Therefore we keep Dense around for public API checks but internally we use Dense_ for inheritance.
    __slots__=('_row')

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._row, attr)

    def __eq__(self, o) -> bool:
        try:
            return len(self) == len(o) and all(map(eq, self, o))
        except:
            return False

    def copy(self) -> list:
        return list(iter(self))

class HashableDense(tuple):

    def __new__(cls, items: Iterable[Any], hash_:int = None) -> str:
        return tuple.__new__(HashableDense,items)

    def __init__(self, items:Iterable[Any], hash_:int = None) -> None:
        self._hash = hash_

    def __hash__(self) -> int:
        if not self._hash:
            self._hash = super().__hash__()
        return self._hash

Dense.register(HashableDense)
Dense.register(list)
Dense.register(tuple)
Dense.register(Dense_)

class Sparse(ABC):

    @abstractmethod
    def __getitem__(self, key) -> Any:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator:
        ...

    @abstractmethod
    def keys(self) -> abc.KeysView:
        ...

    @abstractmethod
    def items(self) -> Iterable:
        ...

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._row, attr)

    def __eq__(self, o: object) -> bool:
        try:
            return dict(self.items()) == dict(o.items())
        except:
            return False

    def copy(self) -> dict:
        return dict(self.items())

class Sparse_:
    __slots__=('_row')
    ##Instantiating classes which inherit from Sparse is moderately expensive due to the ABC checks.
    ##Therefore we keep Sparse around for public API checks but internally we use Sparse_ for inheritance.

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._row, attr)

    def __eq__(self, o: object) -> bool:
        try:
            return dict(self.items()) == dict(o.items())
        except:
            return False

    def copy(self) -> dict:
        return dict(self.items())

class HashableSparse(abc.Mapping):
    __slots__=('_item','_hash')
    def __init__(self,item:Sparse):
        self._item = item

    def __getitem__(self,key):
        return self._item[key]

    def __iter__(self):
        return iter(self._item)

    def __len__(self):
        return len(self._item)

    def __hash__(self) -> int:
        try:
            return self._hash
        except:
            self._hash = hash(frozenset(self._item.items()))
            return self._hash

    def __eq__(self, o: object) -> bool:
        try:
            return frozenset(o.items()) == frozenset(self._item.items())
        except:
            return False

    def __repr__(self) -> str:
        return repr(self._item)

    def __str__(self) -> str:
        return str(self._item)

    def copy(self):
        return self._item.copy()

Sparse.register(HashableSparse)
Sparse.register(abc.Mapping)
Sparse.register(Sparse_)

def is_materialized(item: Any) -> bool:
    return not isinstance(item,(Sparse,Dense)) or isinstance(item,(list,tuple,dict))

def extract_shape(given:Action, comparison:Action, is_comparison_list:bool=False):

    given_ndim = getattr(given,'ndim',-1)

    if given_ndim == -1:
        compare_action = given
        output_shape = None

    elif given_ndim == 0:
        compare_action = given.item()
        output_shape = ()

    else:
        expected_ndims = isinstance(comparison,(list,tuple)) + is_comparison_list

        if expected_ndims == 0:
            compare_action = given.item()
            output_shape = given.shape
        else:
            output_ndims   = given_ndim-expected_ndims

            if not is_comparison_list:
                compare_action = type(comparison)(given[(0,)*output_ndims].tolist())
            else:
                compare_action = given[(0,)*output_ndims].tolist()

            output_shape = (1,)*output_ndims

    return compare_action, output_shape

def create_shape(value:float, shape):

    if shape is None:
        return value
    else:
        try:
            t = torch # type: ignore
        except:
            t = globals()['torch'] = __import__('torch')

        #`t` could also be numpy
        return t.full(shape,value)

class L1Reward(Rewards):
    """A reward function using L1 distance."""
    __slots__ = ('_argmax',)

    def __init__(self, argmax: float) -> None:
        """Instantiate an L1Reward.

        Args:
            argmax: The location where reward is greatest.
        """
        self._argmax = argmax if not hasattr(argmax,'ndim') else argmax.item()

    def __call__(self, action: float) -> float:
        #due to broadcasting this automatically
        #handles shaping when action is Tensor
        return -abs(action-self._argmax)

    def __eq__(self, o: object) -> bool:
        return isinstance(o,L1Reward) and o._argmax == self._argmax

    def __getstate__(self):
        return self._argmax

    def __setstate__(self,args):
        self._argmax = args

    def __repr__(self) -> str:
        am = self._argmax
        return f"L1Reward({try_else(lambda:minimize(am),f'{am:.5f}')})"

class BinaryReward(Rewards):
    """A reward function with two values."""
    __slots__ = ('_argmax','_value')

    def __init__(self, argmax: Action, value:float=1.) -> None:
        """Instantiate BinaryReward.

        Args:
            argmax: The location where reward value 1 is returned.
                At all other actions a reward value of 0 is returned.
            value: The value returned at the argmax.
        """
        self._argmax = argmax if not hasattr(argmax,'ndim') else argmax.tolist()
        self._value  = value

    def __call__(self, action: Action) -> float:
        argmax = self._argmax
        comparable,shape = extract_shape(action,argmax)
        value = self._value if argmax==comparable else 0
        return create_shape(value,shape)

    def __eq__(self, o: object) -> bool:
        return o == self._argmax or \
            (isinstance(o,BinaryReward) and\
            o._argmax == self._argmax and\
            o._value == self._value)

    def __getstate__(self):
        return repr((self._argmax,) if self._value == 1 else (self._argmax,self._value))

    def __setstate__(self,args):
        args = literal_eval(args)
        self._argmax,self._value = (args[0],1) if len(args) == 1 else args

    def __repr__(self) -> str:
        am = self._argmax
        return f"BinaryReward({try_else(lambda:minimize(am),str(am))})"

class HammingReward(Rewards):
    """A reward function using Hamming distance."""
    __slots__ = ('_argmax',)

    def __init__(self, argmax: Sequence[Action]) -> None:
        """Instantiate a HammingReward.

        Args:
            argmax: The set of labels to calculate the Hamming distance from.
        """
        self._argmax = argmax if not hasattr(argmax,'ndim') else argmax.tolist()

    def __call__(self, action: Sequence[Action]) -> float:
        argmax = self._argmax
        comparable,shape = extract_shape(action,argmax[0],True)

        n_intersect = 0

        for a in comparable: n_intersect += a in self._argmax
        n_union = len(argmax) + len(comparable) - n_intersect

        value = n_intersect/n_union

        return create_shape(value,shape)

    def __getstate__(self):
        return repr(self._argmax)

    def __setstate__(self,args):
        self._argmax = literal_eval(args)

    def __repr__(self) -> str:
        am = self._argmax
        return f"HammingReward({try_else(lambda:minimize(am),str(am))})"

class DiscreteReward(Rewards):
    """A reward function mapping actions to rewards."""
    __slots__ = ('_state','_default')

    @overload
    def __init__(self, actions: Sequence[Action], rewards: Sequence[float], *, default: float = 0) -> None:
        ...

    @overload
    def __init__(self, mapping: Mapping[Action,float], *, default: float = 0) -> None:
        ...

    def __init__(self, *args, default=0) -> None:
        """Instantiate a DiscreteReward.

        Args:
            actions: The actions to define rewards for.
            rewards: The rewards for the given actions.
            mapping: A mapping of actions to rewards.
            default: The value to return for actions without mappings.
        """
        if len(args) == 2:
            actions,rewards = args
            self._state = args
            if len(actions) != len(rewards):
                raise CobaException("The given actions and rewards did not line up.")
        else:
            self._state = args[0]
        self._default = default

    @property
    def actions(self):
        return list(self._state.keys()) if isinstance(self._state,dict) else self._state[0]

    @property
    def rewards(self):
        return list(self._state.values()) if isinstance(self._state,dict) else self._state[1]

    def __call__(self, action: Action) -> float:
        if isinstance(self._state,dict):
            comp,shape = extract_shape(action,next(iter(self._state.keys())))
            value = self._state.get(comp,self._default)
            return create_shape(value,shape)
        else:
            actions,rewards = self._state
            comp,shape = extract_shape(action,actions[0])
            value = rewards[actions.index(comp)] if comp in actions else self._default
            return create_shape(value,shape)

    def __repr__(self) -> str:
        st = self._state
        return f"DiscreteReward({try_else(lambda:minimize(st),str(st))})"

    def __eq__(self, o: object) -> bool:

        return o == self.rewards or \
            (isinstance(o,DiscreteReward) and\
            o.actions == self.actions and\
            o.rewards == self.rewards and\
            o._default == self._default)

    def __getstate__(self):
        return repr((self._state,self._default))

    def __setstate__(self,args):
        self._state,self._default = literal_eval(args)
