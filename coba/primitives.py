from abc import ABC, abstractmethod
from numbers import Number
from operator import eq
from collections import abc
from typing import Union, Tuple, Sequence, Mapping, Iterable
from typing import TypeVar, Generic, Optional, Iterator, Any

Context = Union[None, str, Number, Sequence, Mapping]
Action  = Union[str, Number, Sequence, Mapping]
Actions = Union[None, Sequence[Action]]
Reward  = float

Prob   = float
Pmf    = Sequence[Prob]
kwargs = Mapping[str,Any]

Prediction = Union[
    Action,
    Tuple[Action,Prob],
    Tuple[Action     , kwargs],
    Tuple[Action,Prob, kwargs],
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
        return str(self.params)

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
    """Reward an action."""

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
    """Interact with an Environment."""
    __slots__=()

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

    def predict(self, context: 'Context', actions: 'Actions') -> 'Prediction':
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

    def learn(self, context: 'Context', action: 'Action', reward: float, probability: float, **kwargs) -> None:
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
    """Evaluate a Learner in an Environment."""

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

class HashableDense(abc.Sequence):
    __slots__=('_item','_hash')

    def __init__(self, item: Sequence) -> None:
        self._item = item

    def __getitem__(self,index):
        return self._item[index]

    def __iter__(self):
        return iter(self._item)

    def __len__(self):
        return len(self._item)

    def __eq__(self, o: object) -> bool:
        try:
            return len(self._item) == len(o) and all(map(eq, self._item, o))
        except:
            return False

    def __hash__(self) -> int:
        try:
            return self._hash
        except:
            self._hash = hash(tuple(self._item))
            return self._hash

    def __repr__(self) -> str:
        return repr(self._item)

    def __str__(self) -> str:
        return str(self._item)

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
