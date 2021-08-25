from coba.utilities import HashableDict
import collections

from abc import abstractmethod
from itertools import repeat, chain
from typing import Optional, Sequence, List, Callable, Hashable, Any, Union, Iterable, cast

from coba.random import CobaRandom

from coba.pipes import (
    Pipe, Source, Filter,
    CsvReader, ArffReader, LibSvmReader, ManikReader, 
    DiskSource, HttpSource, 
    ResponseToLines, Transpose
)

Action      = Union[Hashable, HashableDict]
Context     = Union[None, Hashable, HashableDict]
Feedback    = Any

class Interaction:
    """A class to contain all data needed to represent an interaction in a bandit simulation."""

    def __init__(self, context: Context, actions: Sequence[Action], feedbacks: Sequence[Feedback]) -> None:
        """Instantiate Interaction.

        Args
            context  : Features describing the interaction's context. Will be `None` for multi-armed bandit simulations.
            actions  : Features describing available actions in the interaction.
            feedbacks: Feedback that will be received based on the action that is taken in the interaction.
        """

        assert actions, "At least one action must be provided for each interaction."

        assert len(actions) == len(feedbacks), "The interaction should have a feedback for each action."

        self._context   =  context if not isinstance(context,dict) else HashableDict(context)
        self._actions   = [ action if not isinstance(action, dict) else HashableDict(action) for action in actions ]
        self._feedbacks = feedbacks

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

        return self._flatten(self._context)

    @property
    def actions(self) -> Sequence[Action]:
        """The interaction's available actions."""

        return [ self._flatten(action) for action in self._actions ]

    @property
    def feedbacks(self) -> Sequence[Feedback]:
        """The interaction's feedback associated with each action."""
        return list(self._feedbacks)

class Simulation(Source[Iterable[Interaction]]):
    """The simulation interface."""

    @abstractmethod
    def read(self) -> Iterable[Interaction]:
        """The sequence of interactions in a simulation.

        Remarks:
            This function should always be "re-iterable".
        """
        ...

class MemorySimulation(Simulation):
    """A Simulation implementation created from in memory sequences of contexts, actions and rewards."""

    def __init__(self, interactions: Sequence[Interaction]) -> None:
        """Instantiate a MemorySimulation.

        Args:
            interactions: The sequence of interactions in this simulation.
        """

        self._interactions = interactions

    def read(self) -> Iterable[Interaction]:
        """Read the interactions in this simulation."""
        
        return self._interactions

class ClassificationSimulation(Simulation):
    """A simulation created from classification dataset with features and labels.

    ClassificationSimulation turns labeled observations from a classification data set
    into interactions. For each interaction the feature set becomes the context and 
    all possible labels become the actions. Rewards for each interaction are created by 
    assigning a reward of 1 for taking the correct action (i.e., choosing the correct
    label)) and a reward of 0 for taking any other action (i.e., choosing any of the
    incorrect lables).

    Remark:
        This class when created from a data set will load all data into memory. Be careful when 
        doing this if you are working with a large dataset. To reduce memory usage you can provide
        meta information upfront that will allow features to be correctly encoded while the
        dataset is being streamed instead of waiting until the end of the data to train an encoder.
    """

    def __init__(self, features: Sequence[Any], labels: Union[Sequence[Action], Sequence[Sequence[Action]]] ) -> None:
        """Instantiate a ClassificationSimulation.

        Args:
            features: The collection of features used for the original classifier problem.
            labels: The collection of labels assigned to each observation of features.
        """

        assert len(features) == len(labels), "Mismatched lengths of features and labels"

        if isinstance(labels[0], collections.Sequence) and not isinstance(labels[0],str):
            labels_flat = list(chain.from_iterable(labels)) #type: ignore
        else:
            labels_flat = labels #type: ignore

        feedback      = lambda action,label: int(is_label(action,label) or in_multilabel(action,label)) #type: ignore
        is_label      = lambda action,label: action == label #type: ignore
        in_multilabel = lambda action,label: isinstance(label,collections.Sequence) and action in label #type: ignore

        contexts  = features 
        actions   = list(sorted(set(labels_flat), key=lambda l: labels_flat.index(l)))
        feedbacks = [ [ feedback(action,label) for action in actions ] for label in labels ]

        self._interactions = list(map(Interaction, contexts, repeat(actions), feedbacks))

    def read(self) -> Iterable[Interaction]:
        """Read the interactions in this simulation."""
        
        return self._interactions

class LambdaSimulation(Simulation):
    """A Simulation created from lambda functions that generate contexts, actions and rewards.

    Remarks:
        This implementation is useful for creating simulations from defined distributions.
    """

    def __init__(self,
        n_interactions: int,
        context       : Callable[[int               ],Context         ],
        actions       : Callable[[int,Context       ],Sequence[Action]], 
        reward        : Callable[[int,Context,Action],float           ]) -> None:
        """Instantiate a LambdaSimulation.

        Args:
            n_interactions: How many interactions the LambdaSimulation should have.
            context: A function that should return a context given an index in `range(n_interactions)`.
            actions: A function that should return all valid actions for a given index and context.
            reward: A function that should return the reward for the index, context and action.
        """

        self._interactions: List[Interaction] = []

        for i in range(n_interactions):
            _context  = context(i)
            _actions  = actions(i, _context)
            _rewards  = [ reward(i, _context, _action) for _action in _actions]

            self._interactions.append(Interaction(_context, _actions, _rewards))

    def read(self) -> Iterable[Interaction]:
        """Read the interactions in this simulation."""
        
        return self._interactions

    def __repr__(self) -> str:
        return '"LambdaSimulation"'

class ReaderSimulation(Simulation):

    def __init__(self, 
        reader      : Filter[Iterable[str], Any], 
        source      : Union[str,Source[Iterable[str]]], 
        label_column: Union[str,int], 
        with_header : bool=True) -> None:
        
        self._reader = reader

        if isinstance(source, str) and source.startswith('http'):
            self._source = Pipe.join(HttpSource(source), [ResponseToLines()])
        elif isinstance(source, str):
            self._source = DiskSource(source)
        else:
            self._source = source
        
        self._label_column = label_column
        self._with_header  = with_header
        self._interactions = cast(Optional[Sequence[Interaction]], None)

    def read(self) -> Iterable[Interaction]:
        """Read the interactions in this simulation."""
        return self._load_interactions()

    def _load_interactions(self) -> Sequence[Interaction]:
        parsed_rows_iter = iter(self._reader.filter(self._source.read()))

        if self._with_header:
            header = next(parsed_rows_iter)
        else:
            header = []

        if isinstance(self._label_column, str):
            label_col_index = header.index(self._label_column)
        else:
            label_col_index = self._label_column

        parsed_cols = list(Transpose().filter(parsed_rows_iter))
        
        label_col    = parsed_cols.pop(label_col_index)
        feature_rows = list(Transpose().filter(parsed_cols))

        is_sparse_labels = len(label_col) == 2 and isinstance(label_col[0],tuple) and isinstance(label_col[1],tuple)
        
        if is_sparse_labels:
            dense_labels: List[Any] = ['0']*len(feature_rows)
            
            for label_row, label_val in zip(*label_col): #type:ignore
                dense_labels[label_row] = label_val

        else:
            dense_labels = list(label_col)

        return ClassificationSimulation(feature_rows, dense_labels).read()

    def __repr__(self) -> str:
        return str(self._source)

class CsvSimulation(ReaderSimulation):
    def __init__(self, source:Union[str,Source[Iterable[str]]], label_column:Union[str,int], with_header:bool=True) -> None:
        super().__init__(CsvReader(), source, label_column, with_header)

    def __repr__(self) -> str:
        return f'{{"CsvSimulation":"{super().__repr__()}"}}'

class ArffSimulation(ReaderSimulation):
    def __init__(self, source:Union[str,Source[Iterable[str]]], label_column:Union[str,int]) -> None:
        super().__init__(ArffReader(skip_encoding=[label_column]), source, label_column)

    def __repr__(self) -> str:
        return f'{{"ArffSimulation":"{super().__repr__()}"}}'    

class LibsvmSimulation(ReaderSimulation):
    def __init__(self, source:Union[str,Source[Iterable[str]]]) -> None:
        super().__init__(LibSvmReader(), source, 0, False)

    def __repr__(self) -> str:
        return f'{{"LibsvmSimulation":"{super().__repr__()}"}}'

class ManikSimulation(ReaderSimulation):
    def __init__(self, source:Union[str,Source[Iterable[str]]]) -> None:
        super().__init__(ManikReader(), source, 0, False)

    def __repr__(self) -> str:
        return f'{{"ManikSimulation":"{super().__repr__()}"}}'

class ValidationSimulation(LambdaSimulation):
    def __init__(self, n_interactions: int=500, n_actions: int=10, n_features: int=10, context_features:bool = True, action_features:bool = True, sparse: bool=False, seed:int=1) -> None:

        self._n_bandits        = n_actions
        self._n_features       = n_features
        self._context_features = context_features
        self._action_features  = action_features
        self._seed             = seed

        r = CobaRandom(seed)

        context: Callable[[int               ], Context         ]
        actions: Callable[[int,Context       ], Sequence[Action]]
        rewards: Callable[[int,Context,Action], float           ]

        sparsify  = lambda x: (tuple(range(len(x))), tuple(x)) if sparse else tuple(x)
        unsparse  = lambda x: x[1] if sparse else x
        normalize = lambda X: [x/sum(X) for x in X]

        if not context_features and not action_features:

            means = [ m/n_actions + 1/(2*n_actions) for m in r.randoms(n_actions) ]

            actions_features = []
            for i in range(n_actions):
                action = [0] * n_actions
                action[i] = 1
                actions_features.append(tuple(action))

            context = lambda i     : None
            actions = lambda i,c   : sparsify(actions_features)
            rewards = lambda i,c,a : means[unsparse(a).index(1)] + (r.random()-.5)/n_actions

        if context_features and not action_features:
            #normalizing allows us to make sure our reward is in [0,1]
            bandit_thetas = [ r.randoms(n_features) for _ in range(n_actions) ]
            theta_totals  = [ sum(theta) for theta in bandit_thetas]
            bandit_thetas = [ [t/norm for t in theta ] for theta,norm in zip(bandit_thetas,theta_totals)]

            actions_features = []
            for i in range(n_actions):
                action = [0] * n_actions
                action[i] = 1
                actions_features.append(tuple(action))

            context = lambda i     : sparsify(r.randoms(n_features))
            actions = lambda i,c   : [sparsify(af) for af in actions_features]
            rewards = lambda i,c,a : sum([cc*t for cc,t in zip(unsparse(c),bandit_thetas[unsparse(a).index(1)])])

        if not context_features and action_features:

            theta = r.randoms(n_features)

            context = lambda i     :   None
            actions = lambda i,c   : [ sparsify(normalize(r.randoms(n_features))) for _ in range(r.randint(2,10)) ]
            rewards = lambda i,c,a : float(sum([cc*t for cc,t in zip(theta,unsparse(a))]))

        if context_features and action_features:

            context = lambda i     :   sparsify(r.randoms(n_features))
            actions = lambda i,c   : [ sparsify(normalize(r.randoms(n_features))) for _ in range(r.randint(2,10)) ]
            rewards = lambda i,c,a : sum([cc*t for cc,t in zip(unsparse(c),unsparse(a))])

        super().__init__(n_interactions, context, actions, rewards)

    def read(self) -> Iterable[Interaction]:
        for i in super().read():
            yield i
            #yield Interaction(i.context, i.actions, [ int(r == max(i.feedbacks)) for r in i.feedbacks ] )


    def __repr__(self) -> str:
        return f"Validation"

class RegressionSimulation(Simulation):
    """A simulation created from regression dataset with features and labels.
    RegressionSimulation turns labeled observations from a regression data set
    into interactions. For each interaction the feature set becomes the context and 
    all possible labels become the actions. Rewards for each interaction are created by 
    assigning a minus absolute error. Rewards are close to zero for taking actions that are 
    closer to the correct action (label) and lower ones for being far from the correct action.
    Remark:
        This class when created from a data set will load all data into memory. Be careful when 
        doing this if you are working with a large dataset. To reduce memory usage you can provide
        meta information upfront that will allow features to be correctly encoded while the
        dataset is being streamed instead of waiting until the end of the data to train an encoder.
    """

    def __init__(self, features: Sequence[Any], labels: Union[Sequence[Action], Sequence[Sequence[Action]]] ) -> None:
        """Instantiate a RegressionSimulation.
        Args:
            features: The collection of features used for the original regression problem.
            labels: The collection of labels assigned to each observation of features.
        """

        assert len(features) == len(labels), "Mismatched lengths of features and labels"

        if isinstance(labels[0], collections.Sequence) and not isinstance(labels[0],str):
            labels_flat = list(chain.from_iterable(labels)) #type: ignore
        else:
            labels_flat = labels #type: ignore

        feedback  = lambda action,label: -abs(float(action)-float(label))

        contexts  = features 
        actions   = list(sorted(set(labels_flat), key=lambda l: labels_flat.index(l)))
        feedbacks = [ [ feedback(action,label) for action in actions ] for label in labels ]

        self._interactions = list(map(Interaction, contexts, repeat(actions), feedbacks))

    def read(self) -> Iterable[Interaction]:
        """Read the interactions in this simulation."""
        
        return self._interactions

    def __repr__(self) -> str:
        return '"Regression Simulation"'