import collections

from itertools import chain, repeat
from numbers import Number
from typing import Sequence, Dict, Any, Iterable, Union, List, Callable, cast, Optional

from coba.pipes import Source, Pipe, Filter, HttpIO, DiskIO, ResponseToLines, Transpose, CsvReader, ArffReader, LibSvmReader, ManikReader
from coba.random import CobaRandom

from coba.environments.core import Context, Action, Simulation, SimulatedInteraction

class MemorySimulation(Simulation):
    """A Simulation implementation created from in memory sequences of contexts, actions and rewards."""

    def __init__(self, interactions: Sequence[SimulatedInteraction]) -> None:
        """Instantiate a MemorySimulation.

        Args:
            interactions: The sequence of interactions in this simulation.
        """

        self._interactions = interactions

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the simulation."""
        return {}

    def read(self) -> Iterable[SimulatedInteraction]:
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
    """

    def __init__(self, features: Sequence[Any], labels: Union[Sequence[Action], Sequence[List[Action]]] ) -> None:
        """Instantiate a ClassificationSimulation.

        Args:
            features: The collection of features used for the original classifier problem.
            labels: The collection of labels assigned to each observation of features.
        """

        assert len(features) == len(labels), "Mismatched lengths of features and labels"

        #how can we tell the difference between featurized labels and multilabels????
        #for now we will assume multilables will be passed in as arrays not tuples...
        if not isinstance(labels[0], collections.Hashable):
            labels_flat = list(chain.from_iterable(labels))
        else:
            labels_flat = list(labels)

        reveal        = lambda action,label: int(is_label(action,label) or in_multilabel(action,label)) #type: ignore
        is_label      = lambda action,label: action == label #type: ignore
        in_multilabel = lambda action,label: isinstance(label,collections.Sequence) and action in label #type: ignore

        # shuffling so that action order contains no statistical information
        # sorting so that the shuffled values are always shuffled in the same order
        actions  = CobaRandom(1).shuffle(sorted(set(labels_flat))) 
        contexts = features
        rewards  = [ [ reveal(action,label) for action in actions ] for label in labels ]

        self._interactions = [ SimulatedInteraction(c,a,rewards=r) for c,a,r in zip(contexts, repeat(actions), rewards) ]

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the simulation."""
        return { }

    def read(self) -> Iterable[SimulatedInteraction]:
        """Read the interactions in this simulation."""

        return self._interactions

class RegressionSimulation(Simulation):
    """A simulation created from regression dataset with features and labels.
    
    RegressionSimulation turns labeled observations from a regression data set
    into interactions. For each interaction the feature set becomes the context and 
    all possible labels become the actions. Rewards for each interaction are created by 
    assigning a minus absolute error. Rewards are close to zero for taking actions that are 
    closer to the correct action (label) and lower ones for being far from the correct action.
    
    """

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the simulation."""
        return {}

    def __init__(self, features: Sequence[Any], labels: Sequence[Number] ) -> None:
        """Instantiate a RegressionSimulation.
    
        Args:
            features: The collection of features used for the original regression problem.
            labels: The collection of labels assigned to each observation of features.
        """

        assert len(features) == len(labels), "Mismatched lengths of features and labels"

        reward   = lambda action,label: 1-abs(float(action)-float(label))
        contexts = features
        actions  = CobaRandom(1).shuffle(sorted(set(labels)))
        rewards  = [ [ reward(action,label) for action in actions ] for label in labels ]

        self._interactions = [ SimulatedInteraction(c,a,rewards=r) for c,a,r in zip(contexts, repeat(actions), rewards) ]

    def read(self) -> Iterable[SimulatedInteraction]:
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

        self._interactions: List[SimulatedInteraction] = []

        for i in range(n_interactions):
            _context  = context(i)
            _actions  = actions(i, _context)
            _rewards  = [ reward(i, _context, _action) for _action in _actions]

            self._interactions.append(SimulatedInteraction(_context, _actions, rewards=_rewards))

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the simulation."""
        return {}

    def read(self) -> Iterable[SimulatedInteraction]:
        """Read the interactions in this simulation."""
        
        return self._interactions

class ReaderSimulation(Simulation):

    def __init__(self, 
        reader      : Filter[Iterable[str], Any], 
        source      : Union[str,Source[Iterable[str]]], 
        label_column: Union[str,int], 
        with_header : bool=True) -> None:
        
        self._reader = reader

        if isinstance(source, str) and source.startswith('http'):
            self._source = Pipe.join(HttpIO(source), [ResponseToLines()])
        elif isinstance(source, str):
            self._source = DiskIO(source)
        else:
            self._source = source
        
        self._label_column = label_column
        self._with_header  = with_header
        self._interactions = cast(Optional[Sequence[SimulatedInteraction]], None)

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the simulation."""
        return {"source": str(self._source) }

    def read(self) -> Iterable[SimulatedInteraction]:
        """Read the interactions in this simulation."""
        return self._load_interactions()

    def _load_interactions(self) -> Sequence[SimulatedInteraction]:
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

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the simulation."""
        return { "csv": super().params["source"] }

class ArffSimulation(ReaderSimulation):
    def __init__(self, source:Union[str,Source[Iterable[str]]], label_column:Union[str,int]) -> None:
        super().__init__(ArffReader(skip_encoding=[label_column]), source, label_column)

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the simulation."""
        return { "arff": super().params["source"] }

class LibsvmSimulation(ReaderSimulation):
    def __init__(self, source:Union[str,Source[Iterable[str]]]) -> None:
        super().__init__(LibSvmReader(), source, 0, False)

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the simulation."""
        return { "libsvm": super().params["source"] }

class ManikSimulation(ReaderSimulation):
    def __init__(self, source:Union[str,Source[Iterable[str]]]) -> None:
        super().__init__(ManikReader(), source, 0, False)

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the simulation."""
        return { "manik": super().params["source"] }

class ValidationSimulation(LambdaSimulation):
    
    def __init__(self, n_interactions: int=500, n_actions: int=10, n_features: int=10, context_features:bool = True, action_features:bool = True, sparse: bool=False, seed:int=1, make_binary=False) -> None:

        self._n_bandits        = n_actions
        self._n_features       = n_features
        self._context_features = context_features
        self._action_features  = action_features
        self._seed             = seed
        self._make_binary      = make_binary

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
            scales        = [ 1 for _ in range(n_features) ]

            actions_features = []
            for i in range(n_actions):
                action = [0] * n_actions
                action[i] = 1
                actions_features.append(tuple(action))

            context = lambda i     : sparsify([ f*s for f,s in zip(r.randoms(n_features),scales) ])
            actions = lambda i,c   : [sparsify(af) for af in actions_features]
            rewards = lambda i,c,a : sum([cc*t for cc,t in zip(unsparse(c),bandit_thetas[unsparse(a).index(1)])])

        if not context_features and action_features:

            theta = r.randoms(n_features)

            context = lambda i     :   None
            actions = lambda i,c   : [ sparsify(normalize(r.randoms(n_features))) for _ in range(n_actions) ]
            rewards = lambda i,c,a : float(sum([cc*t for cc,t in zip(theta,unsparse(a))]))

        if context_features and action_features:

            context = lambda i     :   sparsify(r.randoms(n_features))
            actions = lambda i,c   : [ sparsify(normalize(r.randoms(n_features))) for _ in range(n_actions) ]
            rewards = lambda i,c,a : sum([cc*t for cc,t in zip(unsparse(c),unsparse(a))])

        super().__init__(n_interactions, context, actions, rewards)

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the simulation."""
        return { }

    def read(self) -> Iterable[SimulatedInteraction]:
        for i in super().read():
            if self._make_binary:
                yield SimulatedInteraction(i.context, i.actions, rewards = [ int(r == max(i.reveals)) for r in i.reveals ] )
            else:
                yield i

    def __repr__(self) -> str:
        return f"ValidationSimulation(cf={self._context_features},af={self._action_features},seed={self._seed})"