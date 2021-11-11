import collections

from itertools import chain, repeat
from typing import Sequence, Dict, Any, Iterable, Union, List, Callable, cast, Optional

from coba.pipes import Source, Pipe, Filter, HttpIO, DiskIO, ResponseToLines, CsvReader, ArffReader, LibSvmReader, ManikReader, Structures
from coba.random import CobaRandom
from coba.encodings import InteractionTermsEncoder

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

    def __init__(self, examples: Union[Iterable[Sequence[Any]], Iterable[Dict[Any,Any]]]) -> None:
        """Instantiate a ClassificationSimulation.

        Args:
            examples: The labeled regression examples which will be used to create the contextual bandit simulation.
            label_key: The key that identifies the label in each example.
        """

        features,labels = zip(*examples)

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

    def __init__(self,
        examples: Union[Iterable[Sequence[Any]], Iterable[Dict[Any,Any]]], 
        label_key: Union[int,str]) -> None:
        """Instantiate a RegressionSimulation.
    
        Args:
            examples: The labeled regression examples which will be used to create the contextual bandit simulation.
            label_key: The key that identifies the label in each example.
        """

        examples = list(examples)
        labels = [ example.pop(label_key) for example in examples ]

        reward   = lambda action,label: 1-abs(float(action)-float(label))
        contexts = examples
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
        reader   : Filter[Iterable[str], Any], 
        source   : Union[str,Source[Iterable[str]]], 
        label_col: Union[str,int]) -> None:
        
        self._reader = reader

        if isinstance(source, str) and source.startswith('http'):
            self._source = Pipe.join(HttpIO(source), [ResponseToLines()])
        elif isinstance(source, str):
            self._source = DiskIO(source)
        else:
            self._source = source

        self._label_column = label_col
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
        structured_rows = Structures([None, self._label_column]).filter(parsed_rows_iter)

        return ClassificationSimulation(structured_rows).read()

    def __repr__(self) -> str:
        return str(self._source)

class CsvSimulation(ReaderSimulation):
    def __init__(self, source:Union[str,Source[Iterable[str]]], label_column:Union[str,int], with_header:bool=True) -> None:
        super().__init__(CsvReader(with_header), source, label_column)

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
        super().__init__(LibSvmReader(), source, 0)

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
    
    def __init__(self, n_interactions: int=500, n_actions: int=10, n_context_feats:int = 10, n_action_feats:int = 10, seed:int=1) -> None:

        self._n_actions          = n_actions
        self._n_context_features = n_action_feats
        self._n_action_features  = n_context_feats
        self._seed               = seed

        rng = CobaRandom(seed)

        feature_count = max(1,n_context_feats) * max(1,n_action_feats)
        normalize     = lambda X: [ x/sum(X) for x in X]

        if n_action_feats == 0:
            weights = [ normalize(rng.randoms(feature_count)) for _ in range(n_actions) ]
        else:
            weights = normalize(rng.randoms(feature_count))

        def actions(index:int, context: Context) -> Sequence[Action]:
            return [ rng.randoms(n_action_feats) for _ in range(n_actions)] if n_action_feats else [ str(a) for a in range(n_actions) ]

        def context(index:int) -> Context:
            return None if n_context_feats == 0 else rng.randoms(n_context_feats)

        def reward(index:int, context:Context, action:Action) -> float:
            
            W = weights[int(action)] if isinstance(action,str) else weights
            X = [1]                  if context is None        else context
            A = [1]                  if isinstance(action,str) else action
            F = InteractionTermsEncoder(["xa"]).encode(x=X,a=A)

            r = sum([w*f for w,f in zip(W,F)])
            e = (rng.random()-1/2)*1/8

            return min(1,max(0,r+e))

        super().__init__(n_interactions, context, actions, reward)

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the simulation."""
        return { }

    def __repr__(self) -> str:
        return f"ValidationSimulation(na={self._n_actions},cf={self._n_context_features},af={self._n_action_features},seed={self._seed})"