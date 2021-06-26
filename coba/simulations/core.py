from abc import ABC, abstractmethod
import collections

from itertools import accumulate, repeat, chain
from typing import Optional, Sequence, List, Callable, Hashable, Tuple, Dict, Any, Union, Iterable

from coba.random import CobaRandom

from coba.pipes import (
    Pipe, Source, Filter,
    CsvReader, ArffReader, LibSvmReader, ManikReader, 
    DiskSource, HttpSource, 
    ResponseToLines, Transpose, Flatten
)

Action      = Union[Tuple[Hashable,...], Hashable, dict]
Key         = int
Context     = Optional[Union[Tuple[Hashable,...], Hashable, dict]]

class Interaction:
    """A class to contain all data needed to represent an interaction in a bandit simulation."""

    def __init__(self, key: Key, context: Context, actions: Sequence[Action]) -> None:
        """Instantiate Interaction.

        Args
            context: Features describing the interaction's context. Will be `None` for multi-armed bandit simulations.
            actions: Features describing available actions in the interaction.
            key    : A unique key assigned to this interaction.
        """

        assert actions, "At least one action must be provided to interact"

        self._key     = key
        self._context = context
        self._actions = actions

    @property
    def key(self) -> Key:
        """A unique key identifying the interaction."""
        return self._key

    @property
    def context(self) -> Optional[Context]:
        """The interaction's context description."""

        #context is non-existant or singular so return it as is
        if self._context is None or not isinstance(self._context, collections.Sequence):
            return self._context

        #The context appears to be a sparse representation. Return it as a dictionary. This may be an incorrect assumption.
        #In the future we should probably improve the back end so we can explicit indicate if our context is sparse rather
        #than trying to infer it based on the structure of the context.
        if len(self._context) == 2 and isinstance(self._context[0],tuple) and isinstance(self._context[1],tuple):
            return dict(zip(self._context[0], self._context[1]))

        #context is a standard feature vector so return it as is
        return self._context

    @property
    def actions(self) -> Sequence[Action]:
        """The interaction's available actions."""

        actions = []

        for action in self._actions:
            #context is non-existant or singular so return it as is
            if not isinstance(action, collections.Sequence):
                actions.append(action)

            #The context appears to be a sparse representation. Return it as a dictionary. This may be an incorrect assumption.
            #In the future we should probably improve the back end so we can explicit indicate if our context is sparse rather
            #than trying to infer it based on the structure of the context.
            elif len(action) == 2 and isinstance(action[0],tuple) and isinstance(action[1],tuple):
                actions.append(dict(zip(action[0], action[1])))

            elif isinstance(action, str):
                actions.append(action)
            
            else:
                actions.append(action)

        return actions

class Reward(ABC):

    @abstractmethod
    def observe(self, choices: Sequence[Tuple[Key,Context,Action]] ) -> Sequence[float]:
        ...

class Simulation(ABC):
    """The simulation interface."""

    @property
    @abstractmethod
    def interactions(self) -> Sequence[Interaction]:
        """The sequence of interactions in a simulation.

        Remarks:
            Interactions should always be re-iterable. So long as interactions is a Sequence 
            this will always be the case. If interactions is changed to Iterable in the future
            then it will be possible for it to only allow enumeration one time and care will need
            to be taken.
        """
        ...

    @property
    @abstractmethod
    def reward(self) -> Reward:
        """The reward object which can observe rewards for pairs of actions and interaction keys."""
        ...    

class MemoryReward(Reward):
    def __init__(self, rewards: Sequence[Tuple[Key,Action,float]] = []) -> None:
        self._rewards: Dict[Tuple[Key,Action], float] = { (r[0],self._key(r[1])):r[2] for r in rewards }

    def observe(self, choices: Sequence[Tuple[Key,Context,Action]] ) -> Sequence[float]:
        return [ self._rewards[(key,self._key(action))] for key,_,action in choices ]

    def _key(self, action):

        if isinstance(action,str) or not isinstance(action,(collections.Sequence,dict)) :
            return action

        if len(action) == 2 and isinstance(action[0], tuple) and isinstance(action[1],tuple):
            return tuple(zip(action[0], action[1]))

        if isinstance(action,dict):
            return tuple(action.items())

        return action

class ClassificationReward(Reward):
    def __init__(self, labels: Sequence[Tuple[Key,Union[Action, Sequence[Action]]]] = []) -> None:
        self._labels = dict(labels)

    def add(self, key: Key, action: Action):
        self._labels[key] = action

    def observe(self, choices: Sequence[Tuple[Key,Context,Action]] ) -> Sequence[float]:
        rewards = []

        for key, _, action in choices:
            key_label = self._labels[key]
            reward    = int(action in key_label if isinstance(key_label, collections.Sequence) and not isinstance(key_label, str) else action == key_label)
            rewards.append(reward)

        return rewards

class MemorySimulation(Simulation):
    """A Simulation implementation created from in memory sequences of contexts, actions and rewards."""

    def __init__(self, interactions: Sequence[Interaction], reward: Reward) -> None:
        """Instantiate a MemorySimulation.

        Args:
            interactions: The sequence of interactions in this simulation.
            reward: The reward object to observe in this simulation.
        """

        self._interactions = interactions
        self._reward       = reward

    @property
    def interactions(self) -> Sequence[Interaction]:
        """The interactions in this simulation.

        Remarks:
            See the Simulation base class for more information.
        """
        return self._interactions

    @property
    def reward(self) -> Reward:
        """The reward object which can observe rewards for pairs of actions and interaction keys."""
        return self._reward

class ClassificationSimulation(MemorySimulation):
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
            
        action_set   = list(sorted(set(labels_flat), key=lambda l: labels_flat.index(l) ))
        interactions = [ Interaction(i, context, action_set) for i, context in enumerate(features) ] #type: ignore
        reward       = ClassificationReward(list(enumerate(labels)))

        super().__init__(interactions, reward) #type:ignore

class BatchedSimulation(MemorySimulation):
    """A simulation whose interactions have been batched."""

    def __init__(self, simulation: Simulation, batch_sizes: Sequence[int]) -> None:
        self._simulation = simulation

        #remove Nones and 0s
        batch_sizes  = list(filter(None, batch_sizes))
        batch_slices = list(accumulate([0] + list(batch_sizes)))

        self._batches = [ simulation.interactions[batch_slices[i]:batch_slices[i+1]] for i in range(len(batch_slices)-1) ]

        super().__init__(simulation.interactions[0:sum(batch_sizes)], simulation.reward)

    @property
    def interaction_batches(self) -> Sequence[Sequence[Interaction]]:
        """The sequence of batches of interactions in a simulation."""
        return self._batches

class LambdaSimulation(Source[Simulation]):
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

        interaction_tuples: List[Tuple[Key, Context, Sequence[Action]]] = []
        reward_tuples     : List[Tuple[Key, Action , float           ]] = []

        for i in range(n_interactions):
            _context  = context(i)
            _actions  = actions(i, _context)
            _rewards  = [ reward(i, _context, _action) for _action in _actions]

            interaction_tuples.append( (i, _context, _actions) )
            reward_tuples.extend(zip(repeat(i), _actions, _rewards))

        self._interactions = [ Interaction(key, context, actions) for key,context,actions in interaction_tuples ]
        self._reward       = MemoryReward(reward_tuples)
        self._simulation   = MemorySimulation(self._interactions, self._reward)

    def read(self) -> Simulation:
        return self._simulation

    def __repr__(self) -> str:
        return '"LambdaSimulation"'

class ReaderSimulation(Source[Simulation]):

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

    def read(self) -> Simulation:
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
        feature_rows = list(Transpose().filter(Flatten().filter(parsed_cols)))

        is_sparse_labels = len(label_col) == 2 and isinstance(label_col[0],tuple) and isinstance(label_col[1],tuple)
        
        if is_sparse_labels:
            dense_labels: List[Any] = ['0']*len(feature_rows)
            
            for label_row, label_val in zip(*label_col): #type:ignore
                dense_labels[label_row] = label_val

        else:
            dense_labels = list(label_col)

        return ClassificationSimulation(feature_rows, dense_labels)

    def __repr__(self) -> str:
        return str(self._source)

class CsvSimulation(Source[Simulation]):
    def __init__(self, source:Union[str,Source[Iterable[str]]], label_column:Union[str,int], with_header:bool=True) -> None:
        self._simulation_source = ReaderSimulation(CsvReader(), source, label_column, with_header)

    def read(self) -> Simulation:
        return self._simulation_source.read()

    def __repr__(self) -> str:
        return f'{{"CsvSimulation":"{self._simulation_source}"}}'

class ArffSimulation(Source[Simulation]):
    def __init__(self, source:Union[str,Source[Iterable[str]]], label_column:Union[str,int]) -> None:
        self._simulation_source = ReaderSimulation(ArffReader(skip_encoding=[label_column]), source, label_column)

    def read(self) -> Simulation:
        return self._simulation_source.read()

    def __repr__(self) -> str:
        return f'{{"ArffSimulation":"{self._simulation_source}"}}'    

class LibsvmSimulation(Source[Simulation]):
    def __init__(self, source:Union[str,Source[Iterable[str]]]) -> None:
        self._simulation_source = ReaderSimulation(LibSvmReader(), source, 0, False)

    def read(self) -> Simulation:
        return self._simulation_source.read()

    def __repr__(self) -> str:
        return f'{{"LibsvmSimulation":"{self._simulation_source}"}}'

class ManikSimulation(Source[Simulation]):
    def __init__(self, source:Union[str,Source[Iterable[str]]]) -> None:
        self._simulation_source = ReaderSimulation(ManikReader(), source, 0, False)

    def read(self) -> Simulation:
        return self._simulation_source.read()

    def __repr__(self) -> str:
        return f'{{"ManikSimulation":"{self._simulation_source}"}}'

class ValidationSimulation(LambdaSimulation):
    def __init__(self, n_interactions: int=500, n_actions: int=10, n_features: int=10, context_features:bool = True, action_features:bool = True, sparse: bool=False, seed:int=1000) -> None:

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
            rewards = lambda i,c,a : sum([cc*t for cc,t in zip(unsparse(c),unsparse(a))])/sum(unsparse(a))

        super().__init__(n_interactions, context, actions, rewards)

    def __repr__(self) -> str:
        return f"ValidationSimulation(cf={self._context_features},af={self._action_features},seed={self._seed})"