"""The simulations module contains core classes and types for defining contextual bandit simulations.

This module contains the abstract interface expected for bandit simulations along with the 
class defining an Interaction within a bandit simulation. Additionally, this module also contains 
the type hints for Context, Action and Reward. These type hints don't contain any functionality. 
Rather, they simply make it possible to use static type checking for any project that desires 
to do so.

TODO Add RegressionSimulation
"""

import json

from itertools import accumulate
from abc import ABC, abstractmethod
from typing import (
    Optional, Sequence, List, Callable, 
    Hashable, Any, Tuple, overload, cast,
    Dict
)

import coba.random

from coba.tools import PackageChecker, CobaConfig
from coba.data.sources import Source, HttpSource
from coba.data.encoders import OneHotEncoder
from coba.data.filters import Filter

Context = Optional[Hashable]
Action  = Hashable
Key     = int

class Interaction:
    """A class to contain all data needed to represent an interaction in a bandit simulation."""

    #this is a problem with pylance compaining about covariance in constructor so we have to type ignore it. 
    #See this ticket in mypy for more info https://github.com/python/mypy/issues/2850
    def __init__(self, context: Context, actions: Sequence[Action], key: Key = 0) -> None: #type: ignore
        """Instantiate Interaction.

        Args
            context: Features describing the interactions's context. Will be `None` for multi-armed bandit simulations.
            actions: Features describing available actions in the interaction.
            key    : A unique key assigned to this interaction.
        """

        assert actions, "At least one action must be provided to interact"

        self._context = context
        self._actions = actions
        self._key     = key

    @property
    def context(self) -> Context:
        """The interaction's context description."""
        return self._context

    @property
    def actions(self) -> Sequence[Action]:
        """The interactions's available actions."""
        return self._actions
    
    @property
    def key(self) -> Key:
        """A unique key identifying the interaction."""
        return self._key

class Reward(ABC):

    @abstractmethod
    def observe(self, choices: Sequence[Tuple[Key,Action]] ) -> Sequence[float]:
        ...

class MemoryReward(Reward):
    def __init__(self, rewards: Sequence[Tuple[Key,Action,float]] = []) -> None:
        self._rewards: Dict[Tuple[Key,Action], float] = {}

        for reward in rewards:
            self._rewards[(reward[0],reward[1])] = reward[2]

    def add_observation(self, observation: Tuple[Key,Action,float]) -> None:
        choice = (observation[0],observation[1])
        reward = observation[2]

        if choice in self._rewards: raise Exception("Unable to add an existing observation.")
        
        self._rewards[choice] = reward

    def observe(self, choices: Sequence[Tuple[Key,Action]] ) -> Sequence[float]:
        return [ self._rewards[choice] for choice in choices ]

class ClassificationReward(Reward):

    def __init__(self, labels: Sequence[Tuple[Key,Action]]) -> None:
        self._labels = dict(labels)

    def observe(self, choices: Sequence[Tuple[Key,Action]] ) -> Sequence[float]:
        return [ float(self._labels[choice[0]] == choice[1]) for choice in choices ]

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

class OpenmlSource(Source[Tuple[Sequence[Context], Sequence[Action]]]):

    def __init__(self, id:int, md5_checksum:str = None):
        self._data_id      = id
        self._md5_checksum = md5_checksum

    def read(self) -> Tuple[Sequence[Sequence[Any]], Sequence[Any]]:
        
        #placing some of these at the top would cause circular references
        from coba.data.pipes    import Pipe
        from coba.data.encoders import Encoder, NumericEncoder, OneHotEncoder, StringEncoder
        from coba.data.filters  import CsvReader, LabeledCsvCleaner

        data_id        = self._data_id
        md5_checksum   = self._md5_checksum
        openml_api_key = CobaConfig.Api_Keys['openml']

        data_description_url = f'https://www.openml.org/api/v1/json/data/{data_id}'
        
        type_description_url = f'https://www.openml.org/api/v1/json/data/features/{data_id}'

        if openml_api_key is not None:
            data_description_url += f'?api_key={openml_api_key}'
            type_description_url += f'?api_key={openml_api_key}'

        resp  = ''.join(HttpSource(data_description_url, None, 'descr').read())
        descr = json.loads(resp)["data_set_description"]

        if descr['status'] == 'deactivated':
            raise Exception(f"Openml {data_id} has been deactivated. This is often due to flags on the data.")

        resp  = ''.join(HttpSource(type_description_url, None, 'types').read())
        types = json.loads(resp)["data_features"]["feature"]

        headers : List[str]     = []
        encoders: List[Encoder] = []
        ignored : List[bool]    = []
        target  : str           = ""

        for tipe in types:

            headers.append(tipe['name'])
            ignored.append(tipe['is_ignore'] == 'true' or tipe['is_row_identifier'] == 'true')

            if tipe['is_target'] == 'true':
                target = tipe['name']

            if tipe['data_type'] == 'numeric':
                encoders.append(NumericEncoder())  
            elif tipe['data_type'] == 'nominal' and tipe['is_target'] == 'false':
                encoders.append(OneHotEncoder(singular_if_binary=True))
            elif tipe['data_type'] == 'nominal' and tipe['is_target'] == 'true':
                encoders.append(OneHotEncoder())
            else:
                encoders.append(StringEncoder())

        if isinstance(encoders[headers.index(target)], NumericEncoder):
            target = self._get_classification_target(data_id, openml_api_key)
            ignored[headers.index(target)] = False
            encoders[headers.index(target)] = OneHotEncoder()

        csv_url = f"http://www.openml.org/data/v1/get_csv/{descr['file_id']}"

        source  = HttpSource(csv_url, md5_checksum, f"obser")
        reader  = CsvReader()
        cleaner = LabeledCsvCleaner(target, headers, encoders, ignored, True)

        feature_rows, label_rows = Pipe.join(source, [reader, cleaner]).read()

        return list(feature_rows), list(label_rows)

    def _get_classification_target(self, data_id, openml_api_key):
        task_description_url = f'https://www.openml.org/api/v1/json/task/list/data_id/{data_id}'

        if openml_api_key is not None:        
            task_description_url += f'?api_key={openml_api_key}'

        tasks = json.loads(''.join(HttpSource(task_description_url, None, 'tasks').read()))["tasks"]["task"]

        for task in tasks:
            if task["task_type_id"] == 1: #aka, classification task
                for input in task['input']:
                    if input['name'] == 'target_feature':
                        return input['value'] #just take the first one

        raise Exception(f"Openml {data_id} does not appear to be a classification dataset")

class LambdaSource(Source[Tuple[Sequence[Interaction], Reward]]):

    def __init__(self,
        n_interactions: int,
        context       : Callable[[int               ],Context],
        actions       : Callable[[int,Context       ],Sequence[Action]],
        reward        : Callable[[int,Context,Action],float],
        seed          : int = None) -> None:

        coba.random.seed(seed)

        self._interactions: List[Interaction] = []
        self._reward = MemoryReward()

        for i in range(n_interactions):
            _context  = context(i)
            _actions  = actions(i,_context)
            _rewards  = [reward(i, _context, _action) for _action in _actions]

            self._interactions.append(Interaction(_context, _actions, i)) #type: ignore
            for a,r in zip(_actions, _rewards): self._reward.add_observation((i,a,r))

    def read(self) -> Tuple[Sequence[Interaction], Reward]:
        return (self._interactions, self._reward)

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

class LambdaSimulation(Source[Simulation]):
    """A Simulation created from lambda functions that generate contexts, actions and rewards.

    Remarks:
        This implementation is useful for creating simulations from defined distributions.
    """

    def __init__(self,
        n_interactions: int,
        context       : Callable[[int               ],Context],
        action_set    : Callable[[int,Context       ],Sequence[Action]], 
        reward        : Callable[[int,Context,Action],float],
        seed          : int = None) -> None:
        """Instantiate a LambdaSimulation.

        Args:
            n_interactions: How many interactions the LambdaSimulation should have.
            context: A function that should return a context given an index in `range(n_interactions)`.
            action_set: A function that should return all valid actions for a given index and context.
            reward: A function that should return the reward for the index, context and action.
        """

        self._source = LambdaSource(n_interactions, context, action_set, reward, seed) # type: ignore

    def read(self) -> Simulation:
        return MemorySimulation(*self._source.read()) #type: ignore

    def __repr__(self) -> str:
        return '"LambdaSimulation"'

class ClassificationSimulation(MemorySimulation):
    """A simulation created from classifier data with features and labels.

    ClassificationSimulation turns labeled observations from a classification data set
    set, into interactions. For each interaction the feature set becomes the context and 
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

    def __init__(self, features: Sequence[Context], labels: Sequence[Action]) -> None:
        """Instantiate a ClassificationSimulation.

        Args:
            features: The collection of features used for the original classifier problem.
            labels: The collection of labels assigned to each observation of features.
        """

        assert len(features) == len(labels), "Mismatched lengths of features and labels"


        self.one_hot_encoder = OneHotEncoder(list(set(labels)))

        labels     = self.one_hot_encoder.encode(labels)
        action_set = list(set(labels))

        interactions = [ Interaction(context, action_set, i) for i, context in enumerate(features) ] #type: ignore
        reward      = ClassificationReward(list(enumerate(labels)))

        super().__init__(interactions, reward) #type:ignore

class OpenmlSimulation(Source[Simulation]):
    """A simulation created from openml data with features and labels.

    OpenmlSimulation turns labeled observations from a classification data set
    set, into interactions. For each interaction the feature set becomes the context and 
    all possible labels become the actions. Rewards for each interaction are created by 
    assigning a reward of 1 for taking the correct action (i.e., choosing the correct
    label) and a reward of 0 for taking any other action (i.e., choosing any of the
    incorrect lables).

    Remark:
        This class when created from a data set will load all data into memory. Be careful when 
        doing this if you are working with a large dataset. To reduce memory usage you can provide
        meta information upfront that will allow features to be correctly encoded while the
        dataset is being streamed instead of waiting until the end of the data to train an encoder.
    """

    def __init__(self, id: int, md5_checksum: str = None) -> None:
        self._source = OpenmlSource(id, md5_checksum)

    def read(self) -> Simulation:        
        return ClassificationSimulation(*self._source.read())

    def __repr__(self) -> str:
        return f'{{"OpenmlSimulation":{self._source._data_id}}}'

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

class Shuffle(Filter[Simulation,Simulation]):
    def __init__(self, seed:Optional[int]) -> None:
        self._seed = seed

    def filter(self, item: Simulation) -> Simulation:  
        shuffled_interactions = coba.random.CobaRandom(self._seed).shuffle(item.interactions)
        return MemorySimulation(shuffled_interactions, item.reward)

    def __repr__(self) -> str:
        return f'{{"Shuffle":{self._seed}}}'

class Take(Filter[Simulation,Simulation]):
    def __init__(self, count:Optional[int]) -> None:
        self._count = count

    def filter(self, item: Simulation) -> Simulation:

        if self._count is None:
            return item

        if self._count > len(item.interactions):
            return MemorySimulation([], item.reward)

        return MemorySimulation(item.interactions[0:self._count], item.reward)

    def __repr__(self) -> str:
        return f'{{"Take":{json.dumps(self._count)}}}'

class Batch(Filter[Simulation,BatchedSimulation]):
    
    @overload
    def __init__(self,*,count: int): ...

    @overload
    def __init__(self,*,size: int): ...

    @overload
    def __init__(self,*,sizes: Sequence[int]): ...

    def __init__(self, **kwargs) -> None:
        
        self._kwargs = kwargs
        self._count  = cast(Optional[int], kwargs.get("count", None))
        self._size   = cast(Optional[int], kwargs.get("size", None))
        self._sizes  = cast(Optional[Sequence[int]], kwargs.get("sizes", None))

    def filter(self, item: Simulation) -> BatchedSimulation:
        
        sizes: Optional[Sequence[int]] = None

        if self._count is not None:
            n         = len(item.interactions)
            sizes     = [int(float(n)/(self._count))] * self._count
            remainder = n - sum(sizes)
            for i in range(remainder): sizes[int(i*len(sizes)/remainder)] += 1
        
        if self._size is not None:
            n     = len(item.interactions)
            sizes = [self._size] * int(n/self._size)

        if self._sizes is not None:
            sizes = self._sizes

        if sizes is None:
            raise Exception("We were unable to determine an approriate batch sizes")
        else:
            return BatchedSimulation(item, sizes)

    def __repr__(self) -> str:
        return f'{{"Batch":{json.dumps(self._kwargs, separators=(",",":"))}}}'

class PCA(Filter[Simulation,Simulation]):

    def __init__(self) -> None:
        PackageChecker.numpy("PCA.__init__")

    def filter(self, simulation: Simulation) -> Simulation:
        
        PackageChecker.numpy("PcaSimulation.__init__")

        import numpy as np #type: ignore

        contexts = [ list(cast(Tuple[float,...],i.context)) for i in simulation.interactions]

        feat_matrix          = np.array(contexts)
        comp_vals, comp_vecs = np.linalg.eig(np.cov(feat_matrix.T))

        comp_vecs = comp_vecs[:,comp_vals > 0]
        comp_vals = comp_vals[comp_vals > 0]

        new_contexts = (feat_matrix @ comp_vecs ) / np.sqrt(comp_vals) #type:ignore
        new_contexts = new_contexts[:,np.argsort(-comp_vals)]

        interactions = [ Interaction(tuple(c), i.actions, i.key) for c, i in zip(new_contexts,simulation.interactions) ]

        return MemorySimulation(interactions, simulation.reward)

    def __repr__(self) -> str:
        return '"PCA"'

class Sort(Filter[Simulation,Simulation]):

    def __init__(self, indexes: Sequence[int]) -> None:
        self.indexes = indexes

    def filter(self, simulation: Simulation) -> Simulation:
        
        sort_key            = lambda interaction: tuple([interaction.context[i] for i in self.indexes ])
        sorted_interactions = list(sorted(simulation.interactions, key=sort_key))

        return MemorySimulation(sorted_interactions, simulation.reward)

    def __repr__(self) -> str:
        return f'{{"Sort":{json.dumps(self.indexes, separators=(",",":"))}}}'