"""The simulations module contains core classes and types for defining contextual bandit simulations.

This module contains the abstract interface expected for bandit simulations along with the 
class defining an Interaction within a bandit simulation. Additionally, this module also contains 
the type hints for Context, Action and Reward. These type hints don't contain any functionality. 
Rather, they simply make it possible to use static type checking for any project that desires 
to do so.

TODO Add RegressionSimulation
"""

import json

from itertools import chain, accumulate
from abc import ABC, abstractmethod
from typing import (
    Optional, Sequence, List, Callable, TypeVar, 
    Generic, Hashable, Any, Tuple, overload, cast
)

import coba.random

from coba.tools import PackageChecker, CobaConfig
from coba.data.sources import Source, HttpSource, MemorySource
from coba.data.encoders import OneHotEncoder
from coba.data.filters import Filter

Context = Optional[Hashable]
Action  = Hashable
Reward  = float
Key     = int
Choice  = int

_C_out = TypeVar('_C_out', bound=Context, covariant=True)
_A_out = TypeVar('_A_out', bound=Action , covariant=True)

class Interaction(Generic[_C_out, _A_out]):
    """A class to contain all data needed to represent an interaction in a bandit simulation."""

    #this is a problem with pylance compaining about covariance in constructor so we have to type ignore it. 
    #See this ticket in mypy for more info https://github.com/python/mypy/issues/2850
    def __init__(self, context: _C_out, actions: Sequence[_A_out], key: Key = 0) -> None: #type: ignore
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
    def context(self) -> _C_out:
        """The interaction's context description."""
        return self._context

    @property
    def actions(self) -> Sequence[_A_out]:
        """The interactions's available actions."""
        return self._actions
    
    @property
    def key(self) -> Key:
        """A unique key identifying the interaction."""
        return self._key

class OpenmlClassificationSource(Source[Tuple[Sequence[Context], Sequence[Action]]]):

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

        resp  = ''.join(HttpSource(data_description_url, '.json', None, 'descr').read())
        descr = json.loads(resp)["data_set_description"]

        if descr['status'] == 'deactivated':
            raise Exception(f"Openml {data_id} has been deactivated. This is often due to flags on the data.")

        resp  = ''.join(HttpSource(type_description_url, '.json', None, 'types').read())
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

        source  = HttpSource(csv_url, ".csv", md5_checksum, f"openml {data_id}")
        reader  = CsvReader()
        cleaner = LabeledCsvCleaner(target, headers, encoders, ignored, True)

        feature_rows, label_rows = Pipe.join(source, [reader, cleaner]).read()

        return list(feature_rows), list(label_rows)

    def _get_classification_target(self, data_id, openml_api_key):
        task_description_url = f'https://www.openml.org/api/v1/json/task/list/data_id/{data_id}'

        if openml_api_key is not None:        
            task_description_url += f'?api_key={openml_api_key}'

        tasks = json.loads(''.join(HttpSource(task_description_url, '.json', None, 'tasks').read()))["tasks"]["task"]

        for task in tasks:
            if task["task_type_id"] == 1: #aka, classification task
                for input in task['input']:
                    if input['name'] == 'target_feature':
                        return input['value'] #just take the first one

        raise Exception(f"Openml {data_id} does not appear to be a classification dataset")

class LambdaSource(Source[Tuple[Sequence[Interaction[_C_out, _A_out]], Sequence[Sequence[Reward]]]]):

    def __init__(self,
        n_interactions: int,
        context       : Callable[[int],_C_out],
        action_set    : Callable[[int],Sequence[_A_out]], 
        reward        : Callable[[_C_out,_A_out],Reward],
        seed          : int = None) -> None:

        coba.random.seed(seed)

        interactions: List[Interaction[_C_out, _A_out]] = []
        reward_sets : List[Sequence[Reward]]            = []

        for i in range(n_interactions):
            _context    = context(i)
            _action_set = action_set(i)
            _reward_set = [reward(_context, _action) for _action in _action_set]

            interactions.append(Interaction(_context, _action_set, i)) #type: ignore
            reward_sets.append(_reward_set)

        self._source = MemorySource((interactions, reward_sets))

    def read(self) -> Tuple[Sequence[Interaction[_C_out, _A_out]], Sequence[Sequence[Reward]]]:
        return self._source.read()

class Simulation(Generic[_C_out, _A_out], ABC):
    """The simulation interface."""

    @property
    @abstractmethod
    def interactions(self) -> Sequence[Interaction[_C_out, _A_out]]:
        """The sequence of interactions in a simulation.

        Remarks:
            Interactions should always be re-iterable. So long as interactions is a Sequence 
            this will always be the case. If interactions is changed to Iterable in the future
            then it will be possible for it to only allow enumeration one time and care will need
            to be taken.
        """
        ...

    @abstractmethod
    def reward(self, choices: Sequence[Tuple[Key,Choice]] ) -> Sequence[Reward]:
        """The observed rewards for interactions (identified by its key) and their selected action indexes.

        Args:
            choices: A sequence of tuples containing an interaction key and an action index.

        Returns:
            A sequence of tuples containing context, action, and reward for the requested 
            interaction/action. This sequence will always align with the provided choices.
        """
        ...

class MemorySimulation(Simulation[_C_out, _A_out]):
    """A Simulation implementation created from in memory sequences of contexts, actions and rewards."""

    def __init__(self, 
        interactions: Sequence[Interaction[_C_out, _A_out]],
        reward_sets : Sequence[Sequence[Reward]]) -> None:
        """Instantiate a MemorySimulation.

        Args:
            contexts: A collection of contexts to turn into a simulation.
            action_sets: A collection of action sets to turn into a simulation
            reward_sets: A collection of reward sets to turn into a simulation 
        """

        assert len(interactions) == len(reward_sets), "Mismatched lengths of interactions and rewards"

        self._interactions = interactions

        choices = chain.from_iterable([ [ (i.key, a) for a in range(len(i.actions)) ] for i in self._interactions ])
        rewards = chain.from_iterable(reward_sets)

        self._rewards = dict(zip(choices,rewards))

    @property
    def interactions(self) -> Sequence[Interaction[_C_out,_A_out]]:
        """The interactions in this simulation.

        Remarks:
            See the Simulation base class for more information.
        """
        return self._interactions

    def reward(self, choices: Sequence[Tuple[Key,Choice]]) -> Sequence[Reward]:
        """The observed rewards for interactions (identified by its key) and their selected action indexes.

        Remarks:
            See the Simulation base class for more information.
        """

        return [ self._rewards[choice] for choice in choices]

class ClassificationSimulation(MemorySimulation[_C_out, Tuple[int,...]]):
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

    def __init__(self, features: Sequence[_C_out], labels: Sequence[Action]) -> None:
        """Instantiate a ClassificationSimulation.

        Args:
            features: The collection of features used for the original classifier problem.
            labels: The collection of labels assigned to each observation of features.
        """

        assert len(features) == len(labels), "Mismatched lengths of features and labels"

        label_set  = list(set(labels))
        action_set = OneHotEncoder(label_set).encode(label_set)

        interactions = [ Interaction(context, action_set, i) for i, context in enumerate(features) ] #type: ignore
        rewards      = OneHotEncoder(label_set).encode(labels)

        self.label_set = label_set
        super().__init__(interactions, rewards) #type:ignore

class LambdaSimulation(Source[Simulation[_C_out, _A_out]]):
    """A Simulation created from lambda functions that generate contexts, actions and rewards.

    Remarks:
        This implementation is useful for creating simulations from defined distributions.
    """

    def __init__(self,
        n_interactions: int,
        context       : Callable[[int],_C_out],
        action_set    : Callable[[int],Sequence[_A_out]], 
        reward        : Callable[[_C_out,_A_out],Reward],
        seed          : int = None) -> None:
        """Instantiate a LambdaSimulation.

        Args:
            n_interactions: How many interactions the LambdaSimulation should have.
            context: A function that should return a context given an index in `range(n_interactions)`.
            action_set: A function that should return all valid actions for a given context.
            reward: A function that should return the reward for a context and action.
        """

        self._source = LambdaSource(n_interactions, context, action_set, reward, seed) # type: ignore

    def read(self) -> Simulation[_C_out, _A_out]:
        return MemorySimulation(*self._source.read()) #type: ignore

    def __repr__(self) -> str:
        return '"LambdaSimulation"'

class OpenmlSimulation(Source[ClassificationSimulation[Context]]):
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
        self._openml_source = OpenmlClassificationSource(id, md5_checksum)

    def read(self) -> ClassificationSimulation[Context]:
        with CobaConfig.Logger.log(f"loading {self}..."):
            return ClassificationSimulation(*self._openml_source.read())

    def __repr__(self) -> str:
        return f'{{"OpenmlSimulation":{self._openml_source._data_id}}}'

class ShuffleSimulation(Simulation[_C_out, _A_out]):
    def __init__(self, seed: Optional[int], simulation: Simulation[_C_out, _A_out]) -> None:

        self._simulation = simulation
        self._seed       = seed
        self._interactions = coba.random.CobaRandom(self._seed).shuffle(simulation.interactions)

    @property
    def interactions(self) -> Sequence[Interaction[_C_out,_A_out]]:
        """The interactions in this simulation.

        Remarks:
            See the Simulation base class for more information.
        """
        return self._interactions

    def reward(self, choices: Sequence[Tuple[Key,Choice]]) -> Sequence[Reward]:
        """The observed rewards for interactions (identified by its key) and their selected action indexes.

        Remarks:
            See the Simulation base class for more information.
        """

        return self._simulation.reward(choices)

class TakeSimulation(Simulation[_C_out, _A_out]):
    def __init__(self, count:int, simulation: Simulation[_C_out, _A_out]) -> None:

        self._simulation   = simulation
        self._count        = count
        self._interactions = simulation.interactions[0:count]

    @property
    def interactions(self) -> Sequence[Interaction[_C_out,_A_out]]:
        """The interactions in this simulation.

        Remarks:
            See the Simulation base class for more information.
        """
        return self._interactions

    def reward(self, choices: Sequence[Tuple[Key,Choice]]) -> Sequence[Reward]:
        """The observed rewards for interactions (identified by its key) and their selected action indexes.

        Remarks:
            See the Simulation base class for more information.
        """

        return self._simulation.reward(choices)

class BatchedSimulation(Simulation[_C_out, _A_out]):
    """A simulation whose interactions have been batched."""

    def __init__(self, simulation: Simulation[_C_out, _A_out], batch_sizes: Sequence[int]) -> None:
        self._simulation = simulation

        #remove Nones and 0s
        batch_sizes = list(filter(None, batch_sizes))

        if len(batch_sizes) == 0:
            self._batches = []
            self._interactions = []
        else:
            batch_slices  = list(accumulate([0] + list(batch_sizes)))
            self._batches = [simulation.interactions[batch_slices[i]:batch_slices[i+1]] for i in range(len(batch_slices)-1) ]
            self._interactions = simulation.interactions[0:sum(batch_sizes)]

    @property
    def interaction_batches(self) -> Sequence[Sequence[Interaction[_C_out, _A_out]]]:
        """The sequence of batches of interactions in a simulation."""
        return self._batches

    @property
    def interactions(self) -> Sequence[Interaction[_C_out,_A_out]]:
        """The interactions in this simulation.

        Remarks:
            See the Simulation base class for more information.
        """
        return self._interactions

    def reward(self, choices: Sequence[Tuple[Key,Choice]] ) -> Sequence[Reward]:
        """The observed rewards for interactions (identified by its key) and their selected action indexes.

        Args:
            choices: A sequence of tuples containing an interaction key and an action index.

        Returns:
            A sequence of tuples containing context, action, and reward for the requested 
            interaction/action. This sequence will always align with the provided choices.
        """
        return self._simulation.reward(choices)

class PcaSimulation(Simulation[Tuple[float,...], _A_out]):
    def __init__(self, simulation: Simulation[Tuple[float,...], _A_out]) -> None:
        
        PackageChecker.numpy("PcaSimulation.__init__")
        
        import numpy as np #type: ignore

        feat_matrix          = np.array([list(i.context) for i in simulation.interactions])
        comp_vals, comp_vecs = np.linalg.eig(np.cov(feat_matrix.T))
        
        comp_vecs = comp_vecs[:,comp_vals > 0]
        comp_vals = comp_vals[comp_vals > 0]

        new_contexts = (feat_matrix @ comp_vecs ) / np.sqrt(comp_vals) #type:ignore
        new_contexts = new_contexts[:,np.argsort(-comp_vals)]

        self._simulation = simulation
        self._interactions = [ Interaction(tuple(c), i.actions, i.key) for c, i in zip(new_contexts,simulation.interactions) ]

    @property
    def interactions(self) -> Sequence[Interaction[Tuple[float,...],_A_out]]:
        """The interactions in this simulation.

        Remarks:
            See the Simulation base class for more information.
        """
        return self._interactions

    def reward(self, choices: Sequence[Tuple[Key,Choice]]) -> Sequence[Reward]:
        """The observed rewards for interactions (identified by its key) and their selected action indexes.

        Remarks:
            See the Simulation base class for more information.
        """

        return self._simulation.reward(choices)

class SortSimulation(Simulation[_C_out, _A_out]):
    def __init__(self, context_keys: Sequence[int], simulation: Simulation[_C_out, _A_out]) -> None:
        self._simulation   = simulation
        self._context_keys = context_keys

        sort_key = lambda interaction: tuple([interaction.context[key] for key in self._context_keys ])
        self._interactions = list(sorted(simulation.interactions, key=sort_key))

    @property
    def interactions(self) -> Sequence[Interaction[_C_out,_A_out]]:
        """The interactions in this simulation.

        Remarks:
            See the Simulation base class for more information.
        """
        return self._interactions

    def reward(self, choices: Sequence[Tuple[Key,Choice]]) -> Sequence[Reward]:
        """The observed rewards for interactions (identified by its key) and their selected action indexes.

        Remarks:
            See the Simulation base class for more information.
        """

        return self._simulation.reward(choices)

class Shuffle(Filter[Simulation[Context,Action],Simulation[Context,Action]]):
    def __init__(self, seed:Optional[int]) -> None:
        self._seed = seed

    def filter(self, item: Simulation[Context,Action]) -> Simulation[Context,Action]:        
        return ShuffleSimulation(self._seed, item)

    def __repr__(self) -> str:
        return f'{{"Shuffle":{self._seed}}}'

class Take(Filter[Simulation[Context,Action],Simulation[Context,Action]]):
    def __init__(self, count:Optional[int]) -> None:
        self._count = count

    def filter(self, item: Simulation[Context,Action]) -> Simulation[Context,Action]:

        if self._count is None:
            return TakeSimulation(len(item.interactions), item)

        if self._count > len(item.interactions):
            return TakeSimulation(0, item)

        return TakeSimulation(self._count, item)

    def __repr__(self) -> str:
        return f'{{"Take":{json.dumps(self._count)}}}'

class Batch(Filter[Simulation[Context,Action],BatchedSimulation[Context,Action]]):
    
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

    def filter(self, item: Simulation[Context,Action]) -> BatchedSimulation[Context,Action]:
        
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

class PCA(Filter[Simulation[Tuple[float,...],Action],Simulation[Context,Action]]):

    def __init__(self) -> None:
        PackageChecker.numpy("PCA.__init__")

    def filter(self, item: Simulation[Tuple[float,...],Action]) -> Simulation[Context,Action]:
        return PcaSimulation(item)

    def __repr__(self) -> str:
        return '"PCA"'

class Sort(Filter[Simulation[Context,Action],Simulation[Context,Action]]):

    def __init__(self, context_keys: Sequence[int]) -> None:
        self._context_keys = context_keys

    def filter(self, item: Simulation[Context,Action]) -> Simulation[Context,Action]:        
        return SortSimulation(self._context_keys, item)

    def __repr__(self) -> str:
        return f'{{"Sort":{json.dumps(self._context_keys, separators=(",",":"))}}}'