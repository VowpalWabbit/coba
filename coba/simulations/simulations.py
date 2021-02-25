import collections
import json
from abc import ABC, abstractmethod

from hashlib import md5
from itertools import accumulate
from typing import (
    Optional, Sequence, List, Callable, 
    Hashable, Any, Tuple, overload, cast,
    Dict
)

import coba.random

from coba.tools import PackageChecker, CobaConfig
from coba.data.sources import MemorySource, Source, HttpSource
from coba.data.encoders import OneHotEncoder
from coba.data.filters import Filter

from coba.simulations.rewards import Reward, MemoryReward, ClassificationReward
from coba.simulations import Key, Context, Action

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

        d_key = None
        t_key = None
        o_key = None

        try:
            data_id        = self._data_id
            md5_checksum   = self._md5_checksum

            d_key = f'https://www.openml.org/api/v1/json/data/{data_id}'
            t_key = f'https://www.openml.org/api/v1/json/data/features/{data_id}'

            d_bytes  = self._query(d_key, "descr")            
            d_object = json.loads(d_bytes.decode('utf-8'))["data_set_description"]

            if d_object['status'] == 'deactivated':
                raise Exception(f"Openml {data_id} has been deactivated. This is often due to flags on the data.")

            t_bytes  = self._query(t_key, "types")
            t_object = json.loads(t_bytes.decode('utf-8'))["data_features"]["feature"]

            headers : List[str]     = []
            encoders: List[Encoder] = []
            ignored : List[bool]    = []
            target  : str           = ""

            for tipe in t_object:

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
                target = self._get_classification_target(data_id)
                ignored[headers.index(target)] = False
                encoders[headers.index(target)] = OneHotEncoder()

            o_key   = f"http://www.openml.org/data/v1/get_csv/{d_object['file_id']}"
            o_bytes = self._query(o_key, "obser", md5_checksum)
            
            source  = MemorySource(o_bytes.decode('utf-8').splitlines())
            reader  = CsvReader()
            cleaner = LabeledCsvCleaner(target, headers, encoders, ignored, True)

            feature_rows, label_rows = Pipe.join(source, [reader, cleaner]).read()

            #we only cache after all the data has been successfully loaded
            for key,bytes in [ (d_key, d_bytes), (t_key, t_bytes), (o_key, o_bytes) ]:
                CobaConfig.Cacher.put(key,bytes)

            return list(feature_rows), list(label_rows)

        except:

            #if something went wrong we want to clear the cache 
            #in case the cache has become corrupted somehow
            for key in [d_key, t_key, o_key]:
                if key is not None: CobaConfig.Cacher.rmv(key)

            raise

    def _query(self, url:str, description:str, checksum:str=None) -> bytes:
        
        if url in CobaConfig.Cacher:
            with CobaConfig.Logger.time(f'loading {description} from cache... '):
                bites = CobaConfig.Cacher.get(url)

        else:

            api_key = CobaConfig.Api_Keys['openml']

            with CobaConfig.Logger.time(f'loading {description} from http... '):
                response = HttpSource(url + (f'?api_key={api_key}' if api_key else '')).read()

            if response.status_code == 412:
                if 'please provide api key' in response.text:
                    message = (
                        "An API Key is needed to access openml's rest API. A key can be obtained by creating an "
                        "openml account at openml.org. Once a key has been obtained it should be placed within "
                        "~/.coba as { \"api_keys\" : { \"openml\" : \"<your key here>\", } }.")
                    raise Exception(message) from None

                if 'authentication failed' in response.text:
                    message = (
                        "The API Key you provided no longer seems to be valid. You may need to create a new one"
                        "longing into your openml account and regenerating a key. After regenerating the new key "
                        "should be placed in ~/.coba as { \"api_keys\" : { \"openml\" : \"<your key here>\", } }.")
                    raise Exception(message) from None

            if response.status_code == 404:
                message = (
                    "We're sorry but we were unable to find the requested dataset on openml. The most likely cause "
                    "for this is openml not providing the requested dataset in a format that COBA can process.")
                raise Exception(message) from None

            if "Usually due to high server load" in response.text:
                message = (
                    "Openml reported an error that they believe is likely caused by high server loads ."
                    "Openml recommends that you try again in a few seconds. Additionally, if not already "
                    "done, consider setting up a DiskCache in coba config to reduce the number of openml "
                    "calls in the future.")
                raise Exception(message) from None

            bites = response.content

        if checksum is not None and md5(bites).hexdigest() != checksum:
            
            #if the cache has become corrupted we need to clear it
            CobaConfig.Cacher.rmv(url)

            message = (
                f"The response from {url} did not match the given checksum {checksum}. This could be the result "
                "of network errors or the file becoming corrupted. Please consider downloading the file again. "
                "If the error persists you may want to manually download and reference the file.")
            raise Exception(message) from None

        return bites
        
    def _get_classification_target(self, data_id):

        t_key = f'https://www.openml.org/api/v1/json/task/list/data_id/{data_id}'
        t_bites = self._query(t_key, "tasks")

        tasks = json.loads(t_bites.decode('utf-8'))["tasks"]["task"]

        for task in tasks:
            if task["task_type_id"] == 1: #aka, classification task
                for input in task['input']:
                    if input['name'] == 'target_feature':
                        return input['value'] #just take the first one

        CobaConfig.Cacher.put(t_key,t_bites)

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
