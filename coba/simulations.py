"""The simulations module contains core classes and types for defining contextual bandit simulations.

This module contains the abstract interface expected for bandit simulations along with the 
class defining an Interaction within a bandit simulation. Additionally, this module also contains 
the type hints for State, Action and Reward. These type hints don't contain any functionality. 
Rather, they simply make it possible to use static type checking for any project that desires 
to do so.

Todo:
    * Add RegressionSimulation
"""

import csv
import json
import urllib.request

import time

from itertools import repeat, count, chain
from http.client import HTTPResponse
from warnings import warn
from gzip import  decompress
from contextlib import closing
from abc import ABC, abstractmethod
from hashlib import md5
from typing import Optional, Iterable, Sequence, List, Union, Callable, TypeVar, Generic, Hashable, Dict, cast, Any, ContextManager, IO, Tuple, overload

from coba import random as cb_random
from coba.preprocessing import Metadata, OneHotEncoder, NumericEncoder, Encoder

State  = Optional[Hashable]
Action = Hashable
Reward = float
Key    = int 
Choice = Tuple[Key,int]

_S_out = TypeVar('_S_out', bound=State, covariant=True)
_A_out = TypeVar('_A_out', bound=Action, covariant=True)

class Interaction(Generic[_S_out, _A_out]):
    """A class to contain all data needed to represent an interaction in a bandit simulation."""

    def __init__(self, state: _S_out, actions: Sequence[_A_out]) -> None:
        """Instantiate Interaction.

        Args
            state: Features describing the interactions's state. Will be `None` for multi-armed bandit simulations.
            actions: Features describing available actions in the interaction.
        """

        assert actions, "At least one action must be provided to interact"

        self._state   = state
        self._actions = actions

    @property
    def state(self) -> _S_out:
        """The interaction's state description."""
        return self._state

    @property
    def actions(self) -> Sequence[_A_out]:
        """The interactions's available actions."""
        return self._actions

class KeyedInteraction(Interaction[_S_out, _A_out]):
    """An interaction that's been given a unique identifier."""

    def __init__(self, key: Key, state: _S_out, actions: Sequence[_A_out]) -> None:
        """Instantiate a KeyedInteraction.

        Args:
            key: The unique key identifying the interaction.
            state: The context in which the interaction is ocurring.
            actions: The actions available to take in the interaction.
        """
        
        super().__init__(state,actions)
        
        self._key = key

    @property
    def key(self) -> Key:
        """The unique key associated with the interaction."""
        return self._key

    @property
    def choices(self) -> Sequence[Choice]:
        """A convenience method providing all possible choices in the expected reward format."""
        return list(zip(repeat(self._key), range(len(self._actions))))

class Simulation(Generic[_S_out, _A_out], ABC):
    """The simulation interface."""

    @staticmethod
    def from_json(json_val:Union[str, Dict[str, Any]]) -> 'Simulation[State,Action]':
        """Construct a Simulation object from JSON.

        Args:
            json_val: Either a json string or the decoded json object.

        Returns:
            The Simulation representation of the given JSON string or object.
        """

        config = json.loads(json_val) if isinstance(json_val,str) else json_val

        no_shuffle  : Callable[[Simulation[State,Action]], Simulation] = lambda sim: sim
        seed_shuffle: Callable[[Simulation[State,Action]], Simulation] = lambda sim: ShuffleSimulation(sim, config["seed"])

        lazy_loader: Callable[[Callable[[],Simulation[State,Action]]], Simulation] = lambda sim_factory: LazySimulation(sim_factory)
        now_loader :Callable[[Callable[[],Simulation[State,Action]]], Simulation]  = lambda sim_factory: sim_factory()

        shuffler = seed_shuffle if "seed" in config and config["seed"] is not None else no_shuffle
        loader   = lazy_loader  if "lazy" in config and config["lazy"] == True     else now_loader

        if config["type"] == "classification":
            return  loader(lambda: shuffler(ClassificationSimulation.from_json(config["from"])))

        raise Exception("We were unable to recognize the provided simulation type")

    @property
    @abstractmethod
    def interactions(self) -> Sequence[KeyedInteraction[_S_out, _A_out]]:
        """The sequence of interactions in a simulation.

        Remarks:
            All Benchmark assume that interactions is re-iterable. So long as interactions is 
            a Sequence it will always be re-iterable. If interactions was merely Iterable then 
            it would be possible for it to only allow enumeration one time.
        """
        ...
    
    @abstractmethod
    def rewards(self, choices: Sequence[Choice] ) -> Sequence[Tuple[_S_out, _A_out, Reward]]:
        """The observed rewards for interactions (identified by its key) and their selected action indexes.

        Args:
            choices: A sequence of tuples containing an interaction key and an action index.

        Returns:
            A sequence of tuples containing state, action, and reward for the requested 
            interaction/action. This sequence will always align with the provided choices.
        """
        ...

class LazySimulation(Simulation[_S_out, _A_out]):
    """A Simulation implementation which supports loading and unloading from memory.""" 
    
    def __init__(self, sim_factory = Callable[[],Simulation[_S_out,_A_out]]) -> None:
        """Instantiate a LazySimulation
        
        Args:
            sim_factory: A factory method for loading the simulation when requested.
        """
        self._sim_factory = sim_factory
        self._simulation: Optional[Simulation[_S_out, _A_out]]  = None

    def load(self) -> 'LazySimulation[_S_out,_A_out]':
        """Load the simulation into memory. If already loaded do nothing."""
        
        if self._simulation is None:
            self._simulation = self._sim_factory()

        return self

    def unload(self) -> 'LazySimulation[_S_out,_A_out]':
        """Unload the simulation from memory."""
        
        self._simulation = None
        
        return self


    @property
    def interactions(self) -> Sequence[KeyedInteraction[_S_out,_A_out]]:
        """The interactions in this simulation.

        Remarks:
            See the Simulation base class for more information.
        """

        if self._simulation is not None:
            return self._simulation.interactions
        
        raise Exception("A LazySimulation must be loaded before it can be used.")

    def rewards(self, choices: Sequence[Choice]) -> Sequence[Tuple[_S_out, _A_out,Reward]]:
        """The observed rewards for interactions (identified by its key) and their selected action indexes.

        Remarks:
            See the Simulation base class for more information.
        """
        
        if self._simulation is not None:
            return self._simulation.rewards(choices)

        raise Exception("A LazySimulation must be loaded before it can be used.")

class MemorySimulation(Simulation[_S_out, _A_out]):
    """A Simulation implementation created from in memory sequences of states, actions and rewards."""

    def __init__(self, 
        states: Sequence[_S_out], 
        action_sets: Sequence[Sequence[_A_out]], 
        reward_sets: Sequence[Sequence[Reward]]) -> None:
        """Instantiate a MemorySimulation.

        Args:
            states: A collection of states to turn into a simulation.
            action_sets: A collection of action sets to turn into a simulation
            reward_sets: A collection of reward sets to turn into a simulation 
        """

        assert len(states) == len(action_sets) == len(reward_sets), "Mismatched lengths of states, actions and rewards"

        start = time.time()
        self._interactions = list(map(KeyedInteraction, count(), states, action_sets))
        print(time.time() - start)
        
        start = time.time()
        choices = chain.from_iterable([ i.choices for i in self._interactions])
        rewards = chain.from_iterable(reward_sets)
        print(time.time() - start)

        start = time.time()
        self._rewards = dict(zip(choices,rewards))
        print(time.time() - start)        

    @property
    def interactions(self) -> Sequence[KeyedInteraction[_S_out,_A_out]]:
        """The interactions in this simulation.

        Remarks:
            See the Simulation base class for more information.
        """
        return self._interactions

    def rewards(self, choices: Sequence[Choice]) -> Sequence[Tuple[_S_out, _A_out,Reward]]:
        """The observed rewards for interactions (identified by its key) and their selected action indexes.

        Remarks:
            See the Simulation base class for more information.
        """

        out: List[Tuple[_S_out, _A_out, Reward]] = []

        for choice in choices:

            state  = self._interactions[choice[0]].state
            action = self._interactions[choice[0]].actions[choice[1]]

            out.append((state, action, self._rewards[choice]))

        return out

class LambdaSimulation(Simulation[_S_out, _A_out]):
    """A Simulation created from lambda functions that generate states, actions and rewards.

    Remarks:
        This implementation is useful for creating simulations from defined distributions.
    """

    def __init__(self,
                 n_interactions: int,
                 state: Callable[[int],_S_out],
                 action_set: Callable[[_S_out],Sequence[_A_out]], 
                 reward: Callable[[_S_out,_A_out],Reward]) -> None:
        """Instantiate a LambdaSimulation.

        Args:
            n_interactions: How many interactions the LambdaSimulation should have.
            state: A function that should return a state given an index in `range(n_interactions)`.
            action_set: A function that should return all valid actions for a given state.
            reward: A function that should return the reward for a state and action.
        """

        states     : List[_S_out]           = []
        action_sets: List[Sequence[_A_out]] = []
        reward_sets: List[Sequence[Reward]] = []

        for i in range(n_interactions):
            _state      = state(i)
            _action_set = action_set(_state)
            _reward_set = [reward(_state, _action) for _action in _action_set]

            states     .append(_state)
            action_sets.append(_action_set)
            reward_sets.append(_reward_set)

        self._simulation = MemorySimulation(states, action_sets, reward_sets)

    @property
    def interactions(self) -> Sequence[KeyedInteraction[_S_out,_A_out]]:
        """The interactions in this simulation.

        Remarks:
            See the Simulation base class for more information.
        """
        return self._simulation.interactions

    def rewards(self, choices: Sequence[Choice]) -> Sequence[Tuple[_S_out,_A_out,Reward]]:
        """The observed rewards for interactions (identified by its key) and their selected action indexes.

        Remarks:
            See the Simulation base class for more information.        
        """

        return self._simulation.rewards(choices)

class ShuffleSimulation(Simulation[_S_out, _A_out]):
    """A simulation which created from an existing simulation by shuffling interactions.

    Remarks:
        Shuffling is applied one time upon creation and after that interaction order is fixed.
        Shuffling does not change the original simulation's interaction order or copy the
        original interactions in memory. Shuffling is guaranteed to be deterministic 
        according any given to seed regardless of the local Python execution environment.
    """

    def __init__(self, simulation: Simulation[_S_out,_A_out], seed: Optional[int] = None):
        """Instantiate a ShuffleSimulation

        Args:
            simulation: The simulation we which to shuffle interaction order for.
            seed: The seed we wish to use in determining the shuffle order.
        """

        cb_random.seed(seed)

        self._interactions = cb_random.shuffle(simulation.interactions)
        self._rewards      = simulation.rewards

    @property
    def interactions(self) -> Sequence[KeyedInteraction[_S_out,_A_out]]:
        """The interactions in this simulation.

        Remarks:
            See the Simulation base class for more information.
        """

        return self._interactions

    def rewards(self, choices: Sequence[Choice]) -> Sequence[Tuple[_S_out,_A_out,Reward]]:
        """The observed rewards for interactions (identified by its key) and their selected action indexes.
        
        Remarks:
            See the Simulation base class for more information.        
        """

        return self._rewards(choices)

class ClassificationSimulation(Simulation[_S_out, _A_out]):
    """A simulation created from classifier data with features and labels.
    
    ClassificationSimulation turns labeled observations from a classification data set
    set, into interactions. For each interaction the feature set becomes the state and 
    all possible labels become the actions. Rewards for each interaction are created by 
    assigning a reward of 1 to the correct label (action) for a feature set (state) and 
    a value of 0 for all other labels (actions).

    Remark:
        This class when created from a data set will load all data into memory. Be careful when 
        doing this if you are working with a large dataset. To reduce memory usage you can provide
        meta information upfront that will allow features to be correctly encoded while the
        dataset is being streamed instead of waiting until the end of the data to train an encoder.
    """

    @staticmethod
    def from_json(json_val:Union[str, Dict[str,Any]]) -> 'ClassificationSimulation[State,Action]':
        """Construct a ClassificationSimulation object from JSON.

        Args:
            json_val: Either a json string or the decoded json object.

        Returns:
            The ClassificationSimulation representation of the given JSON string or object.
        """

        config = json.loads(json_val) if isinstance(json_val,str) else json_val

        has_header  : bool                        = True
        default_meta: Metadata[bool,bool,Encoder] = Metadata.default()
        defined_meta: Dict[Any, Metadata]         = {}

        if config["format"] == "openml":
            return ClassificationSimulation.from_openml(config["id"])

        if config["format"] == "csv":
            
            location    : str           = config["location"]
            md5_checksum: Optional[str] = None

            if "md5_checksum" in config:
                md5_checksum = config["md5_checksum"]

            if "has_header" in config:
                has_header = config["has_header"]

            if "column_default" in config:
                default_meta = cast(Metadata[bool,bool,Encoder], Metadata.from_json(config["column_default"]))

            if "column_overrides" in config:
                for key,value in config["column_overrides"].items():
                    defined_meta[key] = Metadata.from_json(value)

            return ClassificationSimulation.from_csv(
                location     = location,
                md5_checksum = md5_checksum,
                has_header   = has_header,
                default_meta = default_meta,
                defined_meta = defined_meta
            )

        if config["format"] == "table":

            table: Iterable[Sequence[str]] = config["table"]

            if "has_header" in config:
                has_header = config["has_header"]

            if "column_default" in config:
                default_meta = cast(Metadata[bool,bool,Encoder], Metadata.from_json(config["column_default"]))

            if "column_overrides" in config:
                for key,value in config["column_overrides"].items():
                    defined_meta[key] = Metadata.from_json(value)

            return ClassificationSimulation.from_table(
                table        = table,
                has_header   = has_header,
                default_meta = default_meta,
                defined_meta = defined_meta
            )

        raise Exception("We were unable to recognize the provided data format.")

    @staticmethod
    def from_openml(data_id:int) -> 'ClassificationSimulation[State,Action]':
        """Create a ClassificationSimulation from a given openml dataset id.

        Args:
            data_id: The unique identifier for a dataset stored on openml.
        """

        print(f"loading openml {data_id}...")

        start = time.time()
        with closing(urllib.request.urlopen(f'https://www.openml.org/api/v1/json/data/{data_id}')) as resp:
            data = json.loads(resp.read())["data_set_description"]

        with closing(urllib.request.urlopen(f'http://www.openml.org/api/v1/json/data/features/{data_id}')) as resp:
            meta = json.loads(resp.read())["data_features"]["feature"]
        print(time.time()-start)

        defined_meta: Dict[str,Metadata] = {}

        for m in meta:

            if m['data_type'] == 'numeric':
                encoder = cast(Encoder, NumericEncoder())
            else:
                encoder = cast(Encoder, OneHotEncoder(m['nominal_value']))

            defined_meta[m["name"]] = Metadata(
                ignore  = m["is_ignore"] == "true" or m["is_row_identifier"] == "true",
                label   = m["is_target"] == "true",
                encoder = encoder
            )

        csv_url = f"http://www.openml.org/data/v1/get_csv/{data['file_id']}"

        return ClassificationSimulation.from_csv(csv_url, defined_meta=defined_meta)

    @staticmethod
    def from_csv(
        location    : str,
        label_col   : Union[None,str,int] = None,
        md5_checksum: Optional[str] = None,
        csv_reader  : Callable[[Iterable[str]], Iterable[Sequence[str]]] = csv.reader,
        has_header  : bool = True,
        default_meta: Metadata[bool,bool,Encoder] = Metadata.default(),
        defined_meta: Dict[Any,Metadata] = {}) -> 'ClassificationSimulation[State,Action]':
        """Create a ClassificationSimulation given the location of a csv formatted dataset.

        Args:
            location: The location of the csv formatted dataset.
            label_col: The name of the column in the csv file that represents the label.
            md5_checksum: The expected md5 checksum of the csv dataset to ensure data integrity.
            csv_reader: A method to parse file lines at csv_path into their string values.
            has_header: Indicates if the csv file has a header row.
            default_meta: The default meta values for all columns unless explictly overridden with column_metas.
            column_metas: Keys are column name or index, values are meta objects that override the default values.
        """
        
        start = time.time()
        is_disk =  not location.lower().startswith('http')
        is_http =      location.lower().startswith('http')

        stream_manager: ContextManager[IO[bytes]]

        if is_disk:
            stream_manager = open(location, 'rb', encoding='utf-8')
        else:
            http_request = urllib.request.Request(location, headers={'Accept-encoding':'gzip'})
            stream_manager = closing(urllib.request.urlopen(http_request))
        print(time.time()-start)

        with stream_manager as raw_stream:

            is_disk_gzip = is_disk and location.lower().endswith(".gz")
            is_http_gzip = is_http and cast(HTTPResponse, raw_stream).info().get('Content-Encoding') == "gzip"

            start = time.time()
            all_bytes = decompress(raw_stream.read()) if is_disk_gzip or is_http_gzip else raw_stream.read()
            print(time.time()-start)

        if md5_checksum is not None and md5_checksum != md5(all_bytes).hexdigest():
            warn(
                "The dataset did not match the expected checksum. This could be the result of network "
                "errors or the file becoming corrupted. Please consider downloading the file again and if "
                "the error persists you may want to manually download and reference the file."
            )
        
        start = time.time()
        all_text  = all_bytes.decode('utf-8')
        all_lines = all_text.split("\n")
        csv_rows  = csv_reader(all_lines)
        print(time.time()-start)

        simulation = ClassificationSimulation.from_table(csv_rows, label_col, has_header, default_meta, defined_meta)

        return simulation

    @staticmethod
    def from_table(
        table       : Iterable[Sequence[str]],
        label_col   : Union[None,str,int] = None,
        has_header  : bool = True,
        default_meta: Metadata[bool,bool,Encoder] = Metadata.default(),
        defined_meta: Dict[Any,Metadata] = {}) -> 'ClassificationSimulation[State,Action]':
        """Create a ClassifierSimulation from the rows contained in a csv formatted dataset.

        Args:
            table: Any iterable of rows (i.e., sequence of str) with each row containing features/labels.
            label_col: Either the column index or the header name for the label column.
            has_header: Indicates if the first row in the table contains column names
            default_meta: The default meta values for all columns unless explictly overridden with column_metas.
            column_metas: Keys are column name or index, values are meta objects that override the default values.
        """

        # In theory we don't have to load the whole file up front. However, in practice,
        # not loading the file upfront is hard due to the fact that Python can't really
        # guarantee a generator will close a file.
        # For more info see https://stackoverflow.com/q/29040534/1066291
        # For more info see https://www.python.org/dev/peps/pep-0533/

        DEFINITE_META  = Metadata[bool,bool,Encoder[Hashable]]
        COLUMN_ENTRIES = List[Union[str,Sequence[Hashable]]]

        itable = filter(None, iter(table)) #filter out empty rows
        
        first  = next(itable)
        n_col  = len(first)
        itable = chain([first], itable)

        header: Sequence[str] = next(itable) if has_header else []

        columns  : List[COLUMN_ENTRIES] = [ []           for _ in range(n_col) ]
        metas    : List[DEFINITE_META ] = [ default_meta for _ in range(n_col) ]
 
        label_index = header.index(label_col) if label_col in header else label_col if isinstance(label_col,int) else None  # type: ignore
        label_meta  = defined_meta.get(label_col, defined_meta.get(label_index, None)) #type: ignore

        if isinstance(label_col, str) and label_col not in header:
            raise Exception("We were unable to find the label column in the header row (or there was no header row).")

        if any(map(lambda key: isinstance(key,str) and key not in header, defined_meta)):
            raise Exception("We were unable to find a meta column in the header row (or there was no header row).")

        if label_meta is not None and label_meta.label == False:
            raise Exception("A meta entry was provided for the label column that was explicitly marked as non-label.")

        def to_column_index(key: Union[int,str]):
            return header.index(key) if isinstance(key,str) else key

        defined_meta = { to_column_index(key):defined_meta[key] for key in defined_meta }
        for key,meta in defined_meta.items(): metas[key] = metas[key].override(meta)

        if label_index is not None:
            metas[label_index] = metas[label_index].override(Metadata(None,True,None))

        start = time.time()
        #first pass, loop through all rows. If meta is marked as ignore place an empty
        # tuple in the column array, if meta has an encoder already fit encode now, if
        #the encoder isn't fit place the string value in the column for later fitting.
        for row in itable:
            for r,col in zip(row, columns):
                col.append(r)
        print(time.time() - start)

        feature_encodings: List[Sequence[Sequence[Hashable]] ] = []
        label_encodings  : List[Sequence[Sequence[Hashable]] ] = [] 

        start = time.time()
        #second pass, loop through all columns. Now that we have the data in column arrays
        #we are able to fit any encoders that need fitting. After fitting we need to encode
        #these column's string values and turn our data back into rows for features and labels.
        for col, m in zip(columns, metas):
            if m.ignore: continue

            encoder = m.encoder if m.encoder.is_fit else m.encoder.fit(col)
            values  = encoder.encode(col)
            dest    = label_encodings if m.label else feature_encodings

            if not isinstance(encoder, OneHotEncoder):
                dest.append([ (v,) for v in values ])
            else:
                dest.append(cast(Sequence[Tuple[int,...]],values))

        features = list(map(tuple,map(chain.from_iterable, zip(*feature_encodings)))) #type: ignore
        labels   = list(map(tuple,map(chain.from_iterable, zip(*label_encodings)))) #type: ignore
        print(time.time() - start)

        start = time.time()
        states  = [ f if len(f) > 1 else f[0] for f in features ]
        actions = [ l if len(l) > 1 else l[0] for l in labels   ]
        print(time.time()-start)

        simulation = ClassificationSimulation(states, actions)

        return simulation

    def __init__(self, features: Sequence[_S_out], labels: Sequence[_A_out]) -> None:
        """Instantiate a ClassificationSimulation.

        Args:
            features: The collection of features used for the original classifier problem.
            labels: The collection of labels assigned to each observation of features.
        """

        assert len(features) == len(labels), "Mismatched lengths of features and labels"

        #reward_encoder = OneHotEncoder(labels).encode(labels)

        action_set = tuple(set(labels))

        states  = features
        actions = list(repeat(action_set, len(states)))
        
        start = time.time()
        rewards = OneHotEncoder(action_set,False,True).encode(labels)
        print(time.time() - start)

        self._simulation = MemorySimulation(states, actions, rewards)        

    @property
    def interactions(self) -> Sequence[KeyedInteraction[_S_out, _A_out]]:
        """The interactions in this simulation.

        Remarks:
            See the Simulation base class for more information.
        """
        return self._simulation.interactions

    def rewards(self, choices: Sequence[Choice]) -> Sequence[Tuple[_S_out, _A_out, Reward]]:
        """The observed rewards for interactions (identified by its key) and their selected action indexes.

        Remarks:
            See the Simulation base class for more information.        
        """

        return self._simulation.rewards(choices)