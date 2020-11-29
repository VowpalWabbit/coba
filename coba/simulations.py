"""The simulations module contains core classes and types for defining contextual bandit simulations.

This module contains the abstract interface expected for bandit simulations along with the 
class defining an Interaction within a bandit simulation. Additionally, this module also contains 
the type hints for Context, Action and Reward. These type hints don't contain any functionality. 
Rather, they simply make it possible to use static type checking for any project that desires 
to do so.

TODO Add RegressionSimulation
"""

import gc
import csv
import json
import collections

from itertools import compress, repeat, count, chain
from abc import ABC, abstractmethod
from typing import (
    Optional, Iterable, Sequence, List, Union, Callable, 
    TypeVar, Generic, Hashable, Dict, Any, Tuple
)

import coba.random

from coba.data.sources import Source, HttpSource, DiskSource, OpenmlSource
from coba.data.definitions import FullMeta, PartMeta
from coba.data.encoders import OneHotEncoder
from coba.execution import ExecutionContext

Context = Optional[Hashable]
Action  = Hashable
Reward  = float
Key     = int 
Choice  = int

_C_out = TypeVar('_C_out', bound=Context, covariant=True)
_A_out = TypeVar('_A_out', bound=Action, covariant=True)

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

class Simulation(Generic[_C_out, _A_out], ABC):
    """The simulation interface."""

    @property
    @abstractmethod
    def interactions(self) -> Sequence[Interaction[_C_out, _A_out]]:
        """The sequence of interactions in a simulation.

        Remarks:
            All Benchmark assume that interactions is re-iterable. So long as interactions is 
            a Sequence it will always be re-iterable. If interactions was merely Iterable then 
            it would be possible for it to only allow enumeration one time.
        """
        ...

    @abstractmethod
    def rewards(self, choices: Sequence[Tuple[Key,Choice]] ) -> Sequence[Reward]:
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
        contexts   : Sequence[_C_out], 
        action_sets: Sequence[Sequence[_A_out]], 
        reward_sets: Sequence[Sequence[Reward]]) -> None:
        """Instantiate a MemorySimulation.

        Args:
            contexts: A collection of contexts to turn into a simulation.
            action_sets: A collection of action sets to turn into a simulation
            reward_sets: A collection of reward sets to turn into a simulation 
        """

        assert len(contexts) == len(action_sets) == len(reward_sets), "Mismatched lengths of contexts, actions and rewards"

        self._interactions = list(map(Interaction, contexts, action_sets, count()))

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

    def rewards(self, choices: Sequence[Tuple[Key,Choice]]) -> Sequence[Reward]:
        """The observed rewards for interactions (identified by its key) and their selected action indexes.

        Remarks:
            See the Simulation base class for more information.
        """

        return [ self._rewards[choice] for choice in choices]

class LazySimulation(Simulation[_C_out, _A_out]):

    def __enter__(self) -> 'LazySimulation':
        """Load the simulation into memory. If already loaded do nothing."""

        self._simulation = self.load_simulation()

        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """Unload the simulation from memory."""

        if self._simulation is not None:
            self._simulation = None
            gc.collect() #in case the simulation is large

    @property
    def interactions(self) -> Sequence[Interaction[Context,Action]]:
        """The interactions in this simulation.

        Remarks:
            See the Simulation base class for more information.
        """

        if self._simulation is not None:
            return self._simulation.interactions
        
        raise Exception("A LazySimulation must be loaded before it can be used.")

    def rewards(self, choices: Sequence[Tuple[Key,Choice]]) -> Sequence[Reward]:
        """The observed rewards for interactions (identified by its key) and their selected action indexes.

        Remarks:
            See the Simulation base class for more information.
        """
        
        if self._simulation is not None:
            return self._simulation.rewards(choices)

        raise Exception("A LazySimulation must be loaded before it can be used.")

    @abstractmethod
    def load_simulation(self) -> Simulation[_C_out, _A_out]: ...

class LambdaSimulation(MemorySimulation[_C_out, _A_out]):
    """A Simulation created from lambda functions that generate contexts, actions and rewards.

    Remarks:
        This implementation is useful for creating simulations from defined distributions.
    """

    def __init__(self,
                 n_interactions: int,
                 context   : Callable[[int],_C_out],
                 action_set: Callable[[int],Sequence[_A_out]], 
                 reward    : Callable[[_C_out,_A_out],Reward],
                 seed: int = None) -> None:
        """Instantiate a LambdaSimulation.

        Args:
            n_interactions: How many interactions the LambdaSimulation should have.
            context: A function that should return a context given an index in `range(n_interactions)`.
            action_set: A function that should return all valid actions for a given context.
            reward: A function that should return the reward for a context and action.
        """

        coba.random.seed(seed)

        contexts   : List[_C_out]           = []
        action_sets: List[Sequence[_A_out]] = []
        reward_sets: List[Sequence[Reward]] = []

        for i in range(n_interactions):
            _context    = context(i)
            _action_set = action_set(i)
            _reward_set = [reward(_context, _action) for _action in _action_set]

            contexts   .append(_context)
            action_sets.append(_action_set)
            reward_sets.append(_reward_set)

        super().__init__(contexts, action_sets, reward_sets)

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

    @staticmethod
    def from_json(json_val:Union[str, Dict[str,Any]]) -> 'ClassificationSimulation[Context]':
        """Construct a ClassificationSimulation object from JSON.

        Args:
            json_val: Either a json string or the decoded json object.

        Returns:
            The ClassificationSimulation representation of the given JSON string or object.
        """

        config = json.loads(json_val) if isinstance(json_val,str) else json_val

        has_header  : bool                = True
        default_meta: FullMeta            = FullMeta()
        defined_meta: Dict[Any, PartMeta] = {}

        if config["format"] == "openml":
            with ExecutionContext.Logger.log(f"loading openml {config['id']}..."):
                return ClassificationSimulation.from_source(OpenmlSource(config["id"], config.get("md5_checksum", None)))

        if config["format"] == "csv":
            location    : str           = config["location"]
            md5_checksum: Optional[str] = None

            if "md5_checksum" in config:
                md5_checksum = config["md5_checksum"]

            if "has_header" in config:
                has_header = config["has_header"]

            if "column_default" in config:
                default_meta =  FullMeta.from_json(config["column_default"])

            if "column_overrides" in config:
                for key,value in config["column_overrides"].items():
                    defined_meta[key] = PartMeta.from_json(value)

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
                default_meta = FullMeta.from_json(config["column_default"])

            if "column_overrides" in config:
                for key,value in config["column_overrides"].items():
                    defined_meta[key] = PartMeta.from_json(value)

            return ClassificationSimulation.from_table(
                table        = table,
                has_header   = has_header,
                default_meta = default_meta,
                defined_meta = defined_meta
            )

        raise Exception("We were unable to recognize the provided data format.")

    @staticmethod
    def from_source(source: Source[Tuple[Sequence[Sequence[Any]], Sequence[Any]]]) -> 'ClassificationSimulation[Context]':
        
        features, actions = source.read()

        if isinstance(source, OpenmlSource) and len(actions[0]) == 1:
            raise Exception("This does not appear to be a classification dataset. Creating a ClassificationSimulation from it will perform poorly.")

        return ClassificationSimulation(features, actions)

    @staticmethod
    def from_csv(
        location    : str,
        label_col   : Union[None,str,int] = None,
        md5_checksum: Optional[str] = None,
        csv_reader  : Callable[[Iterable[str]], Iterable[Sequence[str]]] = csv.reader, #type: ignore #pylance complains
        has_header  : bool = True,
        default_meta: FullMeta = FullMeta(),
        defined_meta: Dict[Any,PartMeta] = {}) -> 'ClassificationSimulation[Context]':
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

        source: Source[Iterable[str]]

        if not location.lower().startswith('http'):
            source = DiskSource(location)
        else: 
            source = HttpSource(location, ".csv", md5_checksum, 'data')

        csv_rows = list(csv_reader(source.read()))

        with ExecutionContext.Logger.log('encoding data... '):
            return ClassificationSimulation.from_table(csv_rows, label_col, has_header, default_meta, defined_meta)

    @staticmethod
    def from_table(
        table       : Iterable[Sequence[str]],
        label_col   : Union[None,str,int] = None,
        has_header  : bool = True,
        default_meta: FullMeta = FullMeta(),
        defined_meta: Dict[Any, PartMeta] = {}) -> 'ClassificationSimulation[Context]':
        """Create a ClassifierSimulation from the rows contained in a csv formatted dataset.

        Args:
            table: Any iterable of rows (i.e., sequence of str) with each row containing features/labels.
            label_col: Either the column index or the header name for the label column.
            has_header: Indicates if the first row in the table contains column names
            default_meta: The default meta values for all columns unless explictly overridden with column_metas.
            defined_meta: Keys are column name or index, values are meta objects that override the default values.
        """

        # In theory we don't have to load the whole file up front. However, in practice,
        # not loading the file upfront is hard due to the fact that Python can't really
        # guarantee a generator will close a file.
        # For more info see https://stackoverflow.com/q/29040534/1066291
        # For more info see https://www.python.org/dev/peps/pep-0533/

        itable = filter(None, iter(table)) #filter out empty rows

        #get first row to determine number of columns and
        #then put the first row back for later processing
        first  = next(itable)
        n_col  = len(first)
        itable = chain([first], itable)
 
        header: Sequence[str] = next(itable) if has_header else []

        label_index = header.index(label_col) if label_col in header else label_col if isinstance(label_col,int) else None  # type: ignore

        if isinstance(label_col, str) and label_col not in header:
            raise Exception("We were unable to find the label column in the header row (or there was no header row).")

        if any(map(lambda key: isinstance(key,str) and key not in header, defined_meta)):
            raise Exception("We were unable to find a meta column in the header row (or there was no header row).")

        def index(key: Union[int,str]):
            return header.index(key) if isinstance(key,str) else key

        over_metas = collections.defaultdict(PartMeta, { index(key):val for key,val in defined_meta.items() } )
        metas      = [ default_meta.override(over_metas[i]) for i in range(n_col) ]

        if label_index is not None:
            metas[label_index] = metas[label_index].override(PartMeta(label=True))

        #after extensive testing I found that performing many loops with simple logic
        #was about 3 times faster than performing one or two loops with complex logic
        
        #extract necessary meta data one time
        is_not_ignores = [ not m.ignore for m in metas ]

        #transform rows into columns
        columns = list(zip(*itable))
        
        #remove ignored columns
        metas   = list(compress(metas, is_not_ignores))
        columns = list(compress(columns, is_not_ignores))

        #create encoding groups according to column type
        label_encodings  : List[Sequence[Hashable]] = []
        feature_encodings: List[Sequence[Hashable]] = []

        #encode columns and place in appropriate group
        for col, m in zip(columns, metas):

            encoding = label_encodings if m.label else feature_encodings
            encoder  = m.encoder if m.encoder.is_fit else m.encoder.fit(col)

            if isinstance(encoder, OneHotEncoder):
                encoding.extend(list(zip(*encoder.encode(col))))
            else:
                encoding.append(encoder.encode(col))

        #transform columns back into rows
        features = list(zip(*feature_encodings)) #type: ignore
        labels   = list(zip(*label_encodings))   #type: ignore

        #turn singular tuples into their values
        contexts  = [ f if len(f) > 1 else f[0] for f in features ]
        actions = [ l if len(l) > 1 else l[0] for l in labels   ]

        return ClassificationSimulation(contexts, actions)

    def __init__(self, features: Sequence[_C_out], labels: Sequence[Action]) -> None:
        """Instantiate a ClassificationSimulation.

        Args:
            features: The collection of features used for the original classifier problem.
            labels: The collection of labels assigned to each observation of features.
        """

        assert len(features) == len(labels), "Mismatched lengths of features and labels"

        action_set = list(set(labels))

        contexts = features
        actions  = list(repeat(OneHotEncoder(action_set).encode(action_set), len(contexts)))
        rewards  = OneHotEncoder(action_set).encode(labels)

        self._action_set = action_set
        super().__init__(contexts, actions, rewards)

class OpenmlSimulation(LazySimulation[_C_out, Tuple[int,...]]):
    """A simulation created from openml data with features and labels.

    OpenmlSimulation turns labeled observations from a classification data set
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

    def __init__(self, data_id: int, md5_checksum: str = None) -> None:
        self._openml_source = OpenmlSource(data_id, md5_checksum)

    def load_simulation(self) -> Simulation[_C_out, Tuple[int,...]]:
        return ClassificationSimulation.from_source(self._openml_source)

class JsonSimulation(LazySimulation[Context, Action]):
    """A Simulation implementation which supports loading and unloading from json representations.""" 
    
    def __init__(self, json_val) -> None:
        """Instantiate a JsonSimulation

        Args:
            json: A json representation that can be turned into a simulation when needed.
        """

        self._json_obj = json.loads(json_val) if isinstance(json_val,str) else json_val

    def load_simulation(self) -> Simulation[Context, Action]:
        """Load the simulation into memory. If already loaded do nothing."""

        if self._json_obj["type"] == "classification":
            return ClassificationSimulation.from_json(self._json_obj["from"])
        else:
            raise Exception("We were unable to recognize the provided simulation type")