"""The simulations module contains core classes and types for defining bandit simulations.

This module contains the abstract interface expected for bandit simulations along with the 
class defining a Round within a bandit simulation. Additionally, this module also contains 
the type hints for State, Action and Reward. These type hints don't contain any functionality. 
Rather, they simply make it possible to use static type checking for any project that desires 
to do so.

Todo:
    * Add RegressionSimulation
"""

import csv
import itertools
import urllib.request
import json
import hashlib

from warnings import warn
from gzip import GzipFile
from contextlib import closing
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import DefaultDict, Optional, Iterator, Iterable, Sequence, List, Union, Callable, TextIO, TypeVar, Generic, Hashable, Dict, Any, cast

import bbench.random
from bbench.preprocessing import DefiniteMeta, OneHotEncoder, PartialMeta, InferredEncoder, NumericEncoder

#state, action, reward types
State  = Optional[Hashable]
Action = Hashable
Reward = float

ST_out = TypeVar('ST_out', bound=State, covariant=True)
AT_out = TypeVar('AT_out', bound=Action, covariant=True)

class Round(Generic[ST_out,AT_out]):
    """A class to contain all data needed to represent a round in a bandit simulation."""

    def __init__(self, 
                 state  : ST_out,
                 actions: Sequence[AT_out],
                 rewards: Sequence[Reward]) -> None:
        """Instantiate Round.

        Args:
            state: Features describing the round's state. Will be `None` for multi-armed bandit simulations.
            actions: Features describing available actions for the given state.
            rewards: The reward that would be received for taking each of the given actions.

        Remarks:
            It is assumed that the following is alwyas true len(actions) == len(rewards).
        """

        assert len(actions) > 0, "At least one action must be provided for the round"
        assert len(actions) == len(rewards), "Mismatched lengths of actions and rewards"

        self._state   = state
        self._actions = actions
        self._rewards = rewards

    @property
    def state(self) -> ST_out:
        """Read-only property providing the round's state."""
        return self._state

    @property
    def actions(self) -> Sequence[AT_out]:
        """Read-only property providing the round's possible actions."""
        return self._actions

    @property
    def rewards(self) -> Sequence[Reward]:
        """Read-only property providing the round's reward for each action."""
        return self._rewards

class Simulation(Generic[ST_out,AT_out], ABC):
    """The simulation interface."""

    @property
    @abstractmethod
    def rounds(self) -> Sequence[Round[ST_out,AT_out]]:
        """A read-only property providing the sequence of rounds in a simulation.

        Returns:
            The simulation's sequence of rounds.

        Remarks:
            All Benchmark assume that rounds is re-iterable. So long as rounds is a
            Sequence[Round] it will always be re-iterable. If rounds were merely 
            Iterable[Round] then it is possible for it to only allow enumeration once.
        """
        ...

class LambdaSimulation(Simulation[ST_out,AT_out]):
    """A Simulation created from lambda functions that generate states, actions and rewards.
    
    Remarks:
        This implementation is useful for creating simulations from defined distributions.
    """

    def __init__(self,
                 n_rounds: int,
                 S: Callable[[int],ST_out],
                 A: Callable[[ST_out],Sequence[AT_out]], 
                 R: Callable[[ST_out,AT_out],Reward]) -> None:
        """Instantiate a LambdaSimulation.

        Args:
            n_rounds: how many rounds the LambdaSimulation should have.
            S: A lambda function that should return a state given an index in `range(n_rounds)`.
            A: A lambda function that should return all valid actions for a given state.
            R: A lambda function that should return the reward for a state and action.
        """

        self._S = S
        self._A = A
        self._R = R

        self._rounds = list(itertools.islice(self._round_generator(), n_rounds))

    @property
    def rounds(self) -> Sequence[Round[ST_out,AT_out]]:
        """The rounds in this simulation.
        
        Remarks:
            See the Simulation base class for more information.
        """
        return self._rounds

    def _round_generator(self) -> Iterator[Round[ST_out,AT_out]]:
        """Generate rounds for this simulation."""

        S = self._S
        A = self._A
        R = self._R

        for i in itertools.count():
            state   = S(i)
            actions = A(state)
            rewards = [R(state,action) for action in actions]

            yield Round(state,actions,rewards)

class MemorySimulation(Simulation[ST_out,AT_out]):
    """A Simulation implementation created from in memory sequences of Rounds.
    
    Remarks:
        This implementation is very useful for unit-testing known edge cases.
    """

    def __init__(self, rounds: Sequence[Round[ST_out,AT_out]]) -> None:
        """Instantiate a MemorySimulation.

        Args:
            rounds: a collection of rounds to turn into a simulation.
        """
        self._rounds = rounds

    @property
    def rounds(self) -> Sequence[Round[ST_out,AT_out]]:
        """The rounds in this simulation.
        
        Remarks:
            See the Simulation base class for more information.
        """
        return self._rounds

class ShuffleSimulation(Simulation[ST_out,AT_out]):
    """A simulation which created from an existing simulation by shuffling rounds.
    
    Remarks:
        Shuffling is applied one time upon creation and after that round order is fixed.
        Shuffling also does not change the original simulation's round order or copy the
        original rounds in memory. Shuffling is guaranteed to be deterministic according
        to seed regardless of the local Python execution environment.
    """
    
    def __init__(self, simulation: Simulation[ST_out,AT_out], seed: Optional[int] = None):
        """Instantiate a ShuffleSimulation
        
        Args:
            simulation: The simulation we which to shuffle round order for.
            seed: The seed we wish to use in determining the shuffle order.
        """

        bbench.random.seed(seed)
        self._rounds = bbench.random.shuffle(simulation.rounds)
    
    @property
    def rounds(self) -> Sequence[Round[ST_out,AT_out]]:
        """The rounds in this simulation.
        
        Remarks:
            See the Simulation base class for more information.
        """

        return self._rounds

class ClassificationSimulation(Simulation[State,Action]):
    """A simulation created from classifier data with features and labels.
    
    ClassificationSimulation turns labeled observations from a classification data set
    set, into rounds. For each round the feature set becomes the state and all possible 
    labels become the actions. Rewards for each round are created by assigning a reward 
    of 1 to the correct label (action) for a feature set (state) and a value of 0 for 
    all other labels (actions).

    Remark:
        This method loads all classification data into memory. Be careful when doing
        this if you are working with a large file. To reduce memory usage you can provide
        meta information upfront that will allow features to be correctly encoded as a
        data set is processed instead of waiting until the end of the data to train an encoder.
    """

    @staticmethod
    def from_openml(data_id:int) -> Simulation:
        """Create a ClassificationSimulation from a given openml dataset id.

        Args:
            data_id: The unique identifier for a dataset stored on openml.
        """

        with closing(urllib.request.urlopen(f'https://www.openml.org/api/v1/json/data/{data_id}')) as resp:
            data = json.loads(resp.read())["data_set_description"]

        with closing(urllib.request.urlopen(f'http://www.openml.org/api/v1/json/data/features/{data_id}')) as resp:
            meta = json.loads(resp.read())["data_features"]["feature"]

        column_metas: Dict[str,PartialMeta] = {}

        for m in meta:
            column_metas[m["name"]] = PartialMeta(
                ignore  = m["is_ignore"] == "true" or m["is_row_identifier"] == "true",
                label   = m["is_target"] == "true",
                encoder = NumericEncoder() if m["data_type"] == "numeric" else OneHotEncoder()
            )
        
        file_url = f"http://www.openml.org/data/v1/get_csv/{data['file_id']}"
        
        return ClassificationSimulation.from_csv_url(file_url, column_metas=column_metas)

    @staticmethod
    def from_csv_url(
        csv_url     : str,
        label_col   : Optional[Union[str,int]] = None,
        md5_checksum: Optional[str] = None,
        csv_reader  : Callable[[TextIO], Iterator[List[str]]] = csv.reader,
        has_header  : bool = True,
        default_meta: DefiniteMeta = DefiniteMeta(ignore=False,label=False,encoder=InferredEncoder()),
        column_metas: Union[Dict[int,PartialMeta],Dict[str,PartialMeta]] = cast(Dict[int,PartialMeta],{})) -> Simulation:
        """Create a ClassificationSimulation from the url to a csv formatted dataset.

        Args:
            csv_url: The url to a csv formatted dataset.
            label_col: The name of the column in the csv file that represents the label.
            md5_checksum: The expected md5 checksum of the csv dataset to ensure data integrity.
            csv_reader: A method to parse file lines at csv_path into their string values.
            has_header: Indicates if the csv file has a header row.
            default_meta: The default meta values for all columns unless explictly overridden with column_metas.
            column_metas: Keys are column name or index, values are meta objects that override the default values.
        """

        csv_request = urllib.request.Request(csv_url, headers={'Accept-encoding':'gzip'})
        with closing(urllib.request.urlopen(csv_request)) as resp:

            actual_md5_checksum  = hashlib.md5()
            is_resp_content_gzip = resp.info().get('Content-Encoding') == "gzip"
            resp_content_stream  = GzipFile(fileobj=resp) if is_resp_content_gzip else resp

            def decoded_lines_and_calc_checksum() -> Iterator[str]:
                for line in resp_content_stream:
                    actual_md5_checksum.update(line)
                    yield line.decode('utf-8')

            csv_rows = csv.reader(decoded_lines_and_calc_checksum())

            simulation = ClassificationSimulation.from_csv_rows(csv_rows, label_col, has_header, default_meta, column_metas)

        # At this time openML only provides md5_checksum for arff files. Because we are reading csv we can't check this.
        if md5_checksum is not None and md5_checksum != actual_md5_checksum.hexdigest():
            warn(
                "The OpenML dataset did not match the expected checksum. This could be the result of network"
                "errors or the file becoming corrupted. Please consider downloading the file again and if the"
                "error persists you may want to manually download and reference the file."
                )

        return simulation

    @staticmethod
    def from_csv_path(
        csv_path    : str,
        label_col   : Optional[Union[str,int]] = None,
        csv_reader  : Callable[[TextIO], Iterator[List[str]]] = csv.reader,
        has_header  : bool = True,
        default_meta: DefiniteMeta = DefiniteMeta(ignore=False,label=False,encoder=InferredEncoder()),
        column_metas: Union[Dict[int,PartialMeta],Dict[str,PartialMeta]] = cast(Dict[int,PartialMeta],{})) -> Simulation:
        """Create a ClassificationSimulation from the path to a csv formatted file.

        Args:
            csv_path: The path to the csv file.
            label_col: The name of the column in the csv file that represents the label.
            csv_reader: A method to parse file lines at csv_path into their string values.
            has_header: Indicates if the csv file has a header row.
            default_meta: The default meta values for all columns unless explictly overridden with column_metas.
            column_metas: Keys are column name or index, values are meta objects that override the default values.
        """

        with open(csv_path, newline='') as csv_file:
            return ClassificationSimulation.from_csv_rows(csv_reader(csv_file), label_col, has_header, default_meta, column_metas)

    @staticmethod
    def from_csv_rows(
        csv_rows    : Iterable[List[str]],
        label_col   : Optional[Union[str,int]] = None,
        has_header  : bool = True,
        default_meta: DefiniteMeta = DefiniteMeta(ignore=False,label=False,encoder=InferredEncoder()),
        column_metas: Union[Dict[int,PartialMeta],Dict[str,PartialMeta]] = cast(Dict[int,PartialMeta],{})) -> Simulation:
        """Create a ClassifierSimulation from the rows contained in a csv formatted dataset.

        Args:
            csv_rows: Any iterable of string values representing a row with features/label.
            label_col: Either the column index or the header name for the label column.
            has_header: Indicates if the first row in csv_rows contains column names
            default_meta: The default meta values for all columns unless explictly overridden with column_metas.
            column_metas: Keys are column name or index, values are meta objects that override the default values.
        """

        # In theory we don't have to load the whole file up front. However, in practice,
        # not loading the file upfront is hard due to the fact that Python can't really
        # guarantee a generator will close a file.
        # For more info see https://stackoverflow.com/q/29040534/1066291
        # For more info see https://www.python.org/dev/peps/pep-0533/

        T_COL_VAL = Union[str,Sequence[Hashable]]
        T_COL     = List[T_COL_VAL]

        csv_iter              = iter(csv_rows)
        header_row: List[str] = next(csv_iter) if has_header else []

        columns : Dict[int, T_COL       ]   = defaultdict(list)
        metas   : Dict[int, DefiniteMeta]   = defaultdict(lambda:default_meta)
        features: Dict[int, List[Hashable]] = defaultdict(list)
        labels  : Dict[int, List[Hashable]] = defaultdict(list)

        label_index = header_row.index(label_col) if label_col in header_row else label_col if isinstance(label_col,int) else None  # type: ignore
        label_meta  = column_metas.get(label_col, column_metas.get(label_index, None)) #type: ignore

        if isinstance(label_col, str) and label_col not in header_row:
            raise Exception("We were unable to find the label column in the header row (or there was no header row).")

        if any(map(lambda key: isinstance(key,str) and key not in header_row, column_metas)):
            raise Exception("We were unable to find a meta column in the header row (or there was no header row).")

        if label_meta is not None and label_meta.label == False:
            raise Exception("A meta entry was provided for the label column that was explicitly marked as non-label.")

        def to_column_index(key: Union[int,str]):
            return header_row.index(key) if isinstance(key,str) else key

        if label_index is not None and label_meta is None:
            metas[label_index] = metas[label_index].apply(PartialMeta(label=True))

        for key,meta in column_metas.items():
            metas[to_column_index(key)] = metas[to_column_index(key)].apply(meta)

        #first pass, loop through all rows. If meta is marked as ignore place an empty
        # tuple in the column array, if meta has an encoder already fit encode now, if
        #the encoder isn't fit place the string value in the column for later fitting.
        for row in (r for r in csv_iter if len(r) > 0):
            for r,col,m in [ (row[i], columns[i], metas[i]) for i in range(len(row)) ]:
                col.append(() if m.ignore else m.encoder.encode(r) if m.encoder.is_fit else r)

        #second pass, loop through all columns. Now that we have the data in column arrays
        #we are able to fit any encoders that need fitting. After fitting we need to encode
        #these column's string values and turn our data back into rows for features and labels.
        for col,m in [ (columns[i], metas[i]) for i in range(len(columns)) if not metas[i].ignore ]:

            #if the encoder isn't already fit we know that col is a List[str]
            encoder = None if m.encoder.is_fit else m.encoder.fit(cast(List[str],col))

            for c,f,l in [ (col[i], features[i], labels[i]) for i in range(len(col)) ]:

                final_value = c if encoder is None else encoder.encode(cast(str,c))

                if m.label:
                    l.extend(final_value)
                else:
                    f.extend(final_value)

        finalize = lambda x: x[0] if len(x) == 1 else tuple(x)

        states  = [ finalize(features[i]) for i in range(len(features)) ]
        actions = [ finalize(labels  [i]) for i in range(len(labels  )) ]

        return ClassificationSimulation(states, actions)

    def __init__(self, features: Sequence[State], labels: Sequence[Action]) -> None:
        """Instantiate a ClassificationSimulation.

        Args:
            features: The collection of features used for the original classifier problem.
            labels: The collection of labels assigned to each observation of features.
        """

        assert len(features) == len(labels), "Mismatched lengths of features and labels"

        states      = features
        action_set  = tuple(set(labels))
        reward_sets = [ [int(label==action) for action in action_set] for label in labels ]

        self._rounds = list(map(Round[State, Action], states, itertools.repeat(action_set), reward_sets))

    @property
    def rounds(self) -> Sequence[Round[State,Action]]:
        """The rounds in this simulation.
        
        Remarks:
            See the Simulation base class for more information.
        """
        return self._rounds
