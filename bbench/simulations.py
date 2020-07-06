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

from gzip import GzipFile
from contextlib import closing
from abc import ABC, abstractmethod
from typing import Optional, Iterator, Iterable, Sequence, List, Union, Callable, TextIO, TypeVar, Generic, Hashable, Dict, Any

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
    def __init__(self, smiulation: Simulation[ST_out,AT_out], seed: Optional[int] = None):

        bbench.random.seed(seed)

        self._rounds = bbench.random.shuffle(list(smiulation.rounds))
    
    @property
    def rounds(self) -> Sequence[Round[ST_out,AT_out]]:

        return self._rounds

class ClassificationSimulation(Simulation[State,Action]):
    """A simulation created from classifier data with features and labels.
    
    Remark:
        ClassificationSimulation creation turns each feature set and label, into a round. 
        In each round the feature set becomes the state and all possible labels become the
        actions. Rewards for each round are created by assigning a reward of 1 to the correct 
        label (action) for a feature set (state) and a value of 0 for all other labels (actions).
    """

    @staticmethod
    def from_openml(data_id:int) -> Simulation:

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
        file_req = urllib.request.Request(file_url, headers={'Accept-encoding':'gzip'})

        with closing(urllib.request.urlopen(file_req)) as resp:

            # actual_md5_checksum = hashlib.md5()

            if resp.info().get('Content-Encoding') == "gzip":
                resp_stream = GzipFile(fileobj=resp)
            else:
                resp_stream = resp

            def decoded_lines() -> Iterator[str]:
                for line in resp_stream:
                    # actual_md5_checksum.update(line)
                    yield line.decode('utf-8')

            simulation = ClassificationSimulation.from_csv_rows(csv.reader(decoded_lines()), column_metas=column_metas)

        # # At this time openML only provides md5_checksum for arff files. Because we are reading csv we can't check this.
        # if actual_md5_checksum.hexdigest() != data["md5_checksum"]:
        #     warn(
        #         "The OpenML dataset did not match the expected checksum. This could be the result of network"
        #         "errors or the file becoming corrupted. Please consider downloading the file again and if the"
        #         "error persists you may want to manually download and reference the file."
        #         )
        
        return simulation

    @staticmethod
    def from_csv_path(
        csv_path    : str,
        label_col   : Union[str,int],
        csv_reader  : Callable[[TextIO], Iterator[List[str]]] = csv.reader,
        has_header  : bool = True,
        default_meta: DefiniteMeta = DefiniteMeta(ignore=False,label=False,encoder=InferredEncoder()),
        column_metas: Union[Dict[int,PartialMeta],Dict[str,PartialMeta]] = {}) -> Simulation:
        """Create a ClassificationSimulation from a csv file with a header row.

        Args:
            csv_path: The path to the csv file.
            label_col: The name of the column in the csv file that represents the label.
            csv_reader: A method to parse file lines at csv_path into their string values.
            has_header: Indicates if the csv file has a header row.
            default_meta: The default meta values for all columns unless explictly overridden with column_metas.
            column_metas: Keys are column name or index, values are meta objects that override the default values.

        Remarks:
            This method will open the file and read it all into memory. Be careful when doing
            this if you are working with a large file. One way to improve on this is to make
            sure column are correctly typed and all string columns are represented as integer
            backed categoricals (aka, `factors` in R).
        """

        with open(csv_path, newline='') as csv_file:
            return ClassificationSimulation.from_csv_rows(csv_reader(csv_file), label_col, has_header, default_meta, column_metas)
    
    @staticmethod
    def from_csv_rows(
        csv_rows    : Iterable[List[str]],
        label_col   : Optional[Union[str,int]] = None,
        has_header  : bool = True,
        default_meta: DefiniteMeta = DefiniteMeta(ignore=False,label=False,encoder=InferredEncoder()),
        column_metas: Union[Dict[int,PartialMeta],Dict[str,PartialMeta]] = {}) -> Simulation:

        """Create a ClassifierSimulation from the string values of a csv file.

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

        csv_iter                         = iter(csv_rows)
        csv_cols  : List[List[Any]]      = []
        header_row: List[str]            = next(csv_iter) if has_header else []

        label_index = header_row.index(label_col) if label_col in header_row else label_col if isinstance(label_col,int) else None  # type: ignore
        label_meta  = column_metas.get(label_col, column_metas.get(label_index, None)) #type: ignore

        if isinstance(label_col, str) and label_col not in header_row:
            raise Exception("We were unable to find the label column in the header row (or there was no header row).")
        
        if any(map(lambda key: isinstance(key,str) and key not in header_row, column_metas)):
            raise Exception("We were unable to find a meta column in the header row (or there was no header row).")

        if label_meta is not None and label_meta.label == False:
            raise Exception("A meta entry was provided for the label column that which was explicitly marked as non-label.")

        index_metas: Dict[int,DefiniteMeta] = dict()

        if label_index is not None and label_meta is None:
            index_metas[label_index] = default_meta.with_overrides(PartialMeta(label=True))

        if column_metas is not None:
            for key,meta in column_metas.items():

                index = header_row.index(key) if isinstance(key ,str) else key

                if index in index_metas:
                    raise Exception("Two separate encodings were provided for the same column in a csv stream.")
                else:
                    index_metas[index] = default_meta.with_overrides(meta)

        row_count = 0

        for row in csv_iter:
            
            if len(row) == 0: continue #ignore empty lines
            
            row_count += 1

            if len(csv_cols) == 0: #first non-empty row after optional header
                csv_cols = [ [] for _ in range(len(row)) ]

                for i in range(len(row)):
                    if i not in index_metas:
                        index_metas[i] = default_meta.with_overrides(None)

            for i,val in enumerate(row):
                if(index_metas[i].ignore):
                    continue
                elif(index_metas[i].encoder.is_fit):
                    csv_cols[i].append(index_metas[i].encoder.encode(val))
                else:
                    csv_cols[i].append(val)

        features: List[List[Hashable]] = [ [] for _ in range(row_count) ]
        labels  : List[List[Hashable]] = [ [] for _ in range(row_count) ]

        for col in index_metas:
            if index_metas[col].ignore: continue

            if index_metas[col].encoder.is_fit:
                encode = lambda x: x
            else:
                encode = index_metas[col].encoder.fit(csv_cols[col]).encode
            
            for row,val in enumerate(csv_cols[col]):
                
                if index_metas[col].label:
                    labels[row].extend(encode(val))
                else:
                    features[row].extend(encode(val))

        to_hashable = lambda x: x[0] if len(x) ==1 else tuple(x)

        return ClassificationSimulation([to_hashable(f) for f in features], [to_hashable(l) for l in labels])

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
