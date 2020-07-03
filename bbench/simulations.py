"""The simulations module contains core classes and types for defining bandit simulations.

This module contains the abstract interface expected for bandit simulations along with the 
class defining a Round within a bandit simulation. Additionally, this module also contains 
the type hints for State, Action and Reward. These type hints don't contain any functionality. 
Rather, they simply make it possible to use static type checking for any project that desires 
to do so.

Todo:
    * Add RegressionSimulation(Simulation)
"""

import csv
import itertools
import urllib.request
import json
import io
import hashlib
import gzip

from warnings import warn
from contextlib import closing
from abc import ABC, abstractmethod
from typing import Optional, Iterator, Iterable, Sequence, List, Union, Callable, TextIO, TypeVar, Generic, Tuple

from bbench.random import shuffle

#state, action, reward types
State  = Optional[Union[str,float,Tuple[Union[str,float],...]]]
Action = Union[str,float,Tuple[Union[str,float],...]]
Reward = float

T_S = TypeVar('T_S', bound=State)
T_A = TypeVar('T_A', bound=Action)

class Round:
    """A class to contain all data needed to represent a round in a bandit simulation."""

    def __init__(self, 
                 state  : State,
                 actions: Sequence[Action],
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
    def state(self) -> State:
        """Read-only property providing the round's state."""
        return self._state

    @property
    def actions(self) -> Sequence[Action]:
        """Read-only property providing the round's possible actions."""
        return self._actions

    @property
    def rewards(self) -> Sequence[Reward]:
        """Read-only property providing the round's reward for each action."""
        return self._rewards

class Simulation(ABC):
    """The simulation interface."""

    @property
    @abstractmethod
    def rounds(self) -> Sequence[Round]:
        """A read-only property providing the sequence of rounds in a simulation.

        Returns:
            The simulation's sequence of rounds.

        Remarks:
            All Benchmark assume that rounds is re-iterable. So long as rounds is a
            Sequence[Round] it will always be re-iterable. If rounds were merely 
            Iterable[Round] then it is possible for it to only allow enumeration once.
        """
        ...

class ClassificationSimulation(Simulation):
    """A simulation created from classifier data with features and labels.
    
    Remark:
        ClassificationSimulation creation turns each feature set and label, into a round. 
        In each round the feature set becomes the state and all possible labels become the
        actions. Rewards for each round are created by assigning a reward of 1 to the correct 
        label (action) for a feature set (state) and a value of 0 for all other labels (actions).
    """

    @staticmethod
    def from_openml(data_id:int) -> Simulation:
        # pylint: disable=no-member #pylint really doesn't like "bunch"

        with closing(urllib.request.urlopen(f'https://www.openml.org/api/v1/json/data/{data_id}')) as resp:
            data = json.loads(resp.read())["data_set_description"]

        with closing(urllib.request.urlopen(f'http://www.openml.org/api/v1/json/data/features/{data_id}')) as resp:
            meta = json.loads(resp.read())["data_features"]["feature"]

        def stater(row: Sequence[str]) -> State:
            state: List[Union[int,float]] = []
            
            for value,desc in zip(row,meta):
                if(desc["is_ignore"] == "true" or desc["is_row_identifier"] == "true"):
                    continue
                
                if(desc["data_type"] == "numeric"):
                    state.append(float(value))

                if(desc["data_type"] == "nominal"):
                    
                    if(len(desc["nominal_value"]) == 2):
                        state.append(int(value == desc["nominal_value"][0]))
                    else:
                        state.extend([int(nv == value) for nv in desc["nominal_value"]])
            
            return tuple(state)

        is_target  = lambda feature: feature["is_target"] == "true"
        is_feature = lambda feature: feature["is_target"] != "true"

        label_col  = list(filter(is_target, meta))[0]['name']
        meta       = list(filter(is_feature, meta))

        file_url = f"http://www.openml.org/data/v1/get_csv/{data['file_id']}"
        file_req = urllib.request.Request(file_url, headers={'Accept-encoding':'gzip'})

        with closing(urllib.request.urlopen(file_req)) as resp:

            # actual_md5_checksum = hashlib.md5()

            if resp.info().get('Content-Encoding') == "gzip":
                resp_stream = gzip.GzipFile(fileobj=resp)
            else:
                resp_stream = resp

            def decoded_lines() -> Iterator[str]:
                for line in resp_stream:
                    # actual_md5_checksum.update(line)
                    yield line.decode('utf-8')

            simulation = ClassificationSimulation.from_csv_rows(csv.reader(decoded_lines()), label_col, csv_stater=stater)

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
        csv_path: str, 
        lbl_column: Union[str,int],
        has_header: bool = True,
        csv_reader: Callable[[TextIO], Iterator[List[str]]] = csv.reader, 
        csv_stater: Callable[[Sequence[str]], State] = lambda row: tuple(row) ) -> Simulation:
        """Create a ClassificationSimulation from a csv file with a header row.

        Args:
            csv_path: The path to the csv file.
            lbl_column: The name of the column in the csv file that represents the label.
            has_header: Indicates if the csv file has a header row.
            csv_reader: A method to parse file lines at csv_path into their string values.
            csv_stater: A method to convert csv string values into state representations.

        Remarks:
            This method will open the file and read it all into memory. Be careful when doing
            this if you are working with a large file. One way to improve on this is to make
            sure column are correctly typed and all string columns are represented as integer
            backed categoricals (aka, `factors` in R).
        """

        with open(csv_path, newline='') as csv_file:
            return ClassificationSimulation.from_csv_rows(csv_reader(csv_file), lbl_column, csv_stater = csv_stater)
    
    @staticmethod
    def from_csv_rows(
        csv_rows  : Iterable[List[str]],
        lbl_column: Union[int,str],
        has_header: bool = True,
        csv_stater: Callable[[Sequence[str]], State] = lambda row: tuple(row)) -> Simulation:

        """Create a ClassifierSimulation from the string values of a csv file.

        Args:
            csv_rows: Any iterable of string values representing a row with features/label.
            label_col: Either the column index or the header name for the label column.
            csv_stater: A method to convert a csv row into state representations.
        """

        # In theory we don't have to load the whole file up front. However, in practice,
        # not loading the file upfront is hard due to the fact that Python can't really
        # guarantee a generator will close a file.
        # For more info see https://stackoverflow.com/q/29040534/1066291
        # For more info see https://www.python.org/dev/peps/pep-0533/

        csv_iter              = iter(csv_rows)
        features: List[State] = []
        labels  : List[str]   = []

        if not has_header and isinstance(lbl_column, str):
            raise Exception("We are unable to determine the label by name because the csv does not have a header.")

        if has_header:
            header_row = next(csv_iter)

        label_index = lbl_column if isinstance(lbl_column, int) else header_row.index(lbl_column)

        for row in csv_iter:

            if(len(row) == 0): continue #ignore empty lines

            features.append(csv_stater(row[:label_index] + row[(label_index+1):]))
            labels  .append(row[label_index])

        return ClassificationSimulation(features, labels)

    def __init__(self, features: Sequence[State], labels: Sequence[Union[str,float]]) -> None:
        """Instantiate a ClassificationSimulation.

        Args:
            features: The collection of features used for the original classifier problem.
            labels: The collection of labels assigned to each observation of features.
        """

        assert len(features) == len(labels), "Mismatched lengths of features and labels"
        
        states      = features
        action_set  = sorted(list(set(labels)))
        reward_sets = [ [int(label==action) for action in action_set] for label in labels ]

        self._rounds = list(map(Round, states, itertools.repeat(action_set), reward_sets))

    @property
    def rounds(self) -> Sequence[Round]:
        """The rounds in this simulation.
        
        Remarks:
            See the Simulation base class for more information.
        """
        return self._rounds

class LambdaSimulation(Simulation, Generic[T_S, T_A]):
    """A Simulation created from lambda functions that generate states, actions and rewards.
    
    Remarks:
        This implementation is useful for creating simulations from defined distributions.
    """

    def __init__(self,
                 n_rounds: int,
                 S: Callable[[int],T_S], 
                 A: Callable[[T_S],Sequence[T_A]], 
                 R: Callable[[T_S,T_A],Reward]) -> None:
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
    def rounds(self) -> Sequence[Round]:
        """The rounds in this simulation.
        
        Remarks:
            See the Simulation base class for more information.
        """
        return self._rounds

    def _round_generator(self) -> Iterator[Round]:
        """Generate rounds for this simulation."""

        S = self._S
        A = self._A
        R = self._R

        for i in itertools.count():
            state   = S(i)
            actions = A(state)
            rewards = [R(state,action) for action in actions]

            yield Round(state,actions,rewards)

class MemorySimulation(Simulation):
    """A Simulation implementation created from in memory sequences of Rounds.
    
    Remarks:
        This implementation is very useful for unit-testing known edge cases.
    """

    def __init__(self, rounds: Sequence[Round]) -> None:
        """Instantiate a MemorySimulation.

        Args:
            rounds: a collection of rounds to turn into a simulation.
        """
        self._rounds = rounds

    @property
    def rounds(self) -> Sequence[Round]:
        """The rounds in this simulation.
        
        Remarks:
            See the Simulation base class for more information.
        """
        return self._rounds

class ShuffleSimulation(Simulation):
    def __init__(self, smiulation: Simulation, seed: Optional[int] = None):

        self._rounds = shuffle(list(smiulation.rounds))
    
    @property
    def rounds(self) -> Sequence[Round]:

        return self._rounds