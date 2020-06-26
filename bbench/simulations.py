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
import random
import itertools

from abc import ABC, abstractmethod
from typing import Optional, Iterator, Sequence, List, Union, Callable, TextIO, TypeVar, Generic, Tuple

from bbench.utilities import check_sklearn_datasets_support

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

        check_sklearn_datasets_support("OpenMLSimulation.__init__")
        import sklearn.datasets as ds #type: ignore
        import numpy as np #type: ignore

        bunch = ds.fetch_openml(data_id=data_id)

        for keys_to_remove in [k for (k,v) in bunch.categories.items() if len(v) <= 2]:
            del bunch.categories[keys_to_remove]

        n_rows = bunch.data.shape[0]
        n_cols = bunch.data.shape[1] - len(bunch.categories.keys()) + sum(map(len,bunch.categories.values()))

        #pre-allocate everything
        feature_matrix = np.empty((n_rows, n_cols))
        matrix_index = 0

        for feature_index,feature_name in enumerate(bunch.feature_names):
            if(feature_name in bunch.categories):
                onehot_index = feature_matrix[:,matrix_index].astype(int)
                feature_matrix[np.arange(n_rows),matrix_index+onehot_index] = 1
                matrix_index += len(bunch.categories[feature_name])
            else:
                feature_matrix[:,matrix_index] = bunch.data[:,feature_index]
                matrix_index += 1

        return ClassificationSimulation(list(map(tuple,feature_matrix)), list(bunch.target))

    @staticmethod
    def from_csv_path(
        csv_path: str, 
        label_col: Union[str,int], 
        csv_reader: Callable[[TextIO], Iterator[List[str]]] = csv.reader, 
        csv_stater: Callable[[Sequence[str]], State] = lambda row: tuple(row) ) -> Simulation:
        """Create a ClassificationSimulation from a csv file with a header row.

        Args:
            csv_path: The path to the csv file.
            label_col: The name of the column in the csv file that represents the label.
            csv_reader: A method to parse file lines at csv_path into their string values.
            csv_stater: A method to convert csv string values into state representations.

        Remarks:
            This method will open the file and read it all into memory. Be careful when doing
            this if you are working with a large file. One way to improve on this is to make
            sure column are correctly typed and all string columns are represented as integer
            backed categoricals (aka, `factors` in R).
        """

        with open(csv_path, newline='') as csv_file:
            return ClassificationSimulation.from_csv_file(csv_file, label_col, csv_reader, csv_stater)

    @staticmethod
    def from_csv_file(
        csv_file: TextIO, 
        label_col: Union[str,int], 
        csv_reader: Callable[[TextIO], Iterator[List[str]]] = csv.reader, 
        csv_stater: Callable[[Sequence[str]], State] = lambda row: tuple(row)) -> Simulation:

        """Create a ClassificationSimulation from the TextIO of a csv file.

        Args:
            csv_file: Any TextIO implementation including `open(csv_path)` and `io.StringIO()`.
            label_col: The name of the column in the csv file that represents the label.
            csv_reader: A method to parse file lines at csv_path into their string values.
            csv_stater: A method to convert csv string values into state representations.

        Remarks:
            This method will open the file and read it all into memory. Be careful when doing
            this if you are working with a large file. One way to improve on this is to make
            sure column are correctly typed and all string columns are represented as integer
            backed categoricals (aka, `factors` in R).
        """

        return ClassificationSimulation.from_csv_rows(csv_reader(csv_file), label_col, csv_stater)
    
    @staticmethod
    def from_csv_rows(
        csv_rows  : Iterator[List[str]],
        label_col : Union[str,int],
        csv_stater: Callable[[Sequence[str]], State] = lambda row: tuple(row)) -> Simulation:

        """Create a ClassifierSimulation from the string values of a csv file.

        Args:
            csv_rows: Any Iterator of string values representing a row of features and a label.
            label_col: The value of the column in the header row representing the label.
            csv_stater: A method to convert csv string values into state representations.

        Remarks:
            This method will open the file and read it all into memory. Be careful when doing
            this if you are working with a large file. One way to improve on this is to make
            sure column are correctly typed and all string columns are represented as integer
            backed categoricals (aka, `factors` in R).
        """

        features: List[State] = []
        labels  : List[str]   = []

        # In theory we don't have to load the whole file up front. However, in practice,
        # not loading the file upfront is hard due to the fact that Python can't really 
        # guarantee a generator will close a file.
        # For more info see https://stackoverflow.com/q/29040534/1066291
        # For more info see https://www.python.org/dev/peps/pep-0533/

        if isinstance(label_col, int):
            label_index = label_col
        else:
            header_row  = next(csv_rows)
            label_index = header_row.index(label_col)

        for row in csv_rows:
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
    def __init__(self, smiulation: Simulation):

        self._rounds = list(smiulation.rounds)
        random.shuffle(self._rounds)
    
    @property
    def rounds(self) -> Sequence[Round]:

        return self._rounds