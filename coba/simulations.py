"""The simulations module contains core classes and types for defining contextual bandit simulations.

This module contains the abstract interface expected for bandit simulations along with the 
class defining an Interaction within a bandit simulation. Additionally, this module also contains 
the type hints for Context, Action and Reward. These type hints don't contain any functionality. 
Rather, they simply make it possible to use static type checking for any project that desires 
to do so.

TODO Add RegressionSimulation
"""

import gc
import json

from itertools import repeat, count, chain
from abc import ABC, abstractmethod
from typing import (
    Optional, Sequence, List, Union, Callable, 
    TypeVar, Generic, Hashable, Dict, Any, Tuple
)

import coba.random

from coba.data.sources import Source, OpenmlSource
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

    _simulation: Optional[Simulation[_C_out, _A_out]]

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
    def interactions(self) -> Sequence[Interaction[_C_out,_A_out]]:
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

        if config["format"] == "openml":
            with ExecutionContext.Logger.log(f"loading openml {config['id']}..."):
                return ClassificationSimulation.from_source(OpenmlSource(config["id"], config.get("md5_checksum", None)))

        raise Exception("We were unable to recognize the provided data format.")

    @staticmethod
    def from_source(source: Source[Tuple[Sequence[Sequence[Any]], Sequence[Any]]]) -> 'ClassificationSimulation[Context]':
        
        features, actions = source.read()

        if isinstance(source, OpenmlSource) and len(actions[0]) == 1:
            raise Exception("This does not appear to be a classification dataset. Creating a ClassificationSimulation from it will perform poorly.")

        return ClassificationSimulation(features, actions)

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

class OpenmlSimulation(LazySimulation[Context, Tuple[int,...]]):
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

    def load_simulation(self) -> Simulation[Context, Tuple[int,...]]:
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