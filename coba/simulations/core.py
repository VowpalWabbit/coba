from abc import ABC, abstractmethod

from itertools import accumulate
from typing import Optional, Sequence, List, Callable, Hashable, Tuple, Dict, Union, Any

import coba.random

from coba.data.sources import Source
from coba.data.encoders import OneHotEncoder

Context = Optional[Union[Hashable,Dict[Any,Hashable]]]
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
    def reward(self) -> 'Reward':
        """The reward object which can observe rewards for pairs of actions and interaction keys."""
        ...    

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