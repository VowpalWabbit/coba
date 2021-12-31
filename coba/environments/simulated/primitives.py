
import collections.abc

from abc import abstractmethod
from itertools import chain, repeat, count, islice
from typing import Sequence, Dict, Any, Iterable, Union, Callable, Tuple, overload, Optional

from coba.random import CobaRandom
from coba.exceptions import CobaException

from coba.environments.primitives import Context, Action, Environment, Interaction

T_Dense_Feat  = Sequence[Any]
T_Sparse_Feat = Dict[Any,Any]
T_Dense_Rows  = Iterable[Tuple[T_Dense_Feat,Any]]
T_Sparse_Rows = Iterable[Tuple[T_Sparse_Feat,Any]]

class SimulatedInteraction(Interaction):
    """A class to contain all data needed to represent an interaction in a simulated bandit interaction."""

    @overload
    def __init__(self,
        context: Context,
        actions: Sequence[Action],
        *,
        rewards: Sequence[float],
        **kwargs) -> None:
        ...
        """Instantiate SimulatedInteraction.

        Args
            context : Features describing the interaction's context. This should be `None` for multi-armed bandit simulations.
            actions : Features describing available actions in the interaction.
            rewards : The reward that will be revealed to learners based on the taken action. We require len(rewards) == len(actions).
            **kwargs: Additional information that should be recorded in the interactions table of an experiment result. If any
                data is a sequence with length equal to actions only the data at the selected action index will be recorded.
        """

    @overload
    def __init__(self,
        context: Context,
        actions: Sequence[Action], 
        *,
        reveals: Sequence[Any],
        **kwargs) -> None:
        ...
        """Instantiate SimulatedInteraction.

        Args
            context : Features describing the interaction's context. Will be `None` for multi-armed bandit simulations.
            actions : Features describing available actions in the interaction.
            reveals : The data that will be revealed to learners based on the selected action. We require len(reveals) == len(actions).
                When working with non-scalar data use "reveals" instead of "rewards" to make it clear to Coba the data is non-scalar.
            **kwargs: Additional information that should be recorded in the interactions table of an experiment result. If any
                data is a sequence with length equal to actions only the data at the selected action index will be recorded.
        """

    @overload
    def __init__(self, 
        context: Context, 
        actions: Sequence[Action], 
        *,
        rewards : Sequence[float],
        reveals : Sequence[Any],
        **kwargs) -> None:
        ...
        """Instantiate SimulatedInteraction.

        Args
            context : Features describing the interaction's context. Will be `None` for multi-armed bandit simulations.
            actions : Features describing available actions in the interaction.
            rewards : A sequence of scalar values representing reward. When both rewards and reveals are provided only 
                reveals will be shown to the learner when an action is selected. The reward values will only be used 
                by Coba when plotting experimental results. We require that len(rewards) == len(actions).
            reveals : The data that will be revealed to learners based on the selected action. We require len(reveals) == len(actions).
                When working with non-scalar data use "reveals" instead of "rewards" to make it clear to Coba the data is non-scalar.
            **kwargs: Additional information that should be recorded in the interactions table of an experiment result. If any
                data is a sequence with length equal to actions only the data at the selected action index will be recorded.
        """

    def __init__(self, context: Context, actions: Sequence[Action], **kwargs) -> None:

        assert kwargs.keys() & {"rewards", "reveals"}, "Interaction requires either a rewards or reveals keyword warg."

        assert "rewards" not in kwargs or len(actions) == len(kwargs["rewards"]), "Interaction rewards must match action length."
        assert "reveals" not in kwargs or len(actions) == len(kwargs["reveals"]), "Interaction reveals must match action length."

        self._context = self._hashable(context)
        self._actions = list(map(self._hashable,actions))

        self._kwargs  = kwargs

        super().__init__(self._context)

    @property
    def context(self) -> Context:
        """The interaction's context description."""

        return self._context

    @property
    def actions(self) -> Sequence[Action]:
        """The interaction's available actions."""

        return self._actions

    @property
    def kwargs(self) -> Dict[str,Any]:
        return self._kwargs

class SimulatedEnvironment(Environment):
    """The interface for a simulated environment."""
    
    @abstractmethod
    def read(self) -> Iterable[SimulatedInteraction]:
        """The sequence of interactions in a simulation.

        Remarks:
            This function should always be "re-iterable".
        """
        ...


class MemorySimulation(SimulatedEnvironment):
    """A Simulation implementation created from in memory sequences of contexts, actions and rewards."""

    def __init__(self, interactions: Sequence[SimulatedInteraction], str="MemorySimulation", params={}) -> None:
        """Instantiate a MemorySimulation.

        Args:
            interactions: The sequence of interactions in this simulation.
        """
        self._interactions = interactions
        self._str          = str
        self._params       = params

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the simulation."""
        return self._params

    def read(self) -> Iterable[SimulatedInteraction]:
        """Read the interactions in this simulation."""
        return self._interactions

    def __str__(self) -> str:
        return self._str

class ClassificationSimulation(SimulatedEnvironment):
    """A simulation created from classification dataset with features and labels.

    ClassificationSimulation turns labeled observations from a classification data set
    into interactions. For each interaction the feature set becomes the context and 
    all possible labels become the actions. Rewards for each interaction are created by 
    assigning a reward of 1 for taking the correct action (i.e., choosing the correct
    label)) and a reward of 0 for taking any other action (i.e., choosing any of the
    incorrect lables).
    """

    @overload
    def __init__(self, examples: Union[T_Dense_Rows,T_Sparse_Rows]) -> None:
        """Instantiate a ClassificationSimulation.

        Args:
            examples: Labeled examples to use when creating the contextual bandit simulation.
        """

    @overload
    def __init__(self, features: Union[Iterable[T_Dense_Feat], Iterable[T_Sparse_Feat]], labels: Iterable[Any]) -> None:
        """Instantiate a ClassificationSimulation.

        Args:
            features: A sequence of features to use as context when creating the contextual bandit simulation.
            labels: A sequence of class labels to use as actions and rewards when creating the contextual bandit simulation
        """

    def __init__(self, *args, **kwargs) -> None:
        """Instantiate a ClassificationSimulation."""

        if len(args) not in [1,2]:
            raise CobaException("We were unable to determine which overloaded constructor to use")

        self._examples = args[0] if len(args) == 1 else zip(args[0],args[1])

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the simulation."""
        return { }

    def read(self) -> Iterable[SimulatedInteraction]:
        """Read the interactions in this simulation."""

        examples = list(self._examples)

        if examples:

            features,labels = zip(*examples)

            #how can we tell the difference between featurized labels and multilabels????
            #for now we will assume multilables will be passed in as arrays not tuples...
            if not isinstance(labels[0], collections.abc.Hashable):
                labels_flat = list(chain.from_iterable(labels))
            else:
                labels_flat = list(labels)

            reward        = lambda action,label: int(is_label(action,label) or in_multilabel(action,label)) #type: ignore
            is_label      = lambda action,label: action == label #type: ignore
            in_multilabel = lambda action,label: isinstance(label,collections.abc.Sequence) and action in label #type: ignore

            # shuffling so that action order contains no statistical information
            # sorting so that the shuffled values are always shuffled in the same order
            actions  = CobaRandom(1).shuffle(sorted(set(labels_flat))) 
            contexts = features
            rewards  = [ [ reward(action,label) for action in actions ] for label in labels ]

            for c,a,r in zip(contexts, repeat(actions), rewards):
                yield SimulatedInteraction(c,a,rewards=r)

class RegressionSimulation(SimulatedEnvironment):
    """A simulation created from regression dataset with features and labels.
    
    RegressionSimulation turns labeled observations from a regression data set
    into interactions. For each interaction the feature set becomes the context and 
    all possible labels become the actions. Rewards for each interaction are created by 
    assigning a minus absolute error. Rewards are close to zero for taking actions that are 
    closer to the correct action (label) and lower ones for being far from the correct action.
    
    """

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the simulation."""
        return {}

    @overload
    def __init__(self, examples: Union[T_Dense_Rows,T_Sparse_Rows]) -> None:
        """Instantiate a RegressionSimulation.

        Args:
            examples: Labeled examples to use when creating the contextual bandit simulation.
        """

    @overload
    def __init__(self, features: Union[Iterable[T_Dense_Feat], Iterable[T_Sparse_Feat]], labels: Iterable[float]) -> None:
        """Instantiate a RegressionSimulation.

        Args:
            features: A sequence of features to use as context when creating the contextual bandit simulation.
            labels: A sequence of numeric labels to use as actions and rewards when creating the contextual bandit simulation
        """
    
    def __init__(self, *args, **kwargs) -> None:
        """Instantiate a RegressionSimulation."""

        if len(args) not in [1,2]:
            raise CobaException("We were unable to determine which overloaded constructor to use")
        
        self._examples = args[0] if len(args) == 1 else zip(args[0],args[1])

    def read(self) -> Iterable[SimulatedInteraction]:
        """Read the interactions in this simulation."""
        
        examples = list(self._examples)

        if examples:

            features,labels = zip(*examples)

            reward   = lambda action,label: 1-round(abs(float(action)-float(label)),2)
            contexts = features
            actions  = CobaRandom(1).shuffle(sorted(set(labels)))
            rewards  = [ [ reward(action,label) for action in actions ] for label in labels ]

            for c,a,r in zip(contexts, repeat(actions), rewards):
                yield SimulatedInteraction(c,a,rewards=r)

class LambdaSimulation(SimulatedEnvironment):
    """A Simulation created from lambda functions that generate contexts, actions and rewards.

    Remarks:
        This implementation is useful for creating a simulation from defined distributions.
    """

    @overload
    def __init__(self,
        n_interactions: Optional[int],
        context       : Callable[[int               ],Context         ],
        actions       : Callable[[int,Context       ],Sequence[Action]],
        reward        : Callable[[int,Context,Action],float           ]) -> None:
        """Instantiate a LambdaSimulation.

        Args:
            n_interactions: An optional integer indicating the number of interactions in the simulation.
            context: A function that should return a context given an index in `range(n_interactions)`.
            actions: A function that should return all valid actions for a given index and context.
            reward: A function that should return the reward for the index, context and action.
        """

    @overload
    def __init__(self, 
        n_interactions: Optional[int],
        context       : Callable[[int               ,CobaRandom],Context         ],
        actions       : Callable[[int,Context       ,CobaRandom],Sequence[Action]],
        reward        : Callable[[int,Context,Action,CobaRandom],float           ],
        seed          : int) -> None:
        """Instantiate a LambdaSimulation.

        Args:
            n_interactions: An optional integer indicating the number of interactions in the simulation.
            context: A function that should return a context given an index and random state.
            actions: A function that should return all valid actions for a given index, context and random state.
            reward: A function that should return the reward for the index, context, action and random state.
            seed: An integer used to seed the random state in order to guarantee repeatability.
        """

    def __init__(self,n_interactions,context,actions,reward,seed=None) -> None:
        """Instantiate a LambdaSimulation."""

        self._n_interactions = n_interactions
        self._context        = context
        self._actions        = actions
        self._reward         = reward
        self._seed           = seed

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the simulation."""

        return {"lambda_seed": self._seed} if self._seed is not None else {}            

    def read(self) -> Iterable[SimulatedInteraction]:
        """Read the interactions in this simulation."""

        rng = None if self._seed is None else CobaRandom(self._seed)

        _context = lambda i    : self._context(i    ,rng) if rng else self._context(i) 
        _actions = lambda i,c  : self._actions(i,c  ,rng) if rng else self._actions(i,c)
        _reward  = lambda i,c,a: self._reward (i,c,a,rng) if rng else self._reward(i,c,a)  

        for i in islice(count(), self._n_interactions):
            context  = _context(i)
            actions  = _actions(i, context)
            rewards  = [ _reward(i, context, action) for action in actions]

            yield SimulatedInteraction(context, actions, rewards=rewards)

    def __str__(self) -> str:
        return "LambdaSimulation"

    def __reduce__(self) -> Tuple[object, ...]:
        if self._n_interactions is not None:
            return (MemorySimulation, (list(self.read()), str(self), self.params))
        else:
            message = (
                "In general LambdaSimulation cannot be pickled because Python is unable to pickle lambda methods. "
                "This is really only a problem if you are trying to perform an experiment with a LambdaSimulation and "
                "multiple processes. There are three options to get around this limitation: (1) run your experiment "
                "on a single process rather than multiple, (2) re-design your LambdaSimulation as a class that inherits "
                "from LambdaSimulation (see coba.environments.simulations.LinearSyntheticSimulation for an example), "
                "or (3) specify a finite number for n_interactions in the LambdaSimulation constructor (this allows "
                "us to create the interactions in memory and convert to a MemorySimulation when pickling).")
            raise CobaException(message)

