import collections.abc

from numbers import Number
from abc import abstractmethod, ABC
from typing import Any, Union, Iterable, Sequence, Mapping, Optional

from coba.utilities import HashableDict
from coba.pipes import Source, SourceFilters
from coba.exceptions import CobaException

Action  = Union[str, Number, tuple, HashableDict]
Context = Union[None, str, Number, tuple, HashableDict]

class Interaction:
    """An individual interaction that occurs in an Environment."""

    def __init__(self, context: Context, actions: Optional[Sequence[Action]], rewards: Optional[Sequence[float]], **kwargs) -> None:
        """Instantiate an Interaction.

        Args:
            context: The context in which the interaction occured.
        """
        self._context = context
        self._actions = actions
        self._rewards = rewards
        self._kwargs  = kwargs

        if rewards and actions and len(rewards) != len(actions):
            raise CobaException("An interaction's reward count must equal its action count.")

    @property
    def context(self) -> Context:
        """The context in which the interaction occured."""
        if not isinstance(self._context, collections.abc.Hashable):
            self._context = self._make_hashable(self._context)
        return self._context

    @property
    def actions(self) -> Sequence[Action]:
        """The actions available in the interaction."""
        if not isinstance(self._actions[0], collections.abc.Hashable):
            self._actions = list(map(self._make_hashable,self._actions))
        return self._actions

    @property
    def rewards(self) -> Sequence[float]:
        return self._rewards

    @property
    def kwargs(self) -> Mapping[str,Any]:
        """Additional information associatd with the Interaction."""
        return self._kwargs

    def _make_hashable(self, feats):
        try:
            return feats.to_builtin()
        except Exception:
            if isinstance(feats, collections.abc.Sequence):
                return tuple(feats)

            if isinstance(feats, collections.abc.Mapping):
                return HashableDict(feats)

            return feats

class GroundedInteraction(Interaction):
    """Logged data that describes an interaction where the choice was already made."""

    def __init__(self,
        context: Context,
        actions: Sequence[Action],
        rewards: Sequence[float],
        feedbacks: Sequence[Any],
        **kwargs) -> None:
        ...
        """Instantiate GroundedInteraction.

        Args
            context: Features describing the interaction's context.
            actions: Features describing available actions during the interaction.
            rewards: The reward for each action in the interaction.
            feedbacks: The feedback for each action in the interaction.
            **kwargs: Additional information that should be recorded in the interactions table of an experiment result.
        """

        super().__init__(context, actions, rewards, **kwargs)

        self._feedbacks = feedbacks

    @property
    def feedbacks(self) -> Sequence[Any]:
        """The feedback for each action in the interaction."""
        return self._feedbacks

class LoggedInteraction(Interaction):
    """Logged data that describes an interaction where the choice was already made."""

    def __init__(self,
        context: Context,
        action: Action,
        reward: float,
        probability: Optional[float] = None,
        actions: Optional[Sequence[Action]] = None,
        rewards: Optional[Sequence[float]] = None,
        **kwargs) -> None:
        """Instantiate LoggedInteraction.

        Args
            context: Features describing the logged context.
            action: Features describing the action taken by the logging policy.
            reward: The reward that was revealed when the logged action was taken.
            probability: The probability that the logged action was taken. That is P(action|context,actions,logging policy).
            actions: All actions that were availble to be taken when the logged action was taken. Necessary for OPE.
            rewards: The rewards to use for off policy evaluation. These rewards will not be shown to any learners. They will
                only be recorded in experimental results. If probability and actions is provided and rewards is None then 
                rewards will be initialized using the IPS estimator.
            **kwargs : Additional information that should be recorded in the interactions table of an experiment result. If
                any data is a sequence with length equal to actions only the data at the selected action index will be recorded.
        """
        self._action      = self._make_hashable(action)
        self._reward      = reward
        self._probability = probability

        if probability and actions and rewards is None:
            rewards = [ int(a==action)*reward/probability for a in actions ]

        super().__init__(context, actions, rewards, **kwargs)

    @property
    def action(self) -> Action:
        """The action that was taken."""
        return self._action

    @property
    def reward(self) -> float:
        """The reward that was observed after taking the given action."""
        return self._reward

    @property
    def probability(self) -> Optional[float]:
        """The probability the action was taken."""
        return self._probability

    @property
    def actions(self) -> Optional[Sequence[Action]]:
        """The actions that were available to the logging policy."""
        return self._actions

    @property
    def rewards(self) -> Optional[Sequence[Action]]:
        """The rewards to use for off policy evaluation."""
        return self._rewards

class SimulatedInteraction(Interaction):
    """Simulated data that describes an interaction where the choice is up to you."""

    def __init__(self,
        context : Context,
        actions : Sequence[Action],
        rewards : Sequence[float],
        **kwargs) -> None:
        ...
        """Instantiate SimulatedInteraction.

        Args
            context : Features describing the interaction's context.
            actions : Features describing available actions during the interaction.
            rewards : The reward for each action in the interaction.
            **kwargs: Additional information that should be recorded in the interactions table of an experiment result. If any
                data is a sequence with length equal to actions only the data at the selected action index will be recorded.
        """

        super().__init__(context, actions, rewards, **kwargs)

class Environment(Source[Iterable[Interaction]], ABC):
    """An Environment that produces Contextual Bandit data"""

    @property
    def params(self) -> Mapping[str,Any]: # pragma: no cover
        """Paramaters describing the simulation.

        Remarks:
            These will become columns in the environments table of experiment results.
        """
        return {}

    @abstractmethod
    def read(self) -> Iterable[Interaction]:
        """The sequence of interactions in the simulation.

        Remarks:
            This function should always be "re-iterable".
        """
        ...

    def __str__(self) -> str:
        return str(self.params) if self.params else self.__class__.__name__

class SimulatedEnvironment(Environment):
    """An environment made from SimulatedInteractions."""

    @abstractmethod
    def read(self) -> Iterable[SimulatedInteraction]:
        """The sequence of interactions in the environment.

        Remarks:
            This function should always be "re-iterable".
        """
        ...

class SafeEnvironment(Environment):
    """A wrapper for environment-likes that guarantees interface consistency."""

    def __init__(self, environment: Environment) -> None:
        """Instantiate a SafeEnvironment.

        Args:
            environment: The environment we wish to make sure has the expected interface
        """

        self._environment = environment if not isinstance(environment, SafeEnvironment) else environment._environment

    @property
    def params(self) -> Mapping[str, Any]:
        try:
            params = self._environment.params
        except AttributeError:
            params = {}

        if "type" not in params:

            if isinstance(self._environment, SourceFilters):
                params["type"] = self._environment._source.__class__.__name__
            else:
                params["type"] = self._environment.__class__.__name__

        return params

    def read(self) -> Iterable[Interaction]:
        return self._environment.read()

    def __str__(self) -> str:
        params = dict(self.params)
        tipe   = params.pop("type")

        if len(params) > 0:
            return f"{tipe}({','.join(f'{k}={v}' for k,v in params.items())})"
        else:
            return tipe
