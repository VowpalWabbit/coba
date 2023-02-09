from abc import abstractmethod, ABC
from typing import Any, Union, Iterable, Sequence, Mapping, overload

from coba.primitives import Context, Action, Actions
from coba.primitives import Reward, SequenceReward, Feedback, SequenceFeedback
from coba.pipes import Source, SourceFilters, Filter
from coba.exceptions import CobaException

class Interaction(dict):
    """An individual interaction that occurs in an Environment."""
    __slots__=()
    keywords = {}

    @property
    def extra(self) -> Mapping[str,Any]:
        return { k:self[k] for k in self.keys()-self.keywords}

class SimulatedInteraction(Interaction):
    """Simulated data that provides labels for every possible action."""
    __slots__=()
    keywords = {'type', 'context', 'actions', 'rewards'}
    
    def __init__(self,
        context : Context,
        actions : Actions,
        rewards : Union[Reward, Sequence[float]],
        **kwargs) -> None:
        """Instantiate SimulatedInteraction.

        Args
            context : Features describing the interaction's context.
            actions : Features describing available actions during the interaction.
            rewards : The reward for each action in the interaction.
            kwargs : Any additional information.
        """

        self['context'] = context
        self['actions'] = actions
        if isinstance(rewards,(list,tuple)):
            if len(rewards) != len(actions): 
                raise CobaException("The given actions and rewards did not line up.")
            self['rewards'] = SequenceReward(rewards)
        else:
            self['rewards'] = rewards

        if kwargs: self.update(kwargs)

class GroundedInteraction(Interaction):
    """Logged data providing a label for one action."""
    __slots__=()
    keywords = {'type', 'context', 'actions', 'rewards', 'feedbacks'}

    def __init__(self,
        context: Context,
        actions: Actions,
        rewards: Union[Reward, Sequence[float]],
        feedbacks: Union[Feedback, Sequence[Any]],
        **kwargs) -> None:
        """Instantiate GroundedInteraction.

        Args
            context: Features describing the interaction's context.
            actions: Features describing available actions during the interaction.
            rewards: The reward for each action in the interaction.
            feedbacks: The feedback for each action in the interaction.
            **kwargs: Additional information that should be recorded in the interactions table of an experiment result.
        """

        self['context'] = context
        self['actions'] = actions
        self['rewards'] = SequenceReward(rewards) if isinstance(rewards,(list,tuple)) else rewards
        self['feedbacks'] = SequenceFeedback(feedbacks) if isinstance(feedbacks,(list,tuple)) else feedbacks

        if kwargs: self.update(kwargs)

class LoggedInteraction(Interaction):
    """Logged data that describes an interaction where the choice was already made."""
    __slots__ = ()
    keywords = {'type', 'context', 'action', 'reward', 'probability', 'actions', 'rewards'}

    @overload
    def __init__(self,
        context: Context,
        action: Action,
        reward: float,
        *,
        probability: float=None,
        actions: Actions=None,
        rewards: Union[Reward, Sequence[float]] = None,
        **kwargs) -> None:
        ...

    def __init__(self,
        context: Context,
        action: Action,
        reward: float,
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
            **kwargs : Any additional information.
        """

        if kwargs.get('actions') is not None \
                and kwargs.get('probability') is not None \
                and kwargs.get('rewards') is None:
            probability       = kwargs['probability']
            actions           = kwargs['actions']
            kwargs['rewards'] = [int(a==action)*reward/probability for a in actions]

        self['context'] = context
        self['action']  = action
        self['reward']  = reward

        if kwargs.get('rewards') is not None and isinstance(kwargs['rewards'],(list,tuple)):
            kwargs['rewards'] = SequenceReward(kwargs['rewards'])

        if kwargs: self.update({k:v for k,v in kwargs.items() if v is not None})

class EnvironmentFilter(Filter[Iterable[Interaction],Iterable[Interaction]], ABC):
    """A filter that can be applied to an Environment."""

    @abstractmethod
    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        """Apply a filter to an Environment's interactions."""
        ...

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
