from abc import abstractmethod, ABC
from typing import Any, Union, Iterable, Sequence, Mapping, overload

from coba.primitives import Context, Action, Actions
from coba.primitives import Reward, SequenceReward, Feedback, SequenceFeedback
from coba.pipes import Source, SourceFilters, Filter

class Interaction(dict):
    """An individual interaction that occurs in an Environment."""
    __slots__=()
    keywords = {}

    @staticmethod
    def from_dict(kwargs_dict: Mapping[str, Any]) -> 'Interaction':
        if 'feedbacks' in kwargs_dict: return GroundedInteraction(**kwargs_dict)
        if 'rewards' in kwargs_dict: return SimulatedInteraction(**kwargs_dict)
        if 'reward' in kwargs_dict: return LoggedInteraction(**kwargs_dict)
        return kwargs_dict

class SimulatedInteraction(Interaction):
    """Simulated data that provides rewards for every possible action."""
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
        self['rewards'] = rewards if not isinstance(rewards,(list,tuple)) else SequenceReward(actions,rewards)

        if kwargs: self.update(kwargs)

class GroundedInteraction(Interaction):
    """A grounded interaction based on Interaction Grounded Learning which feedbacks instead of rewards."""
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

        self['context']   = context
        self['actions']   = actions
        self['rewards']   = SequenceReward(actions,rewards) if isinstance(rewards,(list,tuple)) else rewards
        self['feedbacks'] = SequenceFeedback(actions,feedbacks) if isinstance(feedbacks,(list,tuple)) else feedbacks

        if kwargs: self.update(kwargs)

class LoggedInteraction(Interaction):
    """A logged interaction with an action, reward and optional probability."""
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

        if isinstance(kwargs.get('rewards'),(list,tuple)):
            self['rewards'] = SequenceReward(kwargs['actions'],kwargs.pop('rewards'))

        self['context'] = context
        self['action']  = action
        self['reward']  = reward

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
    def params(self) -> Mapping[str,Any]: #pragma: no cover
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

class SafeEnvironment(Environment):
    """A wrapper for environment-likes that guarantees interface consistency."""

    def __init__(self, environment: Environment) -> None:
        """Instantiate a SafeEnvironment.

        Args:
            environment: The environment we wish to make sure has the expected interface
        """

        self.environment = environment if not isinstance(environment, SafeEnvironment) else environment.environment

    @property
    def params(self) -> Mapping[str, Any]:
        try:
            params = self.environment.params
        except AttributeError:
            params = {}

        if "env_type" not in params:

            if isinstance(self.environment, SourceFilters):
                params["env_type"] = self.environment._source.__class__.__name__
            else:
                params["env_type"] = self.environment.__class__.__name__

        return params

    def read(self) -> Iterable[Interaction]:
        return self.environment.read()

    def __str__(self) -> str:
        params = dict(self.params)
        tipe   = params.pop("env_type")

        if len(params) > 0:
            return f"{tipe}({','.join(f'{k}={v}' for k,v in params.items())})"
        else:
            return tipe

class SimpleEnvironment(Environment):

    def __init__(self, interactions: Sequence[Interaction]=(), params: Mapping[str,Any]={}) -> None:
        self._interactions = interactions
        self._params = params

    @property
    def params(self):
        return self._params

    def read(self) -> Iterable[Interaction]:
        return self._interactions
