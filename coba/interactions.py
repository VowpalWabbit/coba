from typing import Union, Sequence
from coba.primitives import Context, Action, Actions, Rewards, Interaction

class SimulatedInteraction(Interaction):
    """An interaction with reward information for every possible action."""
    __slots__=()

    def __init__(self,
        context: Context,
        actions: Actions,
        rewards: Union[Rewards, Sequence[float]],
        **kwargs) -> None:
        """Instantiate SimulatedInteraction.

        Args:
            context : Features describing the interaction's context.
            actions : Features describing available actions during the interaction.
            rewards : The reward for each action in the interaction.
            kwargs : Any additional information.
        """

        self['context'] = context
        self['actions'] = actions
        self['rewards'] = rewards

        if kwargs: self.update(kwargs)

class GroundedInteraction(Interaction):
    """An interaction with feedbacks for Interaction Grounded Learning."""
    __slots__=()

    def __init__(self,
        context  : Context,
        actions  : Actions,
        rewards  : Union[Rewards, Sequence[float]],
        feedbacks: Union[Rewards, Sequence[float]],
        **kwargs) -> None:
        """Instantiate GroundedInteraction.

        Args:
            context: Features describing the interaction's context.
            actions: Features describing available actions during the interaction.
            rewards: The reward for each action in the interaction.
            feedbacks: The feedback for each action in the interaction.
            **kwargs: Additional information that should be recorded in the interactions table of an experiment result.
        """

        self['context']   = context
        self['actions']   = actions
        self['rewards']   = rewards
        self['feedbacks'] = feedbacks

        if kwargs: self.update(kwargs)

class LoggedInteraction(Interaction):
    """An interaction with the reward and propensity score for a single action."""
    __slots__ = ()

    def __init__(self,
        context: Context,
        action: Action,
        reward: float,
        probability:float = None,
        **kwargs) -> None:
        """Instantiate LoggedInteraction.

        Args:
            context: Features describing the logged context.
            action: Features describing the action taken by the logging policy.
            reward: The reward that was revealed when the logged action was taken.
            probability: The probability that the logged action was taken. That is P(action|context,actions,logging policy).
            actions: All actions that were availble to be taken when the logged action was taken. Necessary for OPE.
            rewards: The rewards to use for off policy evaluation. These rewards will not be shown to any learners. They will
                only be recorded in experimental results. If probability and actions is provided and rewards is None then
                rewards will be initialized using the IPS estimator.
            **kwargs: Any additional information.
        """

        self['context'] = context
        self['action']  = action
        self['reward']  = reward

        if probability is not None:
            self['probability'] = probability

        if kwargs: self.update(kwargs)
