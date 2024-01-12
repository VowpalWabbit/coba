from typing import Union, Sequence
from coba.primitives import Context, Action, Actions, Reward, Prob, Rewards, Interaction

class SimulatedInteraction(Interaction):
    """An interaction with reward information for every possible action."""
    __slots__=()

    def __init__(self,
        context: 'Context',
        actions: 'Actions',
        rewards: Union[Rewards, Sequence['Reward']],
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
        context  : 'Context',
        actions  : 'Actions',
        rewards  : Union[Rewards, Sequence['Reward']],
        feedbacks: Union[Rewards, Sequence['Reward']],
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
        context    : 'Context',
        action     : 'Action',
        reward     : 'Reward',
        probability: 'Prob' = None,
        **kwargs) -> None:
        """Instantiate LoggedInteraction.

        Args:
            context: Features describing the logged context.
            action: Features describing the action taken by the logging policy.
            reward: The reward that was revealed when the logged action was taken.
            probability: The probability that the logged action was taken. That is P(action|context,actions,logging policy).
            **kwargs: Any additional information.
        """

        self['context'] = context
        self['action']  = action
        self['reward']  = reward

        if probability is not None:
            self['probability'] = probability

        if kwargs: self.update(kwargs)
