"""The expected interface for all learner implementations."""

from abc import ABC, abstractmethod
from typing import Any, Sequence, Dict

from coba.simulations import Context, Action, Key

class Learner(ABC):
    """The interface for Learner implementations."""

    @property
    @abstractmethod
    def family(self) -> str:
        """The family of the learner.

        This value is used for descriptive purposes only when creating benchmark results.
        """
        ...

    @property
    @abstractmethod
    def params(self) -> Dict[str,Any]:
        """The parameters used to initialize the learner.

        This value is used for descriptive purposes only when creating benchmark results.
        """
        ...

    def init(self) -> None:
        """An optional initialization method called once after pickling."""        
        pass

    @abstractmethod
    def predict(self, key: Key, context: Context, actions: Sequence[Action]) -> Sequence[float]:
        """Determine a PMF with which to select the given actions.

        Args:
            key: A unique identifier for the interaction that the observed reward 
                came from. This identifier allows learners to share information
                between the choose and learn methods while still keeping the overall 
                learner interface consistent and clean.
            context: The current context. This argument will be None when playing 
                a multi-armed bandit simulation and will contain context features 
                when playing a contextual bandit simulation. Context features could 
                be an individual number (e.g. 1.34), a string (e.g., "hot"), or a 
                tuple of strings and numbers (e.g., (1.34, "hot")) depending on the 
                simulation being played.
            actions: The current set of actions to choose from in the given context. 
                Action sets can be lists of numbers (e.g., [1,2,3,4]), a list of 
                strings (e.g. ["high", "medium", "low"]), or a list of tuples such 
                as in the case of movie recommendations (e.g., [("action", "oscar"), 
                ("fantasy", "razzie")]).

        Returns:
            A sequence of probabilities indicating the probability for each action.
        """
        ...

    @abstractmethod
    def learn(self, key: Key, context: Context, action: Action, reward: float, probability: float) -> None:
        """Learn about the result of an action that was taken in a context.

        Args:
            key: A unique identifier for the interaction that the observed reward 
                came from. This identifier allows learners to share information
                between the choose and learn methods while still keeping the overall 
                learner interface consistent and clean.
            context: The current context. This argument will be None when playing 
                a multi-armed bandit simulation and will contain context features 
                when playing a contextual bandit simulation. Context features could 
                be an individual number (e.g. 1.34), a string (e.g., "hot"), or a 
                tuple of strings and numbers (e.g., (1.34, "hot")) depending on the 
                simulation being played.
            action: The action that was selected to play and observe its reward. 
                An Action can be an individual number (e.g., 2), a string (e.g. 
                "medium"), or a list of some combination of numbers or strings
                (e.g., ["action", "oscar"]).
            reward: The reward received for taking the given action in the given context.
            probability: The probability with wich the given action was selected.
        """
        ...