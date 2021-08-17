"""The expected interface for all learner implementations."""

from abc import ABC, abstractmethod
from numbers import Number
from typing import Any, Sequence, Dict, Union, Tuple, Optional

from coba.simulations import Context, Action

Info = Any
Probs = Sequence[float]

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

    @abstractmethod
    def predict(self, context: Context, actions: Sequence[Action]) -> Union[Probs,Tuple[Probs,Info]]:
        """Determine a PMF with which to select the given actions.

        Args:
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
            Either a sequence of probabilities indicating the probability of taking each action
            or a tuple with a sequence of probabliities and optional information for learning.
        """
        ...

    @abstractmethod
    def learn(self, context: Context, action: Action, reward: float, probability: float, info: Info) -> Optional[Dict[str,Any]]:
        """Learn about the result of an action that was taken in a context.

        Args:
            context: The context in which the action was taken.
            action: The action that was selected to play and observe its reward.
            reward: The reward received for taking the given action in the given context.
            probability: The probability with which the given action was selected.
            info: Optional information provided during prediction step for use in learning.
        Returns:
            An optional dictionary which will be passed to the interactions table in evaluation result.
        """
        ...
    
    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        """An optional method that can be overridden to make Learners picklable."""
        return super().__reduce__()

class FixedLearner(Learner):
    """A Learner implementation that selects actions according to a fixed distribution and learns nothing."""

    @property
    def family(self) -> str:
        """The family of the learner.

        See the base class for more information
        """  
        return "fixed"

    @property
    def params(self) -> Dict[str, Any]:
        """The parameters of the learner.
        
        See the base class for more information
        """
        return {}

    def __init__(self, fixed_pmf: Sequence[float]) -> None:
        
        assert round(sum(fixed_pmf),3) == 1, "The given pmf must sum to one to be a valid pmf."
        assert all([p >= 0 for p in fixed_pmf]), "All given probabilities of the pmf must be greater than or equal to 0."

        self._fixed_pmf = fixed_pmf

    def predict(self, context: Context, actions: Sequence[Action]) -> Probs:
        """Choose an action from the action set.
        
        Args:
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The probability of taking each action. See the base class for more information.
        """
        return self._fixed_pmf

    def learn(self, context: Context, action: Action, reward: float, probability: float, info: Info) -> None:
        """Learns nothing.

        Args:
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
            probability: The probability with which the given action was selected.
            info: Optional information provided during prediction step for use in learning.
        """
        pass

class RandomLearner(Learner):
    """A Learner implementation that selects an action at random and learns nothing."""

    @property
    def family(self) -> str:
        """The family of the learner.

        See the base class for more information
        """  
        return "random"

    @property
    def params(self) -> Dict[str, Any]:
        """The parameters of the learner.
        
        See the base class for more information
        """
        return {}

    def predict(self, context: Context, actions: Sequence[Action]) -> Probs:
        """Choose a random action from the action set.
        
        Args:
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The probability of taking each action. See the base class for more information.
        """
        return [1/len(actions)] * len(actions)

    def learn(self, context: Context, action: Action, reward: float, probability: float, info: Info) -> None:
        """Learns nothing.

        Args:
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
            probability: The probability with which the given action was selected.
            info: Optional information provided during prediction step for use in learning.
        """
        pass

class SafeLearner(Learner):

        @property
        def family(self) -> str:
            try:
                return self._learner.family
            except AttributeError:
                return self._learner.__class__.__name__

        @property
        def params(self) -> Dict[str, Any]:
            try:
                return self._learner.params
            except AttributeError:
                return {}

        def __init__(self, learner: Learner) -> None:
            self._learner = learner

        def predict(self, context: Context, actions: Sequence[Action]) -> Tuple[Probs, Info]:
            predict = self._learner.predict(context, actions)

            predict_has_no_info = len(predict) != 2 or isinstance(predict[0],Number)

            if predict_has_no_info:
                info    = None
                predict = predict
            else:
                info    = predict[1]
                predict = predict[0]

            assert len(predict) == len(actions), "The learner returned an invalid number of probabilities for the actions"
            assert round(sum(predict),2) == 1 , "The learner returned a pmf which didn't sum to one."

            return (predict,info)

        def learn(self, context: Context, action: Action, reward: float, probability:float, info: Info) -> Optional[Dict[str,Any]]:
            return self._learner.learn(context, action, reward, probability, info)
