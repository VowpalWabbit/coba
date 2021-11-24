"""The expected interface for all learner implementations."""

from abc import ABC, abstractmethod
from numbers import Number
from typing import Any, Sequence, Dict, Union, Tuple, Optional

from coba.environments import Context, Action

Info = Any
Probs = Sequence[float]

class Learner(ABC):
    """The interface for Learner implementations."""

    @property
    @abstractmethod
    def params(self) -> Dict[str,Any]:
        """The parameters used to initialize the learner.

        This value is used for descriptive purposes when creating experiment results.
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

class SafeLearner(Learner):

        @property
        def params(self) -> Dict[str, Any]:
            try:
                params = self._learner.params
            except AttributeError:
                params = {}

            if "family" not in params:
                params["family"] = self._learner.__class__.__name__

            return params

        @property
        def full_name(self) -> str:
            params = dict(self.params)
            family = params.pop("family")

            if len(params) > 0:
                return f"{family}({','.join(f'{k}={v}' for k,v in params.items())})"
            else:
                return family

        def __init__(self, learner: Learner) -> None:
            
            self._learner = learner if not isinstance(learner, SafeLearner) else learner._learner

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
            return self._learner.learn(context, action, reward, probability, info) or {}
