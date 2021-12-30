from abc import ABC, abstractmethod
from numbers import Number
from typing import Any, Sequence, Dict, Union, Tuple

from coba.environments import Context, Action

Info  = Any
Probs = Sequence[float]

class Learner(ABC):
    """The Learner interface."""

    @property
    @abstractmethod
    def params(self) -> Dict[str,Any]:
        """Parameters describing the learner (used for descriptive purposes only)."""
        ...

    @abstractmethod
    def predict(self, context: Context, actions: Sequence[Action]) -> Union[Probs,Tuple[Probs,Info]]:
        """Predict which action to take in the context.

        Args:
            context: The current context. It will either be None (multi-armed bandit), 
                a value (a single feature) a hashable tuple (dense context), or a 
                hashable dictionary (sparse context).
            actions: The current set of actions to choose from in the given context. 
                Each action will either be a value (a single feature), a hashable tuple 
                (dense context), or a hashable dictionary (sparse context)..
        
        Returns:
            A PMF over the actions and, optionally, an information object to use when learning.
        """
        ...

    @abstractmethod
    def learn(self, context: Context, action: Action, reward: float, probability: float, info: Info) -> None:
        """Learn about the action taken in the context.

        Args:
            context: The context in which the action was taken.
            action: The action that was taken.
            reward: The reward received for taking the action in the context.
            probability: The probability the given action was taken.
            info: Optional information returned by the prediction method.
        """
        ...

class SafeLearner(Learner):
    """A wrapper for learner-likes that guarantees interface consistency."""

    def __init__(self, learner: Learner) -> None:
        """Instantiate a SafeLearner.
        
        Args:
            learner: The learner we wish to make sure has the expected interface
        """
        
        self._learner = learner if not isinstance(learner, SafeLearner) else learner._learner

    @property
    def full_name(self) -> str:
        """A user-friendly name created from a learner's params for reporting purposes."""

        params = dict(self.params)
        family = params.pop("family")

        if len(params) > 0:
            return f"{family}({','.join(f'{k}={v}' for k,v in params.items())})"
        else:
            return family

    @property
    def params(self) -> Dict[str, Any]:
        try:
            params = self._learner.params
        except AttributeError:
            params = {}

        if "family" not in params:
            params["family"] = self._learner.__class__.__name__

        return params

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

    def learn(self, context: Context, action: Action, reward: float, probability:float, info: Info) -> None:
        self._learner.learn(context, action, reward, probability, info)

    def __str__(self) -> str:
        return self.full_name
