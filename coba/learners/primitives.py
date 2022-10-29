from collections import abc
from warnings import warn
from math import isclose
from abc import ABC, abstractmethod
from typing import Any, Sequence, Union, Tuple, Callable, Mapping

from coba.random import CobaRandom
from coba.environments import Context, Action, Reward, Feedback, Actions

kwargs = Mapping[str,Any]
Score  = float
PMF    = Sequence[float]
PDF    = Callable[[Action],float]

class Probs(list):
    pass

class ActionScore(tuple):
    def __new__(self, action: Action, score: Score):
        return tuple.__new__(ActionScore, (action, score))

Prediction = Union[
    PMF,
    PDF,
    Probs,
    ActionScore,
    Tuple[PMF         , kwargs],
    Tuple[PDF         , kwargs],
    Tuple[Action,Score, kwargs],
    Tuple[ActionScore , kwargs],
]

class CbLearner(ABC):
    """The Learner interface for contextual bandit learning."""

    @property
    def params(self) -> Mapping[str,Any]: # pragma: no cover
        """Parameters describing the learner (used for descriptive purposes only).

        Remarks:
            These will become columns in the learners table of experiment results.
        """
        return {}

    @abstractmethod
    def predict(self, context: Context, actions: Sequence[Action]) -> Prediction:
        """Predict which action to take in the context.

        Args:
            context: The current context. It will either be None (multi-armed bandit),
                a value (a single feature) a hashable tuple (dense context), or a
                hashable dictionary (sparse context).
            actions: The current set of actions to choose from in the given context.
                Each action will either be a value (a single feature), a hashable tuple
                (dense context), or a hashable dictionary (sparse context)..

        Returns:
            A Prediction. Several prediction formats are supported. See the type-hint for these.
        """
        ...

    @abstractmethod
    def learn(self, context: Context, actions: Actions, action: Action, reward: Reward, probability: float, **kwargs:Any) -> None:
        """Learn about the action taken in the context.

        Args:
            context: The context in which the action was taken.
            actions: The set of actions chosen from.
            action: The action that was chosen.
            reward: The reward received for the chosen action.
            probability: The probability the given action was taken.
            **kwargs: Optional information returned with the prediction.
        """
        ...

class IgLearner(ABC):
    """The Learner interface for interaction grounded learning."""

    @property
    def params(self) -> Mapping[str,Any]: # pragma: no cover
        """Parameters describing the learner (used for descriptive purposes only).

        Remarks:
            These will become columns in the learners table of experiment results.
        """
        return {}

    @abstractmethod
    def predict(self, context: Context, actions: Actions) -> Prediction:
        ...

    @abstractmethod
    def learn(self, context: Context, actions: Actions, action: Action, feedback: Feedback, probability: float, **kwargs:Any):
        ...

class SafeLearner:
    """A wrapper for learner-likes that guarantees interface consistency."""

    def __init__(self, learner: CbLearner) -> None:
        """Instantiate a SafeLearner.

        Args:
            learner: The learner we wish to make sure has the expected interface
        """

        self._learner   = learner if not isinstance(learner, SafeLearner) else learner._learner
        self._rng       = CobaRandom(1)
        self._pred_type = None
        self._with_info = None

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
    def params(self) -> Mapping[str, Any]:
        try:
            params = self._learner.params
            params = params if isinstance(params,dict) else params()
        except AttributeError:
            params = {}

        if "family" not in params:
            params["family"] = self._learner.__class__.__name__

        return params

    def predict(self, context: Context, actions: Actions) -> Tuple[Action,Score,kwargs]:
        pred = self._learner.predict(context, actions)

        if self._with_info is None: self._with_info = self._get_with_info(pred)
        if self._pred_type is None: self._pred_type = self._get_pred_type(actions,pred)
        info = pred[ -1] if self._with_info else {}
        pred = pred[:-1] if self._with_info else pred

        if self._pred_type == 0 or self._pred_type == 1:
            if self._with_info: pred = pred[0]
            if self._pred_type == 0: pred = list(map(pred,actions))
            assert len(pred) == len(actions), "The learner returned an invalid number of probabilities for the actions"
            assert isclose(sum(pred), 1, abs_tol=.001), "The learner returned a pmf which didn't sum to one."
            pred = self._rng.choice(list(zip(actions,pred)), pred)

        return pred[0], pred[1], info

    def learn(self, context, actions, action, reward, probability, **kwargs) -> None:
        self._learner.learn(context, actions, action, reward, probability, **kwargs)

    def _get_pred_type(self, actions:Actions, pred: Prediction) -> int:
        #This isn't air-tight. I think it will do for now though.
        #0 == PDF; 1 == PMF; 2 == Tuple[Action,Score]

        parse_err_msg = "The given prediction had an unrecognized format."
        parse_wrn_msg = ("We were unable to infer the given prediction format." 
            " We made our best guess, but to make sure we don't make a mistake"
            " we suggest using either coba.learners.{Probs or ActionScore} to be"
            " explicit.")

        if self._get_with_info(pred):
            assert len(pred) in [2,3], parse_err_msg
            if len(pred) == 3: return 2
            if callable(pred[0]): return 0
            if isinstance(pred[0],Probs): return 1
            if isinstance(pred[0],ActionScore): return 2
            return 1

        else:
            assert callable(pred) or isinstance(pred,abc.Sequence), parse_err_msg

            if callable(pred)                     : return 0
            if len(pred) !=2                      : return 1
            if pred[0] not in actions             : return 1
            if isinstance(pred,Probs)             : return 1
            if isinstance(pred,ActionScore)       : return 2
            if len(actions) !=2                   : return 2
            if not isinstance(pred[0],(int,float)): return 2
            if pred[0] in actions and sum(pred)!=1: return 2

            warn(parse_wrn_msg)
            return 2 if isinstance(pred,tuple) else 1

    def _get_with_info(self, prediction: Prediction) -> bool:
        try: 
            return isinstance(prediction[-1],dict)
        except:
            return False

    def __str__(self) -> str:
        return self.full_name
