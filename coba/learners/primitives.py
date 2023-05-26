
from abc import ABC, abstractmethod
from typing import Any, Sequence, Union, Tuple, Callable, Mapping, Type

from coba.exceptions import CobaException
from coba.primitives import Context, Action, Actions
from coba.primitives import Dense, Sparse, HashableDense, HashableSparse

kwargs = Mapping[str,Any]
Prob   = float
PDF    = Callable[[Action],float]

class PMF(list):
    pass

class ActionProb(tuple):
    def __new__(self, action: Action, prob: Prob):
        return tuple.__new__(ActionProb, (action, prob))

Prediction = Union[
    PMF,
    ActionProb,
    Tuple[PMF        , kwargs],
    Tuple[ActionProb , kwargs],
    Tuple[Action,Prob, kwargs],
]

class Learner(ABC):
    """The Learner interface for contextual bandit learning."""

    @property
    def params(self) -> Mapping[str,Any]: # pragma: no cover
        """Parameters describing the learner (used for descriptive purposes only).

        Remarks:
            These will become columns in the learners table of experiment results.
        """
        return {}

    def request(self, context: Context, actions: Actions, request: Actions) -> Sequence[Prob]:
        """Request the probabilities for specific actions in the given context

        Args:
            context: The current context. It will either be None (multi-armed bandit),
                a value (a single feature) a hashable tuple (dense context), or a
                hashable dictionary (sparse context).
            actions: The current set of actions that can be chosen in the given context.
                Each action will either be a value (a single feature), a hashable tuple
                (dense context), or a hashable dictionary (sparse context).
            request: The requested action or set of action probabilities.

        Returns:
            The requested action probabilities (or densities if actions is continuous).
        """
        raise CobaException((
            "The `request` interface has not been implemented for this learner."
        ))

    def predict(self, context: Context, actions: Sequence[Action]) -> Prediction:
        """Predict which action to take in the context.

        Args:
            context: The current context. It will either be None (multi-armed bandit),
                a value (a single feature) a hashable tuple (dense context), or a
                hashable dictionary (sparse context).
            actions: The current set of actions to choose from in the given context.
                Each action will either be a value (a single feature), a hashable tuple
                (dense context), or a hashable dictionary (sparse context).

        Returns:
            A Prediction. Several prediction formats are supported. See the type-hint for these.
        """
        raise CobaException((
            "The `predict` interface has not been implemented for this learner."
        ))

    @abstractmethod
    def learn(self,
        context: Context,
        actions: Actions,
        action: Action,
        feedback: Union[float,Any],
        score: float,
        **kwargs) -> None:
        """Learn about the action taken in the context.

        Args:
            context: The context in which the action was taken.
            actions: The set of actions chosen from.
            action: Either the action that was chosen or the index of the action that was chosen.
            feedback: This will be reward for contextual bandit problems and feedback for IGL problems.
            score: This will be the probability for the action taken if a PMF/PDF is returned by predict.
                 It will be the score if an action-score pair is returned by predict. And it will be the
                 probability if off-policy learning is being performed on LoggedInteractions.
            **kwargs: Optional information returned with the prediction.
        """
        raise CobaException((
            "The `predict` interface has not been implemented for this learner."
        ))

def requires_hashables(cls:Type[Learner]):

    def make_hashable(item):
        if isinstance(item,Dense): return HashableDense(item)
        if isinstance(item,Sparse): return HashableSparse(item)
        return item

    old_predict = cls.predict
    old_learn   = cls.learn

    def new_predict(self,c,A):
        return old_predict(self,make_hashable(c),list(map(make_hashable,A)))

    def new_learn(self,c,A,a,r,p,**kwargs):
        old_learn(self,make_hashable(c),list(map(make_hashable,A)), make_hashable(a),r,p,**kwargs)

    cls.predict = new_predict
    cls.learn   = new_learn

    return cls
