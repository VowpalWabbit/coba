
from typing import Any, Sequence, Union, Tuple, Mapping, Type

from coba.exceptions import CobaException
from coba.primitives import Context, Action, Actions
from coba.primitives import Dense, Sparse, HashableDense, HashableSparse

Prob   = float
PMF    = Sequence[Prob]
kwargs = Mapping[str,Any]

Prediction = Union[
    PMF,
    Action,
    Tuple[Action,Prob],
    Tuple[PMF        , kwargs],
    Tuple[Action     , kwargs],
    Tuple[Action,Prob, kwargs],
]

class Learner:
    """The Learner interface for contextual bandit learning."""

    @property
    def params(self) -> Mapping[str,Any]:
        """Parameters describing the learner (used for descriptive purposes only).

        Remarks:
            These will become columns in the learners table of experiment results.
        """
        return {}

    def score(self, context: Context, actions: Actions, action: Action) -> Prob:
        """Propensity score a given action (or all actions if action is None) in the context.

        Args:
            context: The current context. It will either be None (multi-armed bandit),
                a value (a single feature), a sequence of values (dense features), or a
                dictionary (sparse features).
            actions: The current set of actions that can be chosen in the given context.
                Each action will either be a value (a single feature), a sequence of values
                (dense features), or a dictionary (sparse features).
            action: The action to propensity score.

        Returns:
            The propensity score for the given action.
        """
        raise CobaException((
            "The `score` interface has not been implemented for this learner."
        ))

    def predict(self, context: Context, actions: Actions) -> Prediction:
        """Predict which action to take in the context.

        Args:
            context: The current context. It will either be None (multi-armed bandit),
                a value (a single feature), a sequence of values (dense features), or a
                dictionary (sparse features).
            actions: The current set of actions to choose from in the given context.
                Each action will either be a value (a single feature), a sequence of values
                (dense features), or a dictionary (sparse features).

        Returns:
            A Prediction. Several prediction formats are supported. See the type-hint for these.
        """
        raise CobaException((
            "The `predict` interface has not been implemented for this learner."
        ))

    def learn(self, context: Context, action: Action, reward: float, probability: float, **kwargs) -> None:
        """Learn about the action taken in the context.

        Args:
            context: The context in which the action was taken.
            action: The action that was taken.
            reward: The reward for the given context and action (feedback for IGL problems).
            probability: The probability the given action was taken.
            **kwargs: Optional information returned during prediction.
        """
        raise CobaException((
            "The `learn` interface has not been implemented for this learner."
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

    def new_learn(self,c,a,r,p,**kwargs):
        old_learn(self,make_hashable(c),make_hashable(a),r,p,**kwargs)

    cls.predict = new_predict
    cls.learn   = new_learn

    return cls
