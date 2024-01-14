import math

from collections import defaultdict
from typing import Any, Mapping, Optional, Hashable, Type, Tuple

from coba.random import CobaRandom
from coba.primitives import Context, Action, Actions, Prob, Pmf
from coba.statistics import OnlineVariance
from coba.primitives import Dense, Sparse, HashableDense, HashableSparse, Learner
from coba.learners.utilities import PMFPredictor

def requires_hashables(cls:Type[Learner]):

    def make_hashable(item):
        if isinstance(item,Dense): return HashableDense(item)
        if isinstance(item,Sparse): return HashableSparse(item)
        return item

    old_predict = cls.predict
    old_learn   = cls.learn

    def new_predict(self, context: 'Context', actions: 'Actions'):
        return old_predict(self, make_hashable(context), list(map(make_hashable,actions)))

    def new_learn(self,context: 'Context', action: 'Action', reward:float, probability: float):
        old_learn(self, make_hashable(context), make_hashable(action), reward, probability)

    cls.predict = new_predict
    cls.learn   = new_learn

    return cls

@requires_hashables
class EpsilonBanditLearner(Learner):
    """Select the greedy action with probability (1-epsilon)."""

    def __init__(self, epsilon: float=.05, seed:int = 1) -> None:
        """Instantiate an EpsilonBanditLearner.

        Args:
            epsilon: We explore with probability epsilon and exploit otherwise.
            seed: The seed used to select actions in predict.
        """
        self._epsilon = epsilon
        self._N: Mapping[Hashable, int            ] = defaultdict(int)
        self._Q: Mapping[Hashable, Optional[float]] = defaultdict(int)
        self._pred = PMFPredictor(self._pmf,seed)

    @property
    def params(self) -> Mapping[str, Any]:
        return {'family': 'epsilon_bandit', 'epsilon': self._epsilon, 'seed': self._pred.seed}

    def _pmf(self, context: 'Context', actions: 'Actions') -> 'Pmf':
        values      = [ self._Q[action] for action in actions ]
        max_value   = None if set(values) == {None} else max(v for v in values if v is not None)
        max_indexes = [i for i in range(len(values)) if values[i]==max_value]

        prob_selected_randomly = [1/len(actions) * self._epsilon] * len(actions)
        prob_selected_greedily = [ int(i in max_indexes)/len(max_indexes) * (1-self._epsilon) for i in range(len(actions))]

        return [p1+p2 for p1,p2 in zip(prob_selected_randomly,prob_selected_greedily)]

    def score(self, context: Context, actions: Actions, action: Action) -> Prob:
        return self._pred.score(context,actions,action)

    def predict(self, context: Context, actions: Actions) -> Tuple['Action','Prob']:
        return self._pred.predict(context,actions)

    def learn(self, context: 'Context', action: 'Action', reward: float, probability: float) -> None:
        alpha = 1/(self._N[action]+1)
        old_Q = self._Q[action] or 0.0

        self._Q[action] = (1-alpha) * old_Q + alpha * reward
        self._N[action] = self._N[action] + 1

@requires_hashables
class UcbBanditLearner(Learner):
    """Select the action with the highest upper confidence bound estimate.

    This algorithm is an implementation of Auer et al. (2002) UCB1-Tuned algorithm
    and requires that all rewards are in [0,1].

    References:
        Auer, Peter, Nicolo Cesa-Bianchi, and Paul Fischer. "Finite-time analysis of
        the multiarmed bandit problem." Machine learning 47.2-3 (2002): 235-256.
    """

    def __init__(self, seed: int = 1):
        """Instantiate a UcbBanditLearner.

        Args:
            seed: The seed used to select actions in predict.
        """
        #these variable names were selected for easier comparison with the original paper
        self._t: int = 0
        self._m: Mapping['Action', float         ] = {}
        self._s: Mapping['Action', int           ] = {}
        self._v: Mapping['Action', OnlineVariance] = {}
        self._pred = PMFPredictor(self._pmf,seed)

    @property
    def params(self) -> Mapping[str, Any]:
        return {'family': 'UCB_bandit', 'seed': self._pred.seed }

    def _pmf(self, context: 'Context', actions: 'Actions') -> 'Pmf':
        never_observed_actions = set(actions) - self._m.keys()

        if never_observed_actions:
            max_actions = never_observed_actions
        else:
            values      = [ self._m[a] + self._Avg_R_UCB(a) for a in actions ]
            max_value   = max(values)
            max_actions = [ a for a,v in zip(actions,values) if v==max_value ]

        return [int(action in max_actions)/len(max_actions) for action in actions]

    def score(self, context: Context, actions: Actions, action: Action) -> Prob:
        return self._pred.score(context,actions,action)

    def predict(self, context: Context, actions: Actions) -> Tuple['Action','Prob']:
        return self._pred.predict(context,actions)

    def learn(self, context: 'Context', action: 'Action', reward: float, probability: float) -> None:
        self._t += 1

        assert 0 <= reward and reward <= 1, "This algorithm assumes that reward has support in [0,1]."

        if action not in self._m:
            self._m[action] = reward
            self._s[action] = 1
            self._v[action] = OnlineVariance()

        else:
            self._m[action] = (1-1/self._s[action]) * self._m[action] + 1/self._s[action] * reward
            self._s[action] += 1
            self._v[action].update(reward)

    def _Avg_R_UCB(self, action: 'Action') -> float:
        """Produce the estimated upper confidence bound (UCB) for E[R|A].

        Args:
            action: The action for which we want to retrieve UCB for E[R|A].

        Returns:
            The estimated UCB for E[R|A].

        Remarks:
            See the beginning of section 4 in the algorithm's paper for this equation.
        """
        ln = math.log; n = self._t; n_j = self._s[action]; V_j = self._Var_R_UCB(action)
        return math.sqrt(ln(n)/n_j * min(1/4,V_j))

    def _Var_R_UCB(self, action: 'Action') -> float:
        """Produce the upper confidence bound (UCB) for Var[R|A].

        Args:
            action: The action for which we want to retrieve UCB for Var[R|A].

        Returns:
            The estimated UCB for Var[R|A].

        Remarks:
            See the beginning of section 4 in the algorithm's paper for this equation.
        """
        ln = math.log; t = self._t; s = self._s[action]; var = self._v[action].variance

        return var + math.sqrt(2*ln(t)/s)

class FixedLearner(Learner):
    """Select actions from a fixed distribution and learn nothing."""

    def __init__(self, pmf: 'Pmf', seed: int = 1) -> None:
        """Instantiate a FixedLearner.

        Args:
            pmf: A PMF whose values are the probability of taking each action.
            seed: The seed used to select actions in predict.
        """
        assert round(sum(pmf),3) == 1, "The given pmf must sum to one to be a valid pmf."
        assert all([p >= 0 for p in pmf]), "All given probabilities of the pmf must be greater than or equal to 0."
        self._fpmf = pmf
        self._pred = PMFPredictor(self._pmf,seed)

    @property
    def params(self) -> Mapping[str, Any]:
        return {'family': 'fixed', 'seed': self._pred.seed}

    def _pmf(self, context: 'Context', actions: 'Actions') -> 'Pmf':
        return self._fpmf

    def score(self, context: Context, actions: Actions, action: Action) -> Prob:
        return self._pred.score(context,actions,action)

    def predict(self, context: Context, actions: Actions) -> Tuple['Action','Prob']:
        return self._pred.predict(context,actions)

    def learn(self, context: 'Context', action: 'Action', reward: float, prob: float) -> None:
        pass

class RandomLearner(Learner):
    """Select actions from a uniform distribution and learn nothing."""

    def __init__(self, seed:int = 1):
        """Instantiate a RandomLearner.

        Args:
            seed: The seed used to select actions in predict.
        """
        self._rng = CobaRandom(seed)

    @property
    def params(self) -> Mapping[str, Any]:
        return {'family':'random', 'seed': self._rng.seed}

    def score(self, context: 'Context', actions: 'Actions', action: 'Action') -> 'Prob':
        return 1/len(actions)

    def predict(self, context: 'Context', actions: 'Actions') -> Tuple['Action','Prob']:
        return self._rng.choicew(actions)

    def learn(self, context: 'Context', action: 'Action', reward: float, probability: float) -> None:
        pass
