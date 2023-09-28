from typing import Any, Mapping, Sequence

from coba.exceptions import CobaException
from coba.utilities import PackageChecker
from coba.primitives import Context, Action
from coba.encodings import InteractionsEncoder

from coba.learners.primitives import PMF, Learner, Actions, Prob

class LinUCBLearner(Learner):
    """A contextual bandit learner that represents expected reward as a
    linear function of context and action features. Exploration is carried
    out according to upper confidence bound estimates.

    This is an implementation of the Chu et al. (2011) LinUCB algorithm using the
    `Sherman-Morrison formula`__ to iteratively calculate the inversion matrix. This
    implementation's computational complexity is linear with respect to feature count.

    Remarks:
        The Sherman-Morrsion implementation used below is given in long form `here`__.

    References:
        Chu, Wei, Lihong Li, Lev Reyzin, and Robert Schapire. "Contextual bandits
        with linear payoff functions." In Proceedings of the Fourteenth International
        Conference on Artificial Intelligence and Statistics, pp. 208-214. JMLR Workshop
        and Conference Proceedings, 2011.

    __ https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
    __ https://research.navigating-the-edge.net/assets/publications/linucb_alternate_formulation.pdf
    """

    def __init__(self, alpha: float = 1, features: Sequence[str] = [1, 'a', 'ax']) -> None:
        """Instantiate a LinUCBLearner.

        Args:
            alpha: This parameter controls the exploration rate of the algorithm. A value of 0 will cause actions
                to be selected based on the current best point estimate (i.e., no exploration) while a value of inf
                means that actions will be selected based solely on the bounds of the action point estimates (i.e.,
                we will always take actions that have the largest bound on their point estimate).
            features: Feature set interactions to use when calculating action value estimates. Context features
                are indicated by x's while action features are indicated by a's. For example, xaa means to cross the
                features between context and actions and actions.
        """
        PackageChecker.numpy("LinUCBLearner")

        self._alpha = alpha

        self._X = features
        self._X_encoder = InteractionsEncoder(features)

        self._theta = None
        self._A_inv = None

    @property
    def params(self) -> Mapping[str, Any]:
        return {'family': 'LinUCB', 'alpha': self._alpha, 'features': self._X}

    def request(self, context: Context, actions: Actions, request: Actions) -> Sequence[Prob]:
        probs = self.predict(context,actions)
        return [ probs[actions.index(a)] for a in request ]

    def predict(self, context: Context, actions: Actions) -> PMF:
        import numpy as np

        if isinstance(actions[0], dict) or isinstance(context, dict):
            raise CobaException("Sparse data cannot be handled by this implementation at this time.")

        if not context:
            self._X_encoder = InteractionsEncoder(list(set(filter(None,[ f.replace('x','') if isinstance(f,str) else f for f in self._X ]))))

        context = context or []
        features = np.array([self._X_encoder.encode(x=context,a=action) for action in actions]).T

        if(self._A_inv is None):
            self._theta = np.zeros(features.shape[0])
            self._A_inv = np.identity(features.shape[0])

        point_estimate = self._theta @ features
        point_bounds   = np.diagonal(features.T @ self._A_inv @ features)

        action_values = point_estimate + self._alpha*np.sqrt(point_bounds)
        max_indexes   = np.where(action_values == np.amax(action_values))[0]

        return [int(ind in max_indexes)/len(max_indexes) for ind in range(len(actions))]

    def learn(self, context: Context, action: Action, reward: float, probability: float) -> None:
        import numpy as np

        if isinstance(action, dict) or isinstance(context, dict):
            raise CobaException("Sparse data cannot be handled by this algorithm.")

        if not context:
            self._X_encoder = InteractionsEncoder(list(set(filter(None,[ f.replace('x','') if isinstance(f,str) else f for f in self._X ]))))

        context = context or []
        features = np.array(self._X_encoder.encode(x=context,a=action)).T

        if(self._A_inv is None):
            self._theta = np.zeros((features.shape[0]))
            self._A_inv = np.identity(features.shape[0])

        r = self._theta @ features
        w = self._A_inv @ features
        v = features    @ w

        self._A_inv = self._A_inv - np.outer(w,w)/(1+v)
        self._theta = self._theta + (reward-r)/(1+v) * w
