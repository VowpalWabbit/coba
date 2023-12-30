from typing import Any, Mapping, Sequence

from coba.exceptions import CobaException
from coba.utilities import PackageChecker
from coba.primitives import Learner, Context, Action, Actions, Prob, PMF
from coba.encodings import InteractionsEncoder

class LinUCBLearner(Learner):
    """A contextual bandit learner using upper confidence bounds to explore.

    This is an implementation of the Chu et al. (2011) LinUCB algorithm. The 
    `Sherman-Morrison formula`__ is utilized to iteratively calculate the 
    inversion matrix. Expected reward is represented as a linear function 
    of context and action features.

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
                means that actions will be selected based solely on the estimated uper bound for each action (i.e.,
                we will always take actions that have the largest upper bound on their point estimate).
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

    def _initialize(self,context,action) -> None:
        if isinstance(action, dict) or isinstance(context, dict):
            raise CobaException("Sparse data cannot be handled by this implementation at this time.")

        if not context:
            self._X_encoder = InteractionsEncoder(list(set(filter(None,[ f.replace('x','') if isinstance(f,str) else f for f in self._X ]))))

        d  = len(self._X_encoder.encode(x=context or [],a=action))
        np = __import__('numpy')

        self._theta = np.zeros(d)
        self._A_inv = np.identity(d)
        self._np    = np

    def score(self, context: Context, actions: Actions, action: Action) -> Prob:
        return self.predict(context,actions)[actions.index(action)]

    def predict(self, context: Context, actions: Actions) -> PMF:
        if self._A_inv is None: self._initialize(context,actions[0])
        np = self._np

        context = context or []
        features = np.array([self._X_encoder.encode(x=context,a=action) for action in actions]).T

        point_estimate = self._theta @ features
        point_bounds   = np.diagonal(features.T @ self._A_inv @ features)

        action_values = point_estimate + self._alpha*np.sqrt(point_bounds)
        max_indexes   = np.where(action_values == np.amax(action_values))[0]

        return [int(ind in max_indexes)/len(max_indexes) for ind in range(len(actions))]

    def learn(self, context: Context, action: Action, reward: float, probability: float) -> None:
        if self._A_inv is None: self._initialize(context,action)

        np = self._np

        context = context or []
        features = np.array(self._X_encoder.encode(x=context,a=action)).T

        r = self._theta @ features
        w = self._A_inv @ features
        v = w           @ features

        self._A_inv = self._A_inv - np.outer(w,w)/(1+v)
        self._theta = self._theta + (reward-r)/(1+v)*w
