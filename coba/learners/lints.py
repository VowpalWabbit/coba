from typing import Any, Mapping, Sequence, Tuple

from coba.exceptions import CobaException
from coba.utilities import PackageChecker
from coba.primitives import Learner, Context, Action, Actions, Prob
from coba.encodings import InteractionsEncoder
from coba.learners.utilities import PMFPredictor

class LinTSLearner(Learner):
    """A contextual bandit learner using Thompson Sampling for exploration.

    This is an implementation of the Agrawal et al. (2013) Thompson Sapmling
    algorithm. The `Sherman-Morrison formula`__ is utilized to iteratively
    calculate the inversion matrix. Expected reward is represented as a
    linear function of context and action features.

    Remarks:
        A small note on the stability of the Sherman-Morrison formula can be found `here`__.

    References:
        Agrawal, Shipra, and Navin Goyal. "Thompson sampling for contextual bandits with
        linear payoffs." International conference on machine learning. PMLR, 2013.

    __ https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
    __ https://scicomp.stackexchange.com/q/20386/46891
    """

    def __init__(self, v: float = 1, features: Sequence[str] = [1, 'a', 'ax'], seed: int = 1) -> None:
        """Instantiate a LinUCBLearner.

        Args:
            v: Modify the exploration rate of the algorithm. A value of 0 will not explore
                while a value of `inf` will explores uniformly forever. The appropriate setting
                of v will depend to some degree on the scale of given feature vectors and rewards.
            features: Feature set interactions to use when calculating action value estimates.
                Context features are indicated by x's while action features are indicated by
                a's. For example, xaa means to cross context and action and action features.
            seed: A seed for a random number generation.
        """
        PackageChecker.numpy("LinTSLearner")

        self._X         = features
        self._X_encoder = InteractionsEncoder(features)

        self._B_inv  = None
        self._mu_hat = None
        self._v      = v
        self._pred   = PMFPredictor(self._pmf,seed)

    @property
    def params(self) -> Mapping[str, Any]:
        return {'family': 'LinTS', 'v': self._v, 'features': self._X, 'seed': self._pred.seed}

    def _initialize(self, context, action) -> None:
        if isinstance(action, dict) or isinstance(context, dict):
            raise CobaException("Sparse data cannot be handled by this implementation at this time.")

        if not context:
            self._X_encoder = InteractionsEncoder(list(set(filter(None,[ f.replace('x','') if isinstance(f,str) else f for f in self._X ]))))

        d  = len(self._X_encoder.encode(x=context or [],a=action))
        np = __import__('numpy')

        self._np     = np
        self._nrng   = np.random.default_rng(1)
        self._mu_hat = np.zeros(d)
        self._B_inv  = np.identity(d)

    def _pmf(self, context, actions):
        if self._B_inv is None: self._initialize(context,actions[0])

        np = self._np

        context  = context or []
        features = np.array([self._X_encoder.encode(x=context,a=action) for action in actions])

        if self._v == 0:
            mu_tilde = self._mu_hat
        else:
            mu_tilde = self._nrng.multivariate_normal(self._mu_hat, self._v*self._B_inv, method='cholesky')

        point_estimates = mu_tilde @ features.T
        max_indexes     = np.where(point_estimates.round(5) == np.amax(point_estimates).round(5))[0]

        return [int(ind in max_indexes)/len(max_indexes) for ind in range(len(actions))]

    def score(self, context: 'Context', actions: 'Actions', action: 'Action') -> 'Prob':
        return self._pred.score(context,actions,action)

    def predict(self, context: 'Context', actions: 'Actions') -> Tuple['Action','Prob']:
        return self._pred.predict(context,actions)

    def learn(self, context: 'Context', action: 'Action', reward: float, probability: float) -> None:
        if self._B_inv is None: self._initialize(context,action)

        np = self._np

        context  = context or []
        features = np.array(self._X_encoder.encode(x=context,a=action)).T

        r = self._mu_hat @ features
        w = self._B_inv  @ features
        v = w            @ features

        self._mu_hat = self._mu_hat + (reward-r)/(1+v)*w
        self._B_inv  = self._B_inv - np.outer(w,w)/(1+v)
