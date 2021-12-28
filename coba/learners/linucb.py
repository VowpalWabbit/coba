from typing import Any, Dict, Sequence

from coba.exceptions import CobaException
from coba.utilities import PackageChecker
from coba.environments import Context, Action
from coba.encodings import InteractionsEncoder

from coba.learners.primitives import Probs, Info, Learner

class LinUCBLearner(Learner):
    """This is an implementation of the Chu et al. (2011) LinUCB algorithm.

    This implementation uses the Sherman-Morrison formula to calculate the inversion matrix
    via iterative vector matrix multiplications. This implementation has computational complexity 
    that is linear with respect to feature count and memory complexity that is polynomial as the 
    inversion matrix is non-sparse and has size |phi|x|phi| (where |phi|
    are the number of features in the linear function). 

    Remarks:

        The Sherman-Morrsion implementation used below is given in long form at:
            https://research.navigating-the-edge.net/assets/publications/linucb_alternate_formulation.pdf

    References:
        Chu, Wei, Lihong Li, Lev Reyzin, and Robert Schapire. "Contextual bandits 
        with linear payoff functions." In Proceedings of the Fourteenth International 
        Conference on Artificial Intelligence and Statistics, pp. 208-214. JMLR Workshop 
        and Conference Proceedings, 2011.
    """

    def __init__(self, alpha: float = 0.2, X: Sequence[str] = ['a', 'ax']) -> None:
        """Instantiate a LinUCBLearner.

        Args:
            alpha: This parameter controls the exploration rate of the algorithm. A value of 0 will cause actions 
                to be selected based on the current best point estimate (i.e., no exploration) while a value of inf
                means that actions will be selected based solely on the bounds of the action point estimates (i.e., 
                we will always take actions that have the largest bound on their point estimate).
            interactions: Feature set interactions to use when calculating action value estimates. Context features
                are indicated by x's while action features are indicated by a's. For example, xaa means to cross the 
                features between context and actions and actions.
        """
        PackageChecker.numpy("LinUCBLearner.__init__")

        self._alpha = alpha

        self._X = X
        self._X_encoder = InteractionsEncoder(X)

        self._theta = None
        self._A_inv = None

    @property
    def params(self) -> Dict[str, Any]:
        """The parameters of the learner.

        See the base class for more information.
        """
        return {'family': 'LinUCB', 'alpha': self._alpha, 'X': self._X}

    def predict(self, context: Context, actions: Sequence[Action]) -> Probs:
        """Determine a PMF with which to select the given actions.

        Args:
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The probability of taking each action. See the base class for more information.
        """
        import numpy as np #type: ignore

        if isinstance(actions[0], dict) or isinstance(context, dict):
            raise CobaException("Sparse data cannot be handled by this algorithm.")

        features: np.ndarray = np.array([self._X_encoder.encode(x=context,a=action) for action in actions]).T

        if(self._A_inv is None):
            self._theta = np.zeros(features.shape[0])
            self._A_inv = np.identity(features.shape[0])

        point_estimate = self._theta @ features
        point_bounds   = np.diagonal(features.T @ self._A_inv @ features)

        action_values = point_estimate + self._alpha*np.sqrt(point_bounds)
        max_indexes   = np.where(action_values == np.amax(action_values))[0]

        return [ int(ind in max_indexes)/len(max_indexes) for ind in range(len(actions))]

    def learn(self, context: Context, action: Action, reward: float, probability: float, info: Info) -> None:
        """Learn from the given interaction.

        Args:
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
            probability: The probability with which the given action was selected.
            info: Optional information provided during prediction step for use in learning.
        """
        import numpy as np #type: ignore

        if isinstance(action, dict) or isinstance(context, dict):
            raise CobaException("Sparse data cannot be handled by this algorithm.")

        features = np.array(self._X_encoder.encode(x=context,a=action)).T

        if(self._A_inv is None):
            self._theta = np.zeros((features.shape[0]))
            self._A_inv = np.identity(features.shape[0])

        r = self._theta @ features
        w = self._A_inv @ features
        v = features    @ w

        self._A_inv = self._A_inv - np.outer(w,w)/(1+v)
        self._theta = self._theta + (reward-r)/(1+v) * w
