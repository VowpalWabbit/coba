import collections
import time

from typing import Any, Dict, Sequence

from coba.utilities import PackageChecker
from coba.simulations import Context, Action
from coba.encodings import InteractionTermsEncoder
from coba.learners.core import Info, Learner

class LinUCBLearner(Learner):
    """A learner using the LinUCB algorithm from "Contextual Bandits with Linear Payoff Functions" by Wei Chu et al."""

    @property
    def params(self) -> Dict[str, Any]:
        """The parameters of the learner.

        See the base class for more information.
        """
        dict = {'family': 'linUCB', 'alpha': self._alpha, 'interactions': self._interactions}
        return dict

    def __init__(self, *, alpha: float, interactions: Sequence[str] = ['a', 'ax'], timeit: bool = False) -> None:
        """Instantiate a linUCBLearner.
        Args:
            alpha: number of standard deviations
            interactions: the set of interactions the learner will use. x refers to context and a refers to actions, 
                e.g. xaa would mean interactions between context and actions and actions. 
        """
        PackageChecker.numpy("linUCBLearner.__init__")

        self._A            = None
        self._b            = None
        self._alpha        = alpha
        self._times        = [0.,0.]
        self._i            = 0
        self._timeit       = timeit

        self._interactions = interactions
        self._interactions_encoder = InteractionTermsEncoder(interactions)

    def predict(self, context: Context, actions: Sequence[Action]) -> Sequence[float]:
        """Determine a PMF with which to select the given actions.

        Args:
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The probability of taking each action. See the base class for more information.
        """
        import numpy as np #type: ignore

        self._i += 1

        self._d   = len(actions[0]) if isinstance(actions[0], collections.Sequence) else 1
        is_sparse = isinstance(actions[0], dict) or isinstance(context, dict)

        if is_sparse:
            raise Exception("Sparse data cannot be handled by this algorithm.")

        features: np.ndarray = self._featurize(context, actions)

        if(self._A is None):
            self._A = np.identity(features.shape[0])
            self._b = np.zeros((features.shape[0], 1))
        
        start_predict = time.time()

        A_inv = np.linalg.inv(self._A)

        self._times[0] += time.time() - start_predict

        theta = np.dot(A_inv, self._b)
        
        term_one = np.zeros([len(actions),1])
        term_two = np.zeros([len(actions),1])

        for i in range(len(actions)):
            term_one[i] = theta.T @ features[:,i]
            term_two[i] = self._alpha * np.sqrt(features[:,i].T @ A_inv @ features[:,i])

        action_values = term_one + term_two

        # if (self._i-1) % 100 == 0 and self._timeit:
        #     print(self._times[0]/(self._i+1))
        #     print(self._times[1]/(self._i+1))

        max_indexes = np.where(action_values == np.amax(action_values))[0]
        
        return [1/len(max_indexes) if ind in max_indexes else 0 for ind in range(len(actions))]

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

        learn_start = time.time()
        
        features: np.ndarray = self._featurize(context, [action])

        self._A = self._A + features@features.T
        self._b = self._b + features*reward 

        self._times[1] += time.time() - learn_start

    def _featurize(self, context, actions):
        import numpy as np
        
        features = []
        
        for action in actions:
            features.append(self._interactions_encoder.encode(x=context,a=action))
        
        return np.array(features).T