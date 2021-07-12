import math

from coba.utilities import PackageChecker
from coba.simulations import Context, Action
from coba.learners.core import Learner, Key
from typing import Any, Dict, Sequence

from copy import deepcopy
from itertools import product

class RegCB(Learner):
    """A learner using the RegCB algorithm by Foster et al.
        and the online bin search implementation by Bietti et al. 
    
    References:
        Foster, Dylan, Alekh Agarwal, Miroslav Dudík, Haipeng Luo, and Robert Schapire.
        "Practical contextual bandits with regression oracles." In International 
        Conference on Machine Learning, pp. 1539-1548. PMLR, 2018.

        Bietti, Alberto, Alekh Agarwal, and John Langford.
        "A contextual bandit bake-off." arXiv preprint 
        arXiv:1802.04064 (2018).
        
    """

    @property
    def family(self) -> str:
        """The family of the learner.

        See the base class for more information
        """
        return f"RegCB"

    @property
    def params(self) -> Dict[str, Any]:
        """The parameters of the learner.

        See the base class for more information
        """
        dict = {'beta': self._beta, 'alpha': self._alpha, 'interactions': self._interactions}
        return dict

    def __init__(self, *, beta: float, alpha: float, seed:int=1, interactions: Sequence[str] = ['a', 'ax']) -> None:
        """Instantiate a RegCBLearner.
        Args:
            beta: square-loss tolerance
            alpha: confidence bounds precision
            seed: randomization of the regressor
            interactions: the set of interactions the learner will use. x refers to context and a refers to actions, 
                e.g. xaa would mean interactions between context and actions and actions. 
        """
        PackageChecker.numpy("RegCBLearner.__init__")
        PackageChecker.sklearn("RegCBLearner.__init__")
        import numpy as np
        from sklearn.linear_model import SGDRegressor
        from sklearn.feature_extraction import FeatureHasher
        
        self._beta = beta
        self._alpha = alpha
        self._iter = 0
        self._epoch = 0

        self._actionEpoch = None
        self._contextEpoch = []
        self._rewardEpoch = None

        self._actionHistory = []
        self._contextHistory = []
        self._rewardHistory = []

        self._core_model = SGDRegressor(eta0=1,power_t=0, random_state=np.random.RandomState(seed))

        self._time = [0,0,0,0]
        self._interactions = interactions
        self._terms        = []
        self._is_sparse = False
        self._h = FeatureHasher()

        for term in self._interactions:
            term = term.lower()
            x_num = term.count('x')
            a_num = term.count('a')
            
            if x_num + a_num != len(term):
                raise Exception("Letters other than x and a were passed for parameter interactions. Please remove other letters/characters.")
            
            self._terms.append((x_num, a_num))

    def predict(self, key: Key, context: Context, actions: Sequence[Action]) -> Sequence[float]:
        """Determine a PMF with which to select the given actions.

        Args:
            key: The key identifying the interaction we are choosing for.
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The probability of taking each action. See the base class for more information.
        """

        if (self._iter < 30):
            return [1/len(actions)] * len(actions)
        
        else:
            
            maxScore  = -float('inf')
            maxAction = None

            for action in actions:

                score = self.bin_search(self._featurize(context,action), len(actions))
                
                if score > maxScore:
                    maxAction = action
                    maxScore  = score
            
            return [int(action == maxAction) for action in actions]
    
    def bin_search(self, features, K_t) -> float:

        y_l = 2
        w   = 1

        f_l_a_w = deepcopy(self._core_model)
        f_l_a_w.partial_fit(features,[y_l],[w])

        f_x_t_a = self._core_model.predict(features)[0]
        s_l_a   = (f_l_a_w.predict(features)[0] - f_x_t_a) / w

        obj = lambda w: w*(f_x_t_a-y_l)**2 - w*(f_x_t_a+s_l_a*w-y_l)**2

        range = [0, (f_x_t_a-y_l)/(-s_l_a)]

        h = range[1]/2

        constraint = self._alpha * math.log(K_t)

        w_old = 0
        w_now = range[1]/2
        o     = obj(w_now)

        while abs(w_now-w_old) > (range[1])*(1/2)**30 or o >= constraint:
            w_diff = abs(w_now-w_old)
            w_old  = w_now
            if o < constraint:
                w_now += w_diff/2
            else:
                w_now -= w_diff/2
            o = obj(w_now)

        return f_x_t_a + s_l_a*w_now

    def learn(self, key: Key, context: Context, action: Action, reward: float, probability: float) -> None:
        """Learn from the given interaction.

        Args:
            key: The key identifying the interaction this observed reward came from.
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
            probability: The probability that the given action was taken.
        """
        self._core_model.partial_fit(self._featurize(context, action), [reward], [1])
        
        self._iter += 1

    def _featurize(self, context, action):
        import numpy as np #type: ignore

        feature_names = feature_values = []

        if isinstance(context, dict):
            context_names = list(context.keys())
            context_values = list(context.values())
        else:
            context_names, context_values = [''], (context or [1])

        if isinstance(action, dict):
            action_names = list(action.keys())
            action_values = list(action.values())
        else:
            action_names, action_values = [''], action

        for term in self._terms:
            feature_names = feature_names + [ "".join(str(names)) for names in product(*[context_names]*term[0], *[action_names]*term[1]) ]
            feature_values = feature_values + [ np.prod(values) for values in product(*[context_values]*term[0], *[action_values]*term[1]) ]

        return feature_values if len(feature_names) != len(feature_values) else self._h.transform([dict(zip(feature_names, feature_values))])
        