import time
import math

from collections import defaultdict
from itertools import product
from typing import Any, Dict, Sequence

from coba.utilities import PackageChecker
from coba.simulations import Context, Action
from coba.learners.core import Learner, Key

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

    def __init__(self, *, beta: float, alpha: float, learning_rate:float=0.1, interactions: Sequence[str] = ['a', 'ax']) -> None:
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
        from sklearn.feature_extraction import FeatureHasher
        from sklearn.preprocessing import PolynomialFeatures
        
        self._beta = beta
        self._alpha = alpha
        self._iter = 0

        self._core_model = []

        self._times         = [0,0,0,0]
        self._interactions  = interactions
        self._terms         = []
        self._learning_rate = learning_rate

        for term in self._interactions:
            term = term.lower()
            x_num = term.count('x')
            a_num = term.count('a')

            if x_num + a_num != len(term):
                raise Exception("Letters other than x and a were passed for parameter interactions. Please remove other letters/characters.")

            self._terms.append((x_num, a_num))

        max_x_term = max(max(term[0] for term in self._terms),1)
        max_a_term = max(max(term[1] for term in self._terms),1)

        self._x_p = PolynomialFeatures(degree=max_x_term, include_bias=False, interaction_only=False)
        self._a_p = PolynomialFeatures(degree=max_a_term, include_bias=False, interaction_only=False)
        self._h   = FeatureHasher(input_type='pair')

    def predict(self, key: Key, context: Context, actions: Sequence[Action]) -> Sequence[float]:
        """Determine a PMF with which to select the given actions.

        Args:
            key: The key identifying the interaction we are choosing for.
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The probability of taking each action. See the base class for more information.
        """

        import numpy as np
        from scipy import sparse

        if self._iter == 0:
            if isinstance(context,dict) or isinstance(actions[0],dict):
                self._core_model = sparse.csr_matrix(self._featurize(context, actions[0]).shape)
            else:
                self._core_model = np.zeros(self._featurize(context, actions[0]).shape)

        if (self._iter < 200):
            return [1/len(actions)] * len(actions)

        else:
            maxScore  = -float('inf')
            maxAction = None

            for action in actions:
                features = self._featurize(context,action)
                score = self._bin_search(features, len(actions))

                if score > maxScore:
                    maxAction = action
                    maxScore  = score

            return [int(action == maxAction) for action in actions]

    def _bin_search(self, features, K_t) -> float:

        start = time.time()

        y_u = 2
        w   = 1

        f_u_a_w = self._update_model(self._core_model, features, y_u, w)

        f_x_t_a = self._predict_model(self._core_model, features)
        s_u_a   = (self._predict_model(f_u_a_w, features) - f_x_t_a) / w

        obj = lambda w: w*(f_x_t_a-y_u)**2 - w*(f_x_t_a+s_u_a*w-y_u)**2

        lower_search_bound = 0
        upper_search_bound = (f_x_t_a-y_u)/(-s_u_a)
        width_search_bound = upper_search_bound - lower_search_bound

        constraint = self._alpha * math.log(K_t)

        w_old = lower_search_bound
        w_now = lower_search_bound + 1/2*width_search_bound
        o     = obj(w_now)

        while abs(w_now-w_old) > width_search_bound*(1/2)**30 or o >= constraint:
            w_diff = abs(w_now-w_old)
            w_old  = w_now
            if o < constraint:
                w_now += w_diff/2
            else:
                w_now -= w_diff/2
            o = obj(w_now)

        self._times[1] += time.time() - start

        return f_x_t_a + s_u_a*w_now

    def learn(self, key: Key, context: Context, action: Action, reward: float, probability: float) -> None:
        """Learn from the given interaction.

        Args:
            key: The key identifying the interaction this observed reward came from.
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
            probability: The probability that the given action was taken.
        """

        start = time.time()
        features = self._featurize(context, action)
        self._core_model = self._update_model(self._core_model, features, reward, 1)
        self._times[2] += time.time()-start

        self._iter += 1

        if self._iter % 800 == 0:
            print(f'avg phi time: {round(self._times[0]/(self._iter),2)}')
            print(f'avg bin time: {round(self._times[1]/(self._iter),2)}')
            print(f'avg lrn time: {round(self._times[2]/(self._iter),2)}')

    def _featurize(self, context, action):
        import numpy as np #type: ignore

        start = time.time()

        is_sparse = isinstance(context, dict) or isinstance(action, dict)

        if isinstance(context, dict):
            context_values = list(context.values())
            context_names  = list([ f"x{k}" for k in context.keys() ])
        else:
            context_values = (context or [1])
            context_names  = [''] if not is_sparse else [ f"x{i}" for i in range(len(context_values)) ]

        if isinstance(action, dict):
            action_names  = list([ f"a{k}" for k in action.keys() ])
            action_values = list(action.values())
        else:
            action_values = action
            action_names  = [''] if not is_sparse else [ f"a{i}" for i in range(len(action_values)) ]
        
        x_feat = self._x_p.fit_transform([context_values])
        a_feat = self._a_p.fit_transform([action_values])
        
        if is_sparse:
            x_names = self._x_p.get_feature_names(context_names)
            a_names = self._a_p.get_feature_names(action_names)

        x_terms_by_degree = {0:[1]}
        a_terms_by_degree = {0:[1]}
        a_names_by_degree = {0:['']}
        x_names_by_degree = {0:['']}

        index = 0
        for degree in range(1,self._x_p.degree+1):
            degree_count = int((len(context_values)**degree + len(context_values))/2)
            x_terms_by_degree[degree] = x_feat[0,index:degree_count]
            
            if is_sparse:
                x_names_by_degree[degree] = x_names[index:degree_count]
            
            index+=degree_count
            
        index = 0
        for degree in range(1,self._a_p.degree+1):
            degree_count = int((len(action_values)**degree + len(action_values))/2)
            a_terms_by_degree[degree] = a_feat[0,index:degree_count]
            
            if is_sparse:
                a_names_by_degree[degree] = a_names[index:degree_count]
            
            index+=degree_count

        features = np.empty((1,0))
        names    = []

        for term in self._terms:            
            features = np.hstack([features,np.outer(x_terms_by_degree[term[0]],a_terms_by_degree[term[1]]).T.reshape((1,-1))])
            
            if is_sparse:
                names += list(map("".join,product(x_names_by_degree[term[0]],a_names_by_degree[term[1]])))

        final_features = features if not is_sparse else self._h.fit_transform([list(zip(names,features[0]))])

        self._times[0] += time.time() - start

        return final_features

    def _predict_model(self, model, features):
        return (model @ features.T)[0,0]

    def _update_model(self, model, features, value, importance):
        error = self._predict_model(model, features) - value
        return model - self._learning_rate*features*error*importance