from coba.utilities import PackageChecker
from coba.simulations import Context, Action
from coba.learners.core import Learner, Key
from typing import Any, Dict, Sequence

class LinUCBLearner(Learner):
    """A learner using the LinUCB algorithm from "Contextual Bandits with Linear Payoff Functions" by Wei Chu et al."""

    @property
    def family(self) -> str:
        """The family of the learner.

        See the base class for more information
        """
        return f"linUCB"

    @property
    def params(self) -> Dict[str, Any]:
        """The parameters of the learner.

        See the base class for more information.
        """
        dict = {'alpha': self._alpha, 'interactions': self._interactions}
        return dict

    def __init__(self, *, alpha: float, interactions: Sequence[str] = ['xa', 'a']) -> None:
        """Instantiate a linUCBLearner.
        Args:
            alpha: number of standard deviations
            interactions: the set of interactions the learner will use. x refers to context and a refers to actions, 
                e.g. xaa would mean interactions between context and actions and actions. 
        """
        PackageChecker.numpy("linUCBLearner.__init__")
        PackageChecker.sklearn("linUCBLearner.__init__")
        import numpy as np
        from sklearn.feature_extraction import FeatureHasher

        self._A = None
        self.b = None
        self._alpha = alpha
        self._theta: Any
        self._interactions = interactions
        self._terms = None

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
        from sklearn.feature_extraction import FeatureHasher
        import scipy.sparse as sp
        from scipy.sparse import linalg


        self._d = len(actions[0])
        actions_array = context_array = interactions = temp_array = None
        self._terms = []
        is_sparse = type(actions[0])==dict

        if (self._terms is not None):
            for term in self._interactions:
                term = term.lower()
                x_num = term.count('x')
                a_num = term.count('a')
                if(x_num + a_num != len(term)):
                    raise Exception("Letters other than x and a were passed for parameter interactions. Please remove other letters/characters.")
                self._terms.append((x_num, a_num))

        if (is_sparse):
            raise Exception("Sparse data cannot be handled by this algorithm.")
            """ Legacy Code
            h = FeatureHasher()
            actions = [{str(k):v for k,v in x.items()} for x in actions]
            actions_array = h.transform(actions)
            context_array = np.fromiter(context.values(), dtype=float) 
            context_array = np.append(context_array, 1.0)
            context_array = context_array[context_array != np.array(None)]
            
            if(self._A is None):
                self._A = sp.csc_matrix(sp.identity(2**20))
                self._b = sp.csr_matrix((2**20, 1), dtype=np.float32)
            
            interactions = []
            for i in range(len(actions)):
                interactions.append(np.outer(actions_array[i], context_array).reshape(-1))
            A_inv = linalg.inv(self._A)
            """
        else:
            actions_array = np.array(actions)
            context_array = np.array(context)
            if (context is None):
                context_array = np.append(context_array, 1.0)
                context_array = context_array[context_array != np.array(None)]
                interactions = np.apply_along_axis(lambda x: np.outer(x, context_array).reshape(-1), 1, actions_array)
            else:
                for term in self._terms:
                    temp_array_1 = [[1.0]]
                    temp_array_2 = np.ones((len(actions), 1 ), dtype=float)
                    for i in range(term[0]):
                        temp_array_1 = np.outer(temp_array_1, context_array).reshape(-1)
                    for j in range(term[1]):
                        temp_array_2 = [np.outer(temp_array_2[i], actions_array[i]).reshape(-1) for i in range(len(actions_array))]
                    temp_array = np.apply_along_axis(lambda x: np.outer(x, temp_array_1).reshape(-1), 1, temp_array_2)

                    if (interactions is None):
                        interactions = temp_array
                    else:
                        interactions = np.hstack((interactions, temp_array))

            if(self._A is None):
                self._A = np.identity((len(interactions[0])))
                self._b = np.zeros([(len(interactions[0])), 1])
            
            A_inv = np.linalg.inv(self._A)

        self._theta = np.dot(A_inv, self._b)
        
        term_one = np.zeros([len(actions),1])
        term_two = np.zeros([len(actions),1])
        for i in range(len(actions)):
            term_one[i] = np.dot(self._theta.T, interactions[i])
            term_two[i] = self._alpha * np.sqrt(np.dot(np.dot(interactions[i].T, A_inv), interactions[i]))

        # pick greatest probability set all others to 0 and it to 1
        action_values = np.add(term_one, term_two)

        index = np.where(action_values == np.amax(action_values))[0]
        ret = [0] * len(action_values)
        ret = [1 if ind in index else 0 for ind, x in enumerate(ret)]


        return ret

    def learn(self, key: Key, context: Context, action: Action, reward: float, probability: float) -> None:
        """Learn from the given interaction.

        Args:
            key: The key identifying the interaction this observed reward came from.
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
            probability: The probability that the given action was taken.
        """
        import numpy as np

        action_array = np.array(action)
        context_array = np.array(context)
        interactions: np.ndarray = None
        
        if (context==None):
            interactions = action_array.T
        else:
            for term in self._terms:
                temp_array_1 = [[1.0]]
                temp_array_2 = [[1.0]]
                for i in range(term[0]):
                    temp_array_1 = np.outer(temp_array_1, context_array).reshape(-1)
                for j in range(term[1]):
                    temp_array_2 = np.outer(temp_array_2, action_array).reshape(-1)
                temp_array = np.outer(temp_array_2, temp_array_1).reshape(-1)

                if (interactions is None):
                    interactions = temp_array
                else:
                    interactions = np.hstack((interactions, temp_array))


        interactions = np.expand_dims(interactions, axis=(1,))

        action = interactions
        self._A = self._A + action*action.T
        self._b = self._b + action*reward 
        ...
