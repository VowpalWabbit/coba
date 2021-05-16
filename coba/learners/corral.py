"""An implementation of the Corral algorithm."""

import math

from typing import Any, Sequence, Optional, Dict

from coba.random import CobaRandom
from coba.simulations import Context, Action, Key
from coba.learners.core import Learner

class CorralLearner(Learner):
    """This is an implementation of the Agarwal et al. (2017) Corral algorithm.

    This algorithm assumes that the reward distribution has support in [0,1]
    and implements the remark on pg. 8 to improve learning efficiency when 
    multiple bandits select the same action.

    References:
        Agarwal, Alekh, Haipeng Luo, Behnam Neyshabur, and Robert E. Schapire. 
        "Corralling a band of bandit algorithms." In Conference on Learning 
        Theory, pp. 12-38. PMLR, 2017.
    """

    def __init__(self, base_learners: Sequence[Learner], eta: float, T: float = math.inf, seed: int = None) -> None:
        """Instantiate a CorralLearner.
        
        Args:
            base_learners: The collection of algorithms to use as base learners.
            eta: The learning rate. In our experiments a value between 0.05 and .10 often seemed best.
            T: The number of interactions expected during the learning process. In our experiments 
                Corral performance seemed relatively insensitive to this value.
            seed: A seed for a random number generation in ordre to get repeatable results.
        """

        self._base_learners = base_learners

        M = len(self._base_learners)

        self._gamma = 1/T
        self._beta  = 1/math.exp(1/math.log(T))

        self._eta_init = eta
        self._etas     = [ eta ] * M
        self._rhos     = [ float(2*M) ] * M
        self._ps       = [ 1/M ] * M
        self._p_bars   = [ 1/M ] * M

        self._random   = CobaRandom(seed)

        self._base_actions : Dict[Key, Sequence[Action]] = {}
        self._base_predicts: Dict[Key, Sequence[float]]  = {}

    @property
    def family(self) -> str:
        """The family of the learner.

        See the base class for more information
        """
        return "corral"
    
    @property
    def params(self) -> Dict[str, Any]:
        """The parameters of the learner.

        See the base class for more information
        """        
        return {"eta": self._eta_init, "B": [ b.family for b in self._base_learners ] }

    def init(self) -> None:
        for learner in self._base_learners:
            try:
                learner.init()
            except:
                pass

    def predict(self, key: Key, context: Context, actions: Sequence[Action]) -> Sequence[float]:
        """Determine a PMF with which to select the given actions.

        Args:
            key: The key identifying the interaction we are choosing for.
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The probability of taking each action. See the base class for more information.
        """
        
        predicts = [ base_algorithm.predict(key, context, actions) for base_algorithm in self._base_learners ]
        
        base_actions  = [ self._random.choice(actions, predict) for predict in predicts                   ]
        base_predicts = [ predict[actions.index(action)] for action,predict in zip(base_actions,predicts) ]

        self._base_actions[key]  = base_actions
        self._base_predicts[key] = base_predicts

        return [ sum([p_b*int(a==b_a) for p_b,b_a in zip(self._p_bars, base_actions)]) for a in actions ]

    def learn(self, key: Key, context: Context, action: Action, reward: float, probability: float) -> None:
        """Learn from the given interaction.

        Args:
            key: The key identifying the interaction this observed reward came from.
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
            probability: The probability that the given action was taken.
        """

        loss = 1-reward

        assert  0 <= loss and loss <= 1, "The current Corral implementation assumes a loss between 0 and 1"

        base_actions  = self._base_actions.pop(key)
        base_predicts = self._base_predicts.pop(key)

        losses  = [ loss/probability   * int(act==action) for act in base_actions ]
        rewards = [ reward/probability * int(act==action) for act in base_actions ]

        for learner, action, R, P in zip(self._base_learners, base_actions, rewards, base_predicts):
            learner.learn(key, context, action, R, P) # COBA learners assume a reward

        self._ps     = list(self._log_barrier_omd(losses))
        self._p_bars = [ (1-self._gamma)*p + self._gamma*1/len(self._base_learners) for p in self._ps ]

        for i in range(len(self._base_learners)):
            if 1/self._p_bars[i] > self._rhos[i]:
                self._rhos[i] = 2/self._p_bars[i]
                self._etas[i] *= self._beta

    def _log_barrier_omd(self, losses) -> Sequence[float]:

        f  = lambda l: float(sum( [ 1/((1/p) + eta*(loss-l)) for p, eta, loss in zip(self._ps, self._etas, losses)]))
        df = lambda l: float(sum( [ eta/((1/p) + eta*(loss-l))**2 for p, eta, loss in zip(self._ps, self._etas, losses)]))

        denom_zeros = [ ((-1/p)-(eta*loss))/-eta for p, eta, loss in zip(self._ps, self._etas, losses) ]

        min_loss = min(losses)
        max_loss = max(losses)

        precision = 4

        def newtons_zero(l,r) -> Optional[float]:
            """Use Newton's method to calculate the root."""
            
            #depending on scales this check may fail though that seems unlikely
            if (f(l+.0001)-1) * (f(r-.00001)-1) >= 0:
                return None

            i = 0
            x = (l+r)/2

            while True:
                i += 1

                if df(x) == 0:
                    raise Exception(f'Something went wrong in Corral (0) {self._ps}, {self._etas}, {losses}, {x}')

                x -= (f(x)-1)/df(x)

                if round(f(x),precision) == 1:
                    return x

                if (i % 30000) == 0:
                    print(i)

        lmbda: Optional[float] = None

        if min_loss == max_loss:
            lmbda = min_loss
        elif min_loss not in denom_zeros and round(f(min_loss),precision) == 1:
            lmbda = min_loss
        elif max_loss not in denom_zeros and round(f(max_loss),precision) == 1:
            lmbda = max_loss
        else:
            brackets = list(sorted(filter(lambda z: min_loss <= z and z <= max_loss, set(denom_zeros + [min_loss, max_loss]))))

            for l_brack, r_brack in zip(brackets[:-1], brackets[1:]):
                lmbda = newtons_zero(l_brack, r_brack)
                if lmbda is not None: break

        if lmbda is None:
            raise Exception(f'Something went wrong in Corral (None) {self._ps}, {self._etas}, {losses}')

        return [ max(1/((1/p) + eta*(loss-lmbda)),.00001) for p, eta, loss in zip(self._ps, self._etas, losses)]