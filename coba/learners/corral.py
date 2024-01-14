import math

from typing import Any, Sequence, Optional, Mapping, Tuple, Literal

from coba.exceptions import CobaException
from coba.primitives import Learner, Context, Action, Actions, Prob, Kwargs
from coba.safety import SafeLearner
from coba.learners.utilities import PMFInfoPredictor

class CorralLearner(Learner):
    """A contextual bandit learner that optimizes a collection of learners.

    This is an implementation of the Agarwal et al. (2017) Corral algorithm
    and requires that the reward is always in [0,1].

    References:
        Agarwal, Alekh, Haipeng Luo, Behnam Neyshabur, and Robert E. Schapire.
        "Corralling a band of bandit algorithms." In Conference on Learning
        Theory, pp. 12-38. PMLR, 2017.
    """

    def __init__(self,
        learners: Sequence[Learner],
        eta     : float = 0.075,
        T       : float = math.inf,
        mode    : Literal["importance","off-policy"] ="importance",
        seed    : int = 1) -> None:
        """Instantiate a CorralLearner.

        Args:
            learners: The collection of base learners.
            eta: The learning rate. This controls how quickly Corral picks a best base_learner.
            T: The number of interactions expected during the learning process. A small T will cause
                the learning rate to shrink towards 0 quickly while a large value for T will cause the
                learning rate to shrink towards 0 slowly. A value of inf means that the learning rate
                will remain constant.
            mode: Determines the method with which feedback is provided to the base learners. The
                original paper used importance sampling. We also support `off-policy`.
            seed: A seed for a random number generation.
        """
        if mode not in ["importance", "off-policy"]:
            raise CobaException("The provided `mode` for CorralLearner was unrecognized.")

        self._base_lrns = [ SafeLearner(learner) for learner in learners]

        M = len(self._base_lrns)

        self._T     = T
        self._gamma = 1/T
        self._beta  = 1/math.exp(1/math.log(T))

        self._eta_init = eta
        self._etas     = [ eta ] * M
        self._rhos     = [ float(2*M) ] * M
        self._ps       = [ 1/M ] * M
        self._p_bars   = [ 1/M ] * M

        self._mode = mode
        self._seed = seed
        self._pred = PMFInfoPredictor(self._pmf,seed*1.234 if seed is not None else seed)

    @property
    def params(self) -> Mapping[str, Any]:
        return { 'family': 'corral', 'eta': self._eta_init, 'mode': self._mode, 'T': self._T, 'B': list(map(str,self._base_lrns)), 'seed': self._seed }

    def _pmf(self, context, actions):
        base_predicts = [ base_algorithm.predict(context, actions) for base_algorithm in self._base_lrns ]
        base_actions, base_probs, base_infos = zip(*base_predicts)
        pmf  = [ sum([p_b*int(a==b_a) for p_b,b_a in zip(self._p_bars, base_actions)]) for a in actions ]
        info = (base_actions, base_probs, base_infos)
        return pmf, {'info':info}

    def score(self, context: 'Context', actions: 'Actions', action: 'Action') -> 'Prob':
        return self._pred.score(context,actions,action)

    def predict(self, context: 'Context', actions: 'Actions') -> Tuple['Action','Prob','Kwargs']:
        return self._pred.predict(context,actions)

    def learn(self, context: 'Context', action: 'Action', reward: float, probability:float, info) -> None:
        assert  0 <= reward and reward <= 1, "This Corral implementation assumes a loss between 0 and 1"

        base_actions = info[0]
        base_probs   = info[1]
        base_infos   = info[2]

        if self._mode == "importance":
            # This is what is in the original paper. It has the following characteristics:
            #   > It is able to provide feedback to every base learner on every iteration
            #   > It uses a reward estimator with higher variance and no bias (aka, importance sampling)
            #   > It is "on-policy" with respect to base learner's prediction distributions
            # The reward, R, supplied to the base learners satisifies E[R|context,A] = E[reward|context,A]
            for learner, A, P, base_info in zip(self._base_lrns, base_actions, base_probs, base_infos):
                R = reward * int(A==action)/probability
                learner.learn(context, A, R, P, **base_info)

        if self._mode == "off-policy":
            # An alternative variation to the paper is provided below. It has the following characterisitcs:
            #   > It is able to provide feedback to every base learner on every iteration
            #   > It uses a MVUB reward estimator (aka, the unmodified, observed reward)
            #   > It is "off-policy" (i.e., base learners receive action feedback distributed differently from their predicts).
            for learner, base_info in zip(self._base_lrns, base_infos):
                learner.learn(context, action, reward, probability, **base_info)

        loss = 1-reward

        instant_loss = [ loss/probability * (base_action==action) for base_action in base_actions ]
        self._ps     = CorralLearner._log_barrier_omd(self._ps, instant_loss, self._etas)
        self._p_bars = [ (1-self._gamma)*p + self._gamma*1/len(self._base_lrns) for p in self._ps ]

        for i in range(len(self._base_lrns)):
            if 1/self._p_bars[i] > self._rhos[i]:
                self._rhos[i] = 2/self._p_bars[i]
                self._etas[i] *= self._beta

    @staticmethod
    def _log_barrier_omd(ps, losses, etas) -> Sequence[float]:

        f  = lambda l: float(sum( [ 1/((1/p) + eta*(loss-l)) for p, eta, loss in zip(ps, etas, losses)]))
        df = lambda l: float(sum( [ eta/((1/p) + eta*(loss-l))**2 for p, eta, loss in zip(ps, etas, losses)]))

        denom_zeros = [ ((-1/p)-(eta*loss))/-eta for p, eta, loss in zip(ps, etas, losses) ]

        min_loss = min(losses)
        max_loss = max(losses)

        precision = 4

        def binary_search(l,r) -> Optional[float]:
            #in theory the above check should guarantee this has a solution
            while True:

                x = (l+r)/2
                y = f(x)

                if round(y,precision) == 1:
                    return x

                if y < 1:
                    l = x

                if y > 1:
                    r = x

        def find_root_of_1():
            brackets = list(sorted(filter(lambda z: min_loss <= z and z <= max_loss, set(denom_zeros + [min_loss, max_loss]))))

            for l_brack, r_brack in zip(brackets[:-1], brackets[1:]):

                if (f(l_brack+.00001)-1) * (f(r_brack-.00001)-1) >= 0:
                    continue
                else:
                    # we use binary search because newtons
                    # method can overshoot our objective
                    return binary_search(l_brack, r_brack)

        if min_loss == max_loss:
            lmbda = min_loss
        elif min_loss not in denom_zeros and round(f(min_loss),precision) == 1:
            lmbda = min_loss
        elif max_loss not in denom_zeros and round(f(max_loss),precision) == 1:
            lmbda = max_loss
        else:
            lmbda = find_root_of_1()

        if lmbda is None:
            raise Exception(f'Something went wrong in Corral OMD {ps}, {etas}, {losses}')

        new_ps = [ 1/((1/p) + eta*(loss-lmbda)) for p, eta, loss in zip(ps, etas, losses)]

        assert round(sum(new_ps),precision) == 1, "An invalid update was made by the log barrier in Corral"

        return new_ps
