"""
This module contains the abstract interface expected for Learner implementations.
This interface is provided for type checking only. It is not required that one
inherit from this interface to implement new learners. In addition, a number 
of learners are provided out of the box for testing and baseline comparisons.
"""

from coba.learners.primitives import Learner, SafeLearner
from coba.learners.bandit     import EpsilonBanditLearner, UcbBanditLearner, FixedLearner, RandomLearner
from coba.learners.corral     import CorralLearner
from coba.learners.vowpal     import VowpalLearner, VowpalMediator
from coba.learners.vowpal     import VowpalEpsilonLearner, VowpalSoftmaxLearner, VowpalBagLearner
from coba.learners.vowpal     import VowpalCoverLearner, VowpalRegcbLearner, VowpalSquarecbLearner, VowpalOffPolicyLearner
from coba.learners.linucb     import LinUCBLearner

__all__ = [
    'Learner',
    'SafeLearner',
    'RandomLearner',
    'FixedLearner',
    'EpsilonBanditLearner',
    'UcbBanditLearner',
    'CorralLearner',
    'LinUCBLearner',
    'VowpalLearner',
    'VowpalEpsilonLearner',
    'VowpalSoftmaxLearner',
    'VowpalBagLearner',
    'VowpalCoverLearner',
    'VowpalRegcbLearner',
    'VowpalSquarecbLearner',
    'VowpalOffPolicyLearner',
    'VowpalMediator'
]