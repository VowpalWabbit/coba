"""This module contains all public learners and learner interfaces."""

from coba.learners.primitives import CbLearner, IgLearner, SafeLearner
from coba.learners.bandit     import EpsilonBanditLearner, UcbBanditLearner, FixedLearner, RandomLearner
from coba.learners.corral     import CorralLearner
from coba.learners.vowpal     import VowpalMediator
from coba.learners.vowpal     import VowpalLearner, VowpalEpsilonLearner, VowpalSoftmaxLearner, VowpalBagLearner
from coba.learners.vowpal     import VowpalCoverLearner, VowpalRegcbLearner, VowpalSquarecbLearner, VowpalOffPolicyLearner
from coba.learners.linucb     import LinUCBLearner

__all__ = [
    'CbLearner',
    'IgLearner',
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
    'VowpalMediator',
]