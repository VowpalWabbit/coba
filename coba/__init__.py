from coba.random import CobaRandom
from coba.contexts import CobaContext, NullLogger

from coba.environments import Environments, ArffSource, CsvSource, LibSvmSource, ManikSource
from coba.environments import Interaction, SimulatedInteraction, LoggedInteraction, GroundedInteraction, LambdaSimulation

from coba.learners import Learner, ActionProb, PMF
from coba.learners import SafeLearner, FixedLearner, RandomLearner
from coba.learners import EpsilonBanditLearner, UcbBanditLearner
from coba.learners import CorralLearner, LinUCBLearner
from coba.learners import VowpalLearner, VowpalEpsilonLearner, VowpalSoftmaxLearner, VowpalBagLearner, VowpalRndLearner
from coba.learners import VowpalCoverLearner, VowpalRegcbLearner, VowpalSquarecbLearner, VowpalOffPolicyLearner

from coba.evaluators import OnPolicyEvaluator, OffPolicyEvaluator, ClassMetaEvaluator

from coba.experiments  import Experiment, Result

from coba.utilities import peek_first
from coba.exceptions import CobaException
from coba.backports import version, PackageNotFoundError

try:
    #Option (5) on https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
    __version__ = version('coba') 
except PackageNotFoundError: #pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "CobaException",
    "CobaContext",
    "Environments",
    "Experiment",
    "Result",
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
    'VowpalRndLearner',
    'CobaRandom',
    'NullLogger',
    'ArffSource',
    'CsvSource',
    'LibSvmSource',
    'ManikSource',
    'SimulatedInteraction',
    'LoggedInteraction',
    'GroundedInteraction',
    'LambdaSimulation',
    'Learner',
    'SafeLearner',
    'peek_first',
    'ActionProb',
    'PMF',
    'Interaction',
    'OnPolicyEvaluator',
    'OffPolicyEvaluator',
    'ClassMetaEvaluator'

]
