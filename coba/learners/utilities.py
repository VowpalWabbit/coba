from typing import Tuple, Callable,Sequence

from coba.random import CobaRandom
from coba.primitives import Context, Action, Actions, Prob, Kwargs

class PMFPredictor:
    def __init__(self, pmf: Callable[['Context','Actions'],Sequence[Prob]], seed:int = 1) -> None:
        """Instantiate a PMFPredictor.

        Args:
            pmf: Return the PMF for given context and actions.
            seed: A seed for a random number generation.
        """
        self._pmfrng  = CobaRandom(seed)
        self._pmfcall = pmf

    @property
    def seed(self) -> int:
        return self._pmfrng.seed

    def score(self, context: 'Context', actions: 'Actions', action: 'Action') -> 'Prob':
        return self._pmfcall(context,actions)[actions.index(action)]

    def predict(self, context: 'Context', actions: 'Actions') -> Tuple['Action','Prob']:
        return self._pmfrng.choicew(actions,self._pmfcall(context,actions))

class PMFInfoPredictor:
    def __init__(self, pmf: Callable[['Context','Actions'],Tuple[Sequence[Prob],'Kwargs']], seed:int = 1) -> None:
        """Instantiate a PMFInfoPredictor.

        Args:
            pmf: Return the PMF for given context and actions.
            seed: A seed for a random number generation.
        """
        self._pmfrng  = CobaRandom(seed)
        self._pmfcall = pmf

    @property
    def seed(self) -> int:
        return self._pmfrng.seed

    def score(self, context: 'Context', actions: 'Actions', action: 'Action') -> 'Prob':
        return self._pmfcall(context,actions)[0][actions.index(action)]

    def predict(self, context: 'Context', actions: 'Actions') -> Tuple['Action','Prob','Kwargs']:
        pmf,info = self._pmfcall(context,actions)
        return *self._pmfrng.choicew(actions,pmf),info
