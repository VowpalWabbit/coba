from abc import ABC, abstractmethod
from typing import Any, Mapping, Iterable, Union, Optional, Callable

from coba.learners import Learner
from coba.environments import Environment

class Evaluator(ABC):

    @property
    def params(self) -> Mapping[str,Any]:
        """Parameters describing the learner (used for descriptive purposes only).

        Remarks:
            These will become columns in the learners table of experiment results.
        """
        return {}

    @abstractmethod
    def evaluate(self, environment: Optional[Environment], learner: Optional[Learner]) -> Union[Mapping[Any,Any],Iterable[Mapping[Any,Any]]]:
        """Evaluate the learner on the given interactions.

        Args:
            environment: The Environment we want to evaluate against.
            learner: The Learner that we wish to evaluate.

        Returns:
            Evaluation results
        """
        ...

class LambdaEvaluator(Evaluator):
    def __init__(self, func: Callable[[Optional[Environment],Optional[Learner]], Union[Mapping, Iterable[Mapping]]]) -> None:
        self._func = func

    def evaluate(self, environment: Optional[Environment], learner: Optional[Learner]) -> Union[Mapping, Iterable[Mapping]]:
        return self._func(environment,learner)

def get_ope_loss(learner) -> float:
    # OPE loss metric is only available for VW models
    try:
        return learner._learner._vw._vw.get_sum_loss()
    except AttributeError:
        return float("nan")
