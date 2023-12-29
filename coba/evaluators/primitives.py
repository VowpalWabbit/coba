from abc import ABC, abstractmethod
from typing import Any, Mapping, Iterable, Union, Optional, Callable

from coba.primitives import Learner

from coba.environments import Environment

class Evaluator(ABC):

    @property
    def params(self) -> Mapping[str,Any]:
        """Parameters describing the evaluator (used for descriptive purposes only).

        Remarks:
            These will become columns in the evaluators table of experiment results.
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

class SafeEvaluator(Evaluator):
    def __init__(self, evaluator: Union[Evaluator, Callable[[Environment,Learner], Union[Mapping, Iterable[Mapping]]]]) -> None:
        self.evaluator = evaluator if not isinstance(evaluator, SafeEvaluator) else evaluator.evaluator

    @property
    def params(self):
        try:
            params = self.evaluator.params
        except:
            params = {}

        if type(self.evaluator).__name__ == 'function':
            params['eval_type'] = self.evaluator.__name__
        else:
            params['eval_type'] = type(self.evaluator).__name__

        return params

    def evaluate(self, environment: Optional[Environment], learner: Optional[Learner]) -> Union[Mapping, Iterable[Mapping]]:
        if callable(self.evaluator):
            return self.evaluator(environment,learner)
        else:
            return self.evaluator.evaluate(environment,learner)

def get_ope_loss(learner) -> float:
    # OPE loss metric is only available for VW models
    try:
        return learner.learner._vw._vw.get_sum_loss()
    except AttributeError:
        return float("nan")
