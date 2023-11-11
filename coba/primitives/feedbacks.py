from typing import Sequence, Callable
from coba.primitives.semantic import Action, Batch

Feedback = Callable[[Action], float]

class SequenceFeedback(Feedback):
    __slots__ = ('_actions','_feedbacks')

    def __init__(self, actions:Sequence[Action], feedbacks: Sequence[float]) -> None:
        self._actions = actions
        self._feedbacks = feedbacks

    def __call__(self, action: Action) -> float:
        return self._feedbacks[self._actions.index(action)]

    def __eq__(self, o: object) -> bool:
        return o == self._feedbacks or (isinstance(o,SequenceFeedback) and o._actions == self._actions and o._feedbacks == self._feedbacks)

class BatchFeedback(Batch):
    def __call__(self, actions: Sequence[Action]) -> Sequence[float]:
        return list(map(lambda f,a: f(a), self, actions))
