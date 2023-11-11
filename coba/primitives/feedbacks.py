from abc import ABC, abstractmethod
from typing import Sequence, Any

from coba.primitives.semantic import Action, Batch

class Feedback(ABC):
    @abstractmethod
    def __call__(self, arg: Action) -> Any:
        ...

class SequenceFeedback(Feedback):
    __slots__ = ('_actions','_feedbacks')

    def __init__(self, actions:Sequence[Action], feedbacks: Sequence[Any]) -> None:
        self._actions = actions
        self._feedbacks = feedbacks

    def __call__(self, action: Action) -> Any:
        return self._feedbacks[self._actions.index(action)]

    def __eq__(self, o: object) -> bool:
        return o == self._feedbacks or (isinstance(o,SequenceFeedback) and o._actions == self._actions and o._feedbacks == self._feedbacks)

class BatchFeedback(Batch):
    def __call__(self, actions: Sequence[Action]) -> Sequence[Any]:
        return list(map(lambda f,a: f(a), self, actions))
