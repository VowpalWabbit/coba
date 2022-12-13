from abc import ABC, abstractmethod
from collections import abc
from typing import Union, Sequence, Any

from coba.primitives.semantic import Action, AIndex

class Feedback(ABC):
    @abstractmethod
    def eval(self, arg: Union[Action,AIndex]) -> Any:
        ...

class SequenceFeedback(Feedback):
    def __init__(self, values: Sequence[Any]) -> None:
        self._values = values

    def eval(self, arg: AIndex) -> Any:
        return self._values[arg]

    def __getitem__(self, index: int) -> Any:
        return self._values[index]
    
    def __len__(self) -> int:
        return len(self._values)

    def __eq__(self, o: object) -> bool:
        return isinstance(o,abc.Sequence) and list(o) == list(self._values)
