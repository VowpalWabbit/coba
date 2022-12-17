from operator import eq
from abc import ABC, abstractmethod
from collections import abc
from itertools import repeat
from typing import Union, Sequence, Iterator, Any
from coba.backports import Literal

from coba.exceptions import CobaException
from coba.primitives.semantic import Action, AIndex, Batch

class Reward(ABC):
    
    @abstractmethod
    def eval(self, arg: Union[Action,AIndex]) -> float:
        ...

    @abstractmethod
    def argmax(self) -> Union[Action,AIndex]:
        ...
    
    def max(self) -> float:
        return self.eval(self.argmax())

class L1Reward(Reward):

    def __init__(self, label: float) -> None:
        self._label = label
        self.eval = lambda arg: arg-label if label > arg else label-arg 

    def argmax(self) -> float:
        return self._label

    def eval(self, arg: float) -> float: #pragma: no cover
        #defined in __init__ for performance
        raise NotImplementedError("This should have been defined in __init__.")

    def __eq__(self, o: object) -> bool:
        return isinstance(o,L1Reward) and o._label == self._label

class HammingReward(Reward):
    def __init__(self, labels: Sequence[AIndex]) -> None:
        self._label  = set(labels)

    def argmax(self) -> Sequence[AIndex]:
        return self._label

    def eval(self, arg: Sequence[AIndex]) -> float:
        return len(self._label.intersection(arg))/len(self._label)

class BinaryReward(Reward):
    def __init__(self, value: Union[Action,AIndex]):
        self._argmax = value

    def argmax(self) -> Union[Action,AIndex]:
        return self._argmax

    def eval(self, arg: Union[Action,AIndex]) -> float:
        return float(self._argmax==arg)

class ScaleReward(Reward):

    def __init__(self, reward: Reward, shift: float, scale: float, target: Literal["argmax","value"]) -> None:

        if target not in ["argmax","value"]:
            raise CobaException("An unrecognized scaling target was requested for a Reward.")

        self._shift  = shift
        self._scale  = scale
        self._target = target
        self._reward = reward

        old_argmax = reward.argmax()
        new_argmax = (old_argmax+shift)*scale if target == "argmax" else old_argmax

        self._old_argmax = old_argmax
        self._new_argmax = new_argmax

        old_eval = reward.eval

        if self._target == "argmax":
            self.eval = lambda arg: old_eval(old_argmax + (arg-new_argmax))
        if self._target == "value":
            self.eval = lambda arg: (old_eval(arg)+shift)*scale

    def argmax(self) -> float:
        return self._new_argmax

    def eval(self, arg: Any) -> float: #pragma: no cover
        #defined in __init__ for performance
        raise NotImplementedError("This should have been defined in __init__.")

class SequenceReward(Reward):
    def __init__(self, values: Sequence[float]) -> None:
        self._values = values

    def eval(self, arg: AIndex) -> float:
        return self._values[arg]

    def argmax(self) -> Action:
        max_r = self._values[0]
        max_a = 0
        for a,r in enumerate(self._values):
            if r > max_r:
                max_a = a
                max_r = r
        return max_a

    def __getitem__(self, index: int) -> float:
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[float]:
        return iter(self._values)

    def __eq__(self, o: object) -> bool:
        try:
            return list(o) == list(self)
        except:
            return False

class MulticlassReward(Reward):
    __slots__=('_actions','_label')
    
    def __init__(self, actions: Sequence[Action], label: AIndex) -> None:
        self._indexes = list(range(len(actions)))
        self._label   = label

    def eval(self, action: AIndex) -> float:
        return int(self._label == action)

    def argmax(self) -> Action:
        return self._label

    def __getitem__(self, index: int) -> float:
        #we do this strange index lookup to let
        #the _indexes list handle a bad index value
        return self._indexes[index] == self._label

    def __len__(self) -> int:
        return len(self._indexes)

    def __iter__(self) -> Iterator[float]:
        return iter(map(int,map(eq, range(len(self._indexes)), repeat(self._label))))

    def __eq__(self, o: object) -> bool:
        return isinstance(o,abc.Sequence) and list(o) == list(self)

class BatchReward(Batch):
    
    def eval(self, actions: Sequence[Action]) -> Sequence[float]:
        return list(map(lambda r,a: r.eval(a), self, actions))

    def argmax(self) -> Sequence[Action]:
        return [r.argmax() for r in self]

    def max(self) -> float:
        return [r.max() for r in self]
