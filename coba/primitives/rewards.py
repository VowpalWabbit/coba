from abc import ABC, abstractmethod
from typing import Sequence, Optional, Mapping, Any
from coba.backports import Literal

from coba.exceptions import CobaException
from coba.primitives.semantic import Action, Batch

class Reward(ABC):
    __slots__ = ()

    @abstractmethod
    def eval(self, action: Action) -> float:
        ...

    @abstractmethod
    def argmax(self) -> Action:
        ...

    @abstractmethod
    def max(self) -> float:
        ...

class IPSReward(Reward):
    __slots__ = ('_reward','_action')

    def __init__(self, reward: float, action: Action, probability: Optional[float]) -> None:
        self._reward = reward/(probability or 1)
        self._action = action

    def eval(self, arg: Action) -> float:
        return self._reward if arg == self._action else 0

    def argmax(self) -> Action:
        return self._action
    
    def max(self) -> float:
        return self._reward 

    def __reduce__(self):
        #this makes the pickle smaller
        return IPSReward, (self._reward, self._action, 1)
    
    def __eq__(self, o: object) -> bool:
        return isinstance(o,IPSReward) and o._reward == self._reward and o._action == self._action

class L1Reward(Reward):
    __slots__ = ('_label',)

    def __init__(self, action: float) -> None:
        self._label = action

    def eval(self, action: float) -> float: #pragma: no cover
        return action-self._label if self._label > action else self._label-action 

    def argmax(self) -> float:
        return self._label

    def max(self) -> float:
        return 0

    def __reduce__(self):
        #this makes the pickle smaller
        return L1Reward, (self._label,)

    def __eq__(self, o: object) -> bool:
        return isinstance(o,L1Reward) and o._label == self._label

class HammingReward(Reward):
    __slots__ = ('_labels',)

    def __init__(self, labels: Sequence[Action]) -> None:
        self._labels = set(labels)

    def eval(self, arg: Sequence[Action]) -> float:
        return len(self._labels.intersection(arg))/len(self._labels.union(arg))

    def argmax(self) -> Sequence[Action]:
        return self._labels
    
    def max(self) -> float:
        return 1

    def __reduce__(self):
        #this makes the pickle smaller
        return HammingReward, (tuple(self._labels),)

class BinaryReward(Reward):
    __slots__ = ('_argmax',)

    def __init__(self, action: Action):
        self._argmax = action

    def eval(self, action: Action) -> float:
        return float(self._argmax==action)

    def argmax(self) -> Action:
        return self._argmax

    def max(self) -> Action:
        return 1

    def __reduce__(self):
        #this makes the pickle smaller
        return BinaryReward, (self._argmax,)
    
    def __eq__(self, o: object) -> bool:
        return o == self._argmax or (isinstance(o,BinaryReward) and o._argmax == self._argmax)

class ScaleReward(Reward):
    def __init__(self, reward: Reward, shift: float, scale: float, target: Literal["argmax","value"]) -> None:

        if target not in ["argmax","value"]:
            raise CobaException("An unrecognized scaling target was requested for a Reward.")

        self._reward = reward
        self._shift  = shift
        self._scale  = scale
        self._target = target

        old_argmax = reward.argmax()
        new_argmax = (old_argmax+shift)*scale if target == "argmax" else old_argmax

        self._old_argmax = old_argmax
        self._new_argmax = new_argmax

        self._old_eval = reward.eval

    def eval(self, action: Any) -> float: #pragma: no cover
        if self._target == "argmax":
            return self._old_eval(self._old_argmax + (action-self._new_argmax))
        if self._target == "value":
            return (self._old_eval(action)+self._shift)*self._scale

    def argmax(self) -> float:
        return self._new_argmax
    
    def max(self) -> float:
        return self.eval(self._new_argmax)

    def __reduce__(self):
        #this makes the pickle smaller
        return ScaleReward, (self._reward, self._shift, self._scale, self._target)

class SequenceReward(Reward):
    __slots__ = ('_actions','_rewards')

    def __init__(self, actions: Sequence[Action], rewards: Sequence[float]) -> None:
        if len(actions) != len(rewards): 
            raise CobaException("The given actions and rewards did not line up.")

        self._actions = actions
        self._rewards = rewards

    def eval(self, action: Action) -> float:
        return self._rewards[self._actions.index(action)]

    def argmax(self) -> Action:
        return self._actions[self._rewards.index(max(self._rewards))]

    def max(self) -> float:
        return max(self._rewards)

    def __reduce__(self):
        #this makes the pickle smaller
        return SequenceReward, (self._actions,self._rewards)

    def __eq__(self, o: object) -> bool:
        return o == self._rewards or (isinstance(o,SequenceReward) and o._actions == self._actions and o._rewards == self._rewards)

class MappingReward(Reward):
    __slots__ = ('_mapping','eval')

    def __init__(self, mapping: Mapping[Action,float]) -> None:
        self._mapping = mapping
        self.eval     = mapping.__getitem__

    def argmax(self) -> Action:
        max_r = -float('inf')
        max_a = 0
        for a,r in self._mapping.items():
            if r > max_r:
                max_a = a
                max_r = r
        return max_a

    def max(self) -> float:
        return max(self._mapping.values())

    def __reduce__(self):
        #this makes the pickle smaller
        return MappingReward, (self._mapping,)

    def __eq__(self, o: object) -> bool:
        return o == self._mapping or (isinstance(o,MappingReward) and o._mapping == self._mapping)

class MulticlassReward(Reward):
    __slots__=('_label',)
    
    def __init__(self, label: Action) -> None:
        self._label = label

    def eval(self, action: Action) -> float:
        return int(self._label == action)

    def argmax(self) -> Action:
        return self._label
    
    def max(self) -> float:
        return 1

    def __reduce__(self):
        return MulticlassReward, (self._label,)
    
    def __eq__(self, o: object) -> bool:
        return o == self._label or (isinstance(o,MulticlassReward) and o._label == self._label)

class BatchReward(Batch):
    def eval(self, actions: Sequence[Action]) -> Sequence[float]:
        return list(map(lambda r,a: r.eval(a), self, actions))

    def argmax(self) -> Sequence[Action]:
        return [r.argmax() for r in self]

    def max(self) -> float:
        return [r.max() for r in self]
