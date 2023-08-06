from abc import ABC, abstractmethod
from typing import Sequence, Optional, Mapping

from coba.exceptions import CobaException
from coba.primitives.semantic import Action, Batch

def argmax(actions: Sequence[Action], rewards: 'Reward') -> Action:
    max_r = -float('inf')
    max_a = 0
    for a in actions:
        r = rewards.eval(a)
        if r > max_r:
            max_a = a
            max_r = r

    return max_a

class Reward(ABC):
    __slots__ = ()

    @abstractmethod
    def eval(self, action: Action) -> float:
        ...

    def to_json(self) -> Mapping[str, str]:
        return {key.lstrip('_'): getattr(self, key, None) for key in self.__slots__}

class IPSReward(Reward):
    __slots__ = ('_reward','_action')

    def __init__(self, reward: float, action: Action, probability: Optional[float] = None) -> None:
        self._reward = reward/(probability or 1)
        self._action = action

    def eval(self, arg: Action) -> float:
        return self._reward if arg == self._action else 0

    def __reduce__(self):
        #this makes the pickle smaller
        return IPSReward, (self._reward, self._action, 1)
    
    def __eq__(self, o: object) -> bool:
        return isinstance(o,IPSReward) and o._reward == self._reward and o._action == self._action

class L1Reward(Reward):
    __slots__ = ('_label',)

    def __init__(self, action: float) -> None:
        self._label = action

    def eval(self, action: float) -> float:
        return action-self._label if self._label > action else self._label-action 

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

    def __reduce__(self):
        #this makes the pickle smaller
        return HammingReward, (tuple(self._labels),)

class BinaryReward(Reward):
    __slots__ = ('_maxarg',)

    def __init__(self, action: Action) -> None:
        self._maxarg = action

    def eval(self, action: Action) -> float:
        return float(self._maxarg==action)

    def __reduce__(self):
        #this makes the pickle smaller
        return BinaryReward, (self._maxarg,)
    
    def __eq__(self, o: object) -> bool:
        return o == self._maxarg or (isinstance(o,BinaryReward) and o._maxarg == self._maxarg)

class SequenceReward(Reward):
    __slots__ = ('_actions','_rewards')

    def __init__(self, actions: Sequence[Action], rewards: Sequence[float]) -> None:
        if len(actions) != len(rewards): 
            raise CobaException("The given actions and rewards did not line up.")

        self._actions = actions
        self._rewards = rewards

    def eval(self, action: Action) -> float:
        return self._rewards[self._actions.index(action)]

    def __reduce__(self):
        #this makes the pickle smaller
        return SequenceReward, (self._actions,self._rewards)

    def __eq__(self, o: object) -> bool:
        return o == self._rewards or (isinstance(o,SequenceReward) and o._actions == self._actions and o._rewards == self._rewards)

class MappingReward(Reward):
    __slots__ = ('_mapping',)

    def __init__(self, mapping: Mapping[Action,float]) -> None:
        self._mapping = mapping

    def eval(self, action: Action) -> float:
        return self._mapping[action]

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

    def __reduce__(self):
        return MulticlassReward, (self._label,)
    
    def __eq__(self, o: object) -> bool:
        return o == self._label or (isinstance(o,MulticlassReward) and o._label == self._label)

class BatchReward(Batch):
    def eval(self, actions: Sequence[Action]) -> Sequence[float]:
        return list(map(lambda r,a: r.eval(a), self, actions))
