from typing import Sequence, Mapping, Callable

from coba.exceptions import CobaException
from coba.primitives.semantic import Action, Batch

Reward = Callable[[Action],float]

def argmax(actions: Sequence[Action], reward: Reward) -> Action:
    max_r = -float('inf')
    max_a = 0
    for a in actions:
        r = reward(a)
        if r > max_r:
            max_a = a
            max_r = r

    return max_a

class L1Reward:
    __slots__ = ('_label',)

    def __init__(self, action: float) -> None:
        self._label = action

    def __call__(self, action: float) -> float:
        return action-self._label if self._label > action else self._label-action

    def __reduce__(self):
        #this makes the pickle smaller
        return L1Reward, (self._label,)

    def __eq__(self, o: object) -> bool:
        return isinstance(o,L1Reward) and o._label == self._label

class HammingReward:
    __slots__ = ('_labels',)

    def __init__(self, labels: Sequence[Action]) -> None:
        self._labels = set(labels)

    def __call__(self, arg: Sequence[Action]) -> float:
        return len(self._labels.intersection(arg))/len(self._labels.union(arg))

    def __reduce__(self):
        #this makes the pickle smaller
        return HammingReward, (tuple(self._labels),)

class BinaryReward:
    __slots__ = ('_argmax','_value')

    def __init__(self, action: Action, value:float=1) -> None:
        self._argmax = action
        self._value  = value

    def __call__(self, action: Action) -> float:
        return self._value if self._argmax==action else 0

    def __reduce__(self):
        #this makes the pickle smaller
        return BinaryReward, (self._argmax,self._value)

    def __eq__(self, o: object) -> bool:
        return o == self._argmax or (isinstance(o,BinaryReward) and o._argmax == self._argmax)

class SequenceReward:
    __slots__ = ('_actions','_rewards')

    def __init__(self, actions: Sequence[Action], rewards: Sequence[float]) -> None:
        if len(actions) != len(rewards):
            raise CobaException("The given actions and rewards did not line up.")

        self._actions = actions
        self._rewards = rewards

    def __call__(self, action: Action) -> float:
        return self._rewards[self._actions.index(action)]

    def __reduce__(self):
        #this makes the pickle smaller
        return SequenceReward, (self._actions,self._rewards)

    def __eq__(self, o: object) -> bool:
        return o == self._rewards or (isinstance(o,SequenceReward) and o._actions == self._actions and o._rewards == self._rewards)

class MappingReward:
    __slots__ = ('_mapping',)

    def __init__(self, mapping: Mapping[Action,float]) -> None:
        self._mapping = mapping

    def __call__(self, action: Action) -> float:
        return self._mapping[action]

    def __reduce__(self):
        #this makes the pickle smaller
        return MappingReward, (self._mapping,)

    def __eq__(self, o: object) -> bool:
        return o == self._mapping or (isinstance(o,MappingReward) and o._mapping == self._mapping)

class BatchReward(Batch):
    def __call__(self, actions: Sequence[Action]) -> Sequence[float]:
        return list(map(lambda r,a: r(a), self, actions))
