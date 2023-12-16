from typing import Sequence, Mapping, Callable

from coba.exceptions import CobaException
from coba.primitives.semantic import Action

Reward = Callable[[Action],float]

def extract_shape(given:Action, comparison:Action, is_comparison_list:bool=False):

    given_ndim = getattr(given,'ndim',-1)

    if given_ndim == -1:
        compare_action = given
        output_shape = None

    elif given_ndim == 0:
        compare_action = given.item()
        output_shape = ()

    else:
        expected_ndims = isinstance(comparison,(list,tuple)) + is_comparison_list

        if expected_ndims == 0:
            compare_action = given.item()
            output_shape = given.shape
        else:
            output_ndims   = given_ndim-expected_ndims
            compare_action = given[(0,)*output_ndims].tolist()
            output_shape   = (1,)*output_ndims

    return compare_action, output_shape

def create_shape(value:float, shape):

    if shape is None:
        return value
    else:
        try:
            t = torch # type: ignore
        except:
            t = globals()['torch'] = __import__('torch')

        #`t` could also be numpy
        return t.full(shape,value)

class L1Reward:
    __slots__ = ('_argmax',)

    def __init__(self, argmax: float) -> None:
        self._argmax = argmax if not hasattr(argmax,'ndim') else argmax.item()

    def __call__(self, action: float) -> float:
        #due to broadcasting this automatically
        #handles shaping when action is Tensor
        return -abs(action-self._argmax)

    def __reduce__(self):
        #this makes the pickle smaller
        return L1Reward, (self._argmax,)

    def __eq__(self, o: object) -> bool:
        return isinstance(o,L1Reward) and o._argmax == self._argmax

class BinaryReward:
    __slots__ = ('_argmax','_value')
    def __init__(self, argmax: Action, value:float=1.) -> None:
        self._argmax = argmax if not hasattr(argmax,'ndim') else argmax.tolist()
        self._value  = value

    def __call__(self, action: Action) -> float:
        argmax = self._argmax
        comparable,shape = extract_shape(action,argmax)
        value = self._value if argmax==comparable else 0
        return create_shape(value,shape)

    def __reduce__(self):
        #this makes the pickle smaller
        return BinaryReward, (self._argmax,self._value)

    def __eq__(self, o: object) -> bool:
        return o == self._argmax or (isinstance(o,BinaryReward) and o._argmax == self._argmax)

class HammingReward:
    __slots__ = ('_argmax',)

    def __init__(self, argmax: Sequence[Action]) -> None:
        self._argmax = argmax if not hasattr(argmax,'ndim') else argmax.tolist()

    def __call__(self, action: Sequence[Action]) -> float:

        argmax = self._argmax
        comparable,shape = extract_shape(action,argmax[0],True)

        n_intersect = 0

        for a in comparable: n_intersect += a in self._argmax
        n_union = len(argmax) + len(comparable) - n_intersect

        value = n_intersect/n_union

        return create_shape(value,shape)

    def __reduce__(self):
        #this makes the pickle smaller
        return HammingReward, (tuple(self._argmax),)

class SequenceReward:
    __slots__ = ('_actions','_rewards')

    def __init__(self, actions: Sequence[Action], rewards: Sequence[float]) -> None:
        if len(actions) != len(rewards):
            raise CobaException("The given actions and rewards did not line up.")

        self._actions = actions if not hasattr(actions,'ndim') else actions.tolist()
        self._rewards = rewards if not hasattr(rewards,'ndim') else rewards.tolist()

    def __call__(self, action: Action) -> float:
        comparable,shape = extract_shape(action,self._actions[0])
        value = self._rewards[self._actions.index(comparable)]
        return create_shape(value,shape)

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
        comparable,shape = extract_shape(action,next(iter(self._mapping.keys())))
        comparable = tuple(comparable) if isinstance(comparable,list) else comparable
        value = self._mapping[comparable]
        return create_shape(value,shape)

    def __reduce__(self):
        #this makes the pickle smaller
        return MappingReward, (self._mapping,)

    def __eq__(self, o: object) -> bool:
        return o == self._mapping or (isinstance(o,MappingReward) and o._mapping == self._mapping)

class ProxyReward:
    __slots__ = ('_reward', '_mapping')
    def __init__(self, reward: Reward, mapping: Mapping):
        self._reward = reward
        self._mapping = mapping

    def __call__(self, action:Action):
        comparable,shape = extract_shape(action,next(iter(self._mapping.keys())))
        comparable = tuple(comparable) if isinstance(comparable,list) else comparable
        value = self._reward(self._mapping[comparable])
        return create_shape(value,shape)
