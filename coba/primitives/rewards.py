from typing import Sequence, Mapping, Callable, overload

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

            if not is_comparison_list:
                compare_action = type(comparison)(given[(0,)*output_ndims].tolist())
            else:
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

    def __eq__(self, o: object) -> bool:
        return o == self._argmax or \
            (isinstance(o,BinaryReward) and\
            o._argmax == self._argmax and\
            o._value == self._value)

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

class DiscreteReward:
    __slots__ = ('_state','_default')

    @overload
    def __init__(self, actions: Sequence[Action], rewards: Sequence[float], *, default: float = 0) -> None:
        ...

    @overload
    def __init__(self, mapping: Mapping[Action,float], *, default: float = 0) -> None:
        ...

    def __init__(self, *args, default=0) -> None:
        if len(args) == 2:
            actions,rewards = args
            self._state = args
            if len(actions) != len(rewards):
                raise CobaException("The given actions and rewards did not line up.")

        else:
            self._state = args[0]
        self._default = default

    @property
    def actions(self):
        return list(self._state.keys()) if isinstance(self._state,dict) else self._state[0]

    @property
    def rewards(self):
        return list(self._state.values()) if isinstance(self._state,dict) else self._state[1]

    def __call__(self, action: Action) -> float:
        if isinstance(self._state,dict):
            comp,shape = extract_shape(action,next(iter(self._state.keys())))
            value = self._state.get(comp,self._default)
            return create_shape(value,shape)
        else:
            actions,rewards = self._state
            comp,shape = extract_shape(action,actions[0])
            value = rewards[actions.index(comp)] if comp in actions else self._default
            return create_shape(value,shape)

    def __eq__(self, o: object) -> bool:

        return o == self.rewards or \
            (isinstance(o,DiscreteReward) and\
            o.actions == self.actions and\
            o.rewards == self.rewards and\
            o._default == self._default)
