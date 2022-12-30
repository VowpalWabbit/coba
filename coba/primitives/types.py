from typing import Sequence

class Categorical(str):
    __slots__ = ('levels','as_int','as_onehot')

    def __new__(cls, value:str, levels: Sequence[str]) -> str:
        return str.__new__(Categorical,value)

    def __init__(self, value:str, levels: Sequence[str]) -> None:
        self.levels = levels
        self.as_int = levels.index(value)
        onehot = [0]*len(levels)
        onehot[self.as_int] = 1
        self.as_onehot = tuple(onehot)

    def __repr__(self) -> str:
        return f"Categorical('{self}',{self.levels})"
