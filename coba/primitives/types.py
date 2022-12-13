from typing import Sequence

class Categorical(str):
    __slots__ = ('levels','onehot')
    
    def __new__(cls, value:str, levels: Sequence[str]) -> str:
        return str.__new__(Categorical,value)
    
    def __init__(self, value:str, levels: Sequence[str]) -> None:
        self.levels = levels
        self.onehot = [0]*len(levels)
        self.onehot[levels.index(value)] = 1
        self.onehot = tuple(self.onehot)

    def __repr__(self) -> str:
        return f"Categorical('{self}',{self.levels})"
