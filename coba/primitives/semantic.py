from typing import Union, Sequence, Mapping

Context = Union[None, str, int, float, Sequence, Mapping]
Action  = Union[str, int, float, Sequence, Mapping]
Actions = Sequence[Action]

class Batch(list):
    pass
