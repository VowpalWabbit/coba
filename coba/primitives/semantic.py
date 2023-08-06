from typing import Sequence, Mapping

Context = None|str|int|float|Sequence|Mapping
Action  = None|str|int|float|Sequence|Mapping
Actions = None|Sequence[Action]

class Batch(list):
    pass
