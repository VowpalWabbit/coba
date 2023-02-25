from typing import Iterable, Mapping

from coba.pipes import Filter
from coba.environments.primitives import Interaction

class SimpleEnvironment(Filter[Iterable[Mapping], Iterable[Interaction]]):
    def filter(self, items: Iterable[Mapping]) -> Iterable[Interaction]:
        yield from map(Interaction.from_dict,items)
