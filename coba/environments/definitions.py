import collections.abc

from itertools import product
from typing import Sequence, Any, overload, Iterable

from coba.registry import CobaRegistry
from coba.pipes import Source, JsonDecode, UrlSource, Pipes
from coba.exceptions import CobaException

from coba.environments.primitives import Environment

class EnvironmentDefinitionFileV1(Source[Sequence[Environment]]):

    @overload
    def __init__(self, filesource:Source[Iterable[str]]) -> None: ...

    @overload
    def __init__(self, filename:str) -> None: ...

    def __init__(self,arg) -> None:

        self._source = UrlSource(arg) if isinstance(arg,str) else arg

    def read(self) -> Sequence[Environment]:

        definitions: dict = JsonDecode().filter('\n'.join(self._source.read()))

        variables = { k: CobaRegistry.construct(v) for k,v in definitions.get("variables",{}).items() }

        def _construct(item:Any) -> Sequence[Any]:
            result = None

            if isinstance(item, str) and item in variables:
                result = variables[item]

            if isinstance(item, str) and item not in variables:
                result = CobaRegistry.construct(item)

            if isinstance(item, dict):
                result = CobaRegistry.construct(item)

            if isinstance(item, list):
                pieces = list(map(_construct, item))
                
                if hasattr(pieces[0][0],'read'):
                    result = [ Pipes.join(s, f) for s in pieces[0] for f in product(*pieces[1:])]
                else:
                    result = sum(pieces,[])

            if result is None:
                raise CobaException(f"We were unable to construct {item} in the given environment definition file.")

            return result if isinstance(result, collections.abc.Sequence) else [result]

        if not isinstance(definitions['environments'], list): definitions['environments'] = [definitions['environments']]

        return [ environment for recipe in definitions['environments'] for environment in _construct(recipe)]