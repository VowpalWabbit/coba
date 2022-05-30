import collections.abc

from itertools import product
from typing import Sequence, Any, overload, Iterable, Union, Dict

from coba.registry import JsonMakerV1, CobaRegistry, JsonMakerV2
from coba.pipes import Source, JsonDecode, UrlSource, Pipes
from coba.exceptions import CobaException

from coba.environments.primitives import Environment

class EnvironmentsDefinitionV1(Source[Sequence[Environment]]):

    @overload
    def __init__(self, filesource:Source[Iterable[str]]) -> None: ...

    @overload
    def __init__(self, filename:str) -> None: ...

    def __init__(self,arg) -> None:

        self._source = UrlSource(arg) if isinstance(arg,str) else arg

    def read(self) -> Sequence[Environment]:

        definitions: dict = JsonDecode().filter('\n'.join(self._source.read()))

        variables = { k: self._construct_via_method(v) for k,v in definitions.get("variables",{}).items() }

        def _construct(item:Any) -> Sequence[Any]:
            result = None

            if isinstance(item, str) and item in variables:
                result = variables[item]

            if isinstance(item, str) and item not in variables:
                result = self._construct_via_method(item)

            if isinstance(item, dict):
                result = self._construct_via_method(item)

            if isinstance(item, list):
                pieces = list(map(_construct, item))

                if hasattr(pieces[0][0],'read'):
                    result = [ Pipes.join(s,*f) for s in pieces[0] for f in product(*pieces[1:])]
                else:
                    result = sum(pieces,[])

            if result is None:
                raise CobaException(f"We were unable to construct {item} in the given environment definition file.")

            return result if isinstance(result, collections.abc.Sequence) else [result]

        if not isinstance(definitions['environments'], list): definitions['environments'] = [definitions['environments']]

        return [ environment for recipe in definitions['environments'] for environment in _construct(recipe)]
    
    def _construct_via_method(self, item):
        return JsonMakerV1(CobaRegistry.registry).make(item)

class EnvironmentsDefinitionV2(Source[Sequence[Environment]]):

    @overload
    def __init__(self, filesource:Source[Iterable[str]]) -> None: ...

    @overload
    def __init__(self, filename:str) -> None: ...

    def __init__(self,arg) -> None:
        self._source = UrlSource(arg) if isinstance(arg,str) else arg

    def read(self) -> Sequence[Environment]:
        definition: dict = JsonDecode().filter('\n'.join(self._source.read()))
        variables = { k: JsonMakerV2(CobaRegistry.registry).make(v,strict=True) for k,v in definition.get("variables",{}).items() }

        recipes = definition['environments']
        recipes = recipes if isinstance(recipes, list) else [recipes]

        environments = []
        for recipe in recipes:
            environments.extend(self._make(recipe, variables))
        return environments
    
    def _make(self, item:Union[str,list,dict], vars:Dict[str,Any]) -> Sequence[Any]:
        result = None

        if isinstance(item, str) and item in vars:
            result = vars[item]

        if isinstance(item, str) and item not in vars:
            result = JsonMakerV2(CobaRegistry.registry).make(item)

        if isinstance(item, dict):
            result = JsonMakerV2(CobaRegistry.registry).make(item)

        if isinstance(item, list):
            pieces = [ self._make(i,vars) for i in item]

            if hasattr(pieces[0][0],'read') and hasattr(pieces[1][0],'filter'):
                result = [ Pipes.join(s,*f) for s in pieces[0] for f in product(*pieces[1:])]
            else:
                result = sum(pieces,[])

        if result is None:
            raise CobaException(f"We were unable to construct {item} in the given environment definition file.")

        return result if isinstance(result, collections.abc.Sequence) else [result]
