import collections.abc

from itertools import product
from typing import Sequence, Any, overload, Iterable, Union, Dict

from coba.registry import JsonMakerV1, CobaRegistry, JsonMakerV2
from coba.pipes import Source, JsonDecode, UrlSource, Pipes
from coba.exceptions import CobaException

from coba.environments.primitives import Environment

class EnvironmentsTemplateV1(Source[Sequence[Environment]]):

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

class EnvironmentsTemplateV2(Source[Sequence[Environment]]):

    @overload
    def __init__(self, source:Source[Iterable[str]], **user_vars) -> None: ...

    @overload
    def __init__(self, url:str, **user_vars) -> None: ...

    def __init__(self,arg, **user_vars) -> None:
        self._source = UrlSource(arg) if isinstance(arg,str) else arg
        self._user_variables = { f"${k}" if k[0] != "$" else k :v for k,v in user_vars.items() }

    def read(self) -> Sequence[Environment]:
        definition: dict = JsonDecode().filter('\n'.join(self._source.read()))
        
        old_variables = { **definition.get("variables",{}), **self._user_variables}
        new_variables = self._fill(old_variables,old_variables)
        num_fillings  = 0

        while old_variables != new_variables:
            num_fillings += 1
            old_variables = new_variables
            new_variables = self._fill(old_variables,old_variables)

            if num_fillings > 4:
                raise CobaException("There appears to be an infinite loop in the template.")
        
        variables = { k: JsonMakerV2(CobaRegistry.registry).make(v,strict=False) for k,v in new_variables.items() }

        recipes = definition['environments']
        recipes = recipes if isinstance(recipes, list) else [recipes]
        recipes = self._fill(recipes, variables)

        missing = self._missing(recipes)
        if missing: raise CobaException(f"The following variables were not defined: [{','.join(missing)}]")

        environments = []
        for recipe in recipes:
            environments.extend(self._make(recipe))
        return environments
    
    def _make(self, item:Union[str,list,dict] ) -> Sequence[Any]:
        
        result = item

        if isinstance(item, str):
            result = JsonMakerV2(CobaRegistry.registry).make(item)

        if isinstance(item, dict):
            result = JsonMakerV2(CobaRegistry.registry).make(item)

        if isinstance(item, list):
            pieces = [ self._make(i) for i in item]

            if hasattr(pieces[0][0],'read') and hasattr(pieces[1][0],'filter'):
                result = [ Pipes.join(s,*f) for s in pieces[0] for f in product(*pieces[1:])]
            else:
                result = sum(pieces,[])
    
        if result is None:
            raise CobaException(f"We were unable to construct {item} in the given environment definition file.")

        return result if isinstance(result, collections.abc.Sequence) else [result]

    def _fill(self, target: Any, variables:Dict[str,Any]):
        if isinstance(target,dict):
            return { k: self._fill(v,variables) for k,v in target.items() }
        if isinstance(target,list):
            return [ self._fill(i,variables) for i in target ]
        if isinstance(target,str) and target in variables:
            return variables[target]
        return target

    def _missing(self, target: Any) -> Sequence[str]:
        if isinstance(target,dict):
            return sum([self._missing(v) for v in target.values()],[])
        if isinstance(target,list):
            return sum([self._missing(i) for i in target],[])
        if isinstance(target,str) and len(target) > 1 and target.startswith("$") and not target[1:].isdigit():
            return [target]
        return []