import collections.abc

from itertools import product
from typing import Sequence, Dict, Any

from coba.registry import CobaRegistry
from coba.pipes import Filter
from coba.exceptions import CobaException

from coba.environments.filters import FilteredEnvironment
from coba.environments.simulated import SimulatedEnvironment

class EnvironmentDefinitionFileV1(Filter[Dict[str,Any], Sequence[SimulatedEnvironment]]):

    def filter(self, config: Dict[str,Any]) -> Sequence[SimulatedEnvironment]:

        variables = { k: CobaRegistry.construct(v) for k,v in config.get("variables",{}).items() }

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
                    result = [ FilteredEnvironment(s, *f) for s in pieces[0] for f in product(*pieces[1:])]
                else:
                    result = sum(pieces,[])

            if result is None:
                raise CobaException(f"We were unable to construct {item} in the given environment definition file.")

            return result if isinstance(result, collections.abc.Sequence) else [result]

        if not isinstance(config['environments'], list): config['environments'] = [config['environments']]

        return [ environment for recipe in config['environments'] for environment in _construct(recipe)]
