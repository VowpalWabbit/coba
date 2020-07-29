"""Simple one-off utility methods with no clear home."""

import copy
import json
import collections

from itertools import repeat
from typing import Union, Dict, Any, cast, List, Mapping, MutableMapping, Optional
from pathlib import Path

def check_matplotlib_support(caller_name: str) -> None:
    """Raise ImportError with detailed error message if matplotlib is not installed.

    Functionality requiring the matplotlib module should call this helper and then lazily import.

    Args:    
        caller_name: The name of the caller that requires matplotlib.

    Remarks:
        This pattern borrows heavily from sklearn. As of 6/20/2020 sklearn code could be found
        at https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/__init__.py
    """
    try:
        import matplotlib # type: ignore
    except ImportError as e:
        raise ImportError(
            caller_name + " requires matplotlib. You can "
            "install matplotlib with `pip install matplotlib`."
        ) from e

def check_vowpal_support(caller_name: str) -> None:
    """Raise ImportError with detailed error message if vowpalwabbit is not installed.

    Functionality requiring the vowpalwabbit module should call this helper and then lazily import.

    Args:    
        caller_name: The name of the caller that requires matplotlib.

    Remarks:
        This pattern was inspired by sklearn (see coba.utilities.check_matplotlib_support for more information).
    """
    try:
        import vowpalwabbit # type: ignore
    except ImportError as e:
        raise ImportError(
            caller_name + " requires vowpalwabbit. You can "
            "install vowpalwabbit with `pip install vowpalwabbit`."
        ) from e

class JsonTemplating():
    
    @staticmethod
    def parse(json_val:Union[str, Any]):
        
        root = json.loads(json_val) if isinstance(json_val, str) else json_val

        if "templates" in root:

            templates: Dict[str,Dict[str,Any]] = root.pop("templates")
            nodes    : List[Any]               = [root]
            scopes   : List[Dict[str,Any]]     = [{}]

            def materialize_template(document: MutableMapping[str,Any], template: Mapping[str,Any]):

                for key in template:
                    if key in document:
                        if isinstance(template[key], collections.Mapping) and isinstance(template[key], collections.Mapping):
                            materialize_template(document[key],template[key])
                    else:
                        document[key] = template[key]
            
            def materialize_variables(document: MutableMapping[str,Any], variables: Mapping[str,Any]):
                for key in document:
                    if isinstance(document[key],str) and document[key] in variables:
                        document[key] = variables[document[key]]

            while len(nodes) > 0:
                node  = nodes.pop()
                scope = scopes.pop().copy()  #this could absolutely be made more memory-efficient if needed

                if isinstance(node, collections.MutableMapping):

                    if "template" in node and node["template"] not in templates:
                        raise Exception(f"We were unable to find template '{node['template']}'.")

                    keys      = list(node.keys())
                    template  = templates[node.pop("template")] if "template" in node else cast(Dict[str,Any], {})
                    variables = { key:node.pop(key) for key in keys if key.startswith("$") }

                    template = copy.deepcopy(template)
                    scope.update(variables)

                    materialize_template(node, template)
                    materialize_variables(node, scope)

                    for child_node, child_scope in zip(node.values(), repeat(scope)):
                        nodes.append(child_node)
                        scopes.append(child_scope)
                
                if isinstance(node, collections.Sequence) and not isinstance(node, str):
                    for child_node, child_scope in zip(node, repeat(scope)):
                        nodes.append(child_node)
                        scopes.append(child_scope)
        
        return root

class coba_config:

    _search_paths = [Path("./.coba"), Path.home() / ".coba"]

    openml_api_key : Optional[str]
    cache_directory: Optional[str]

    try:

        config = {}

        for potential_path in _search_paths:
            if potential_path.exists():
                with open(potential_path) as fs:
                    config = json.load(fs)
                break

        openml_api_key  = config.get("openml_api_key", None)
        cache_directory = config.get("cache_directory", None)

    except:
        openml_api_key = None
        cache_directory = None