from typing import Dict, Any
from importlib_metadata import entry_points

registry: Dict[str, Any] = {}

def register_class(name: str, cls: Any) -> None:
    registry[name] = cls

def retrieve_class(name:str) -> Any:

    if len(registry) == 0:
        for eps in entry_points()['coba.register']:
            eps.load()

    return registry[name]

def create_class(creation_script: Any) -> Any:
    
    if isinstance(creation_script, str):
        return retrieve_class(creation_script)()