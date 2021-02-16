"""Global class registration and creation.

This functionality both makes it possible for other packages to easily extend COBA,
and for the core functionality to specify classes creation recipes in config files.
"""

import collections

from importlib_metadata import entry_points #type: ignore
from typing import Dict, Any

class CobaRegistry:
    
    _registry: Dict[str,type] = {}

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()

    @classmethod
    def register(cls, name:str, tipe:type) -> None:
        cls._registry[name] = tipe

    @classmethod
    def retrieve(cls, name:str) -> type:
        if len(cls._registry) == 0:
            for eps in entry_points()['coba.register']:
                eps.load()

        return cls._registry[name]

    @classmethod
    def construct(cls, recipe:Any) -> Any:
        name   = ""
        args   = []
        kwargs = {}

        if isinstance(recipe, str):
            name = recipe
    
        if isinstance(recipe, collections.Mapping):
            mutable_recipe = dict(recipe)

            name   = mutable_recipe.pop("name"  , "")
            args   = mutable_recipe.pop("args"  , [])
            kwargs = mutable_recipe.pop("kwargs", {})

            if ( 
                len(mutable_recipe) > 1 or 
                len(mutable_recipe) == 1 and (name != "" or args != []) or 
                len(mutable_recipe) == 0 and name == ""
            ):
                raise Exception(f"Invalid recipe {str(recipe)}")

            if len(mutable_recipe) == 1:
                name,args = list(mutable_recipe.items())[0]

        if not isinstance(args, list):
            args = [args]

        try:
            return cls.retrieve(name)(*args, **kwargs)
        except KeyError:
            raise Exception(f"Unknown recipe {str(recipe)}")