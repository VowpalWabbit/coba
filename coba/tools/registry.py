"""Global class registration and creation.

This functionality both makes it possible for other packages to easily extend COBA,
and for the core functionality to specify classes creation recipes in config files.
"""

from itertools import repeat
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
        args   = None
        kwargs = None
        make   = "singular"

        if not cls._is_valid_recipe(recipe):
            raise Exception(f"Invalid recipe {str(recipe)}")

        if isinstance(recipe, str):
            name = recipe
    
        if isinstance(recipe, dict):
            mutable_recipe = dict(recipe)

            make   = mutable_recipe.pop("make"  , "singular")
            name   = mutable_recipe.pop("name"  , ""        )
            args   = mutable_recipe.pop("args"  , None      )
            kwargs = mutable_recipe.pop("kwargs", None      )

            if len(mutable_recipe) == 1:
                name,args = list(mutable_recipe.items())[0]

        if make == "singular":
            return cls._construct_single(recipe, name, args, kwargs)
        else:
            if not isinstance(kwargs, list): kwargs = repeat(kwargs)
            if not isinstance(args  , list):   args = repeat(args)
            return [ cls._construct_single(recipe, name, a, k) for a,k in zip(args, kwargs) ]
        
    @classmethod
    def _is_valid_recipe(cls, recipe:Any) -> bool:

        if isinstance(recipe, str):
            return True

        if isinstance(recipe, dict):
            keywords  = ["name", "args", "kwargs", "make"]
            freewords = [ key for key in recipe if key not in keywords ]

            no_unknown_words = len(freewords) <= 1 
            contains_name    = "name" in keywords or len(freewords) == 1
            name_collision   = "name" in recipe and len(freewords) == 1
            args_collision   = "args" in recipe and len(freewords) == 1
            no_collision     = not name_collision and not args_collision

            return no_unknown_words and contains_name and no_collision

        return False

    @classmethod
    def _construct_single(cls, recipe, name, args, kwargs) -> Any:
        try:
            if args is not None and not isinstance(args, list): args = [args]
            
            if args is not None and kwargs is not None:
                return cls.retrieve(name)(*args, **kwargs)
            elif args is not None and kwargs is None:
                return cls.retrieve(name)(*args)
            elif args is None and kwargs is not None:
                return cls.retrieve(name)(**kwargs)
            else:
                return cls.retrieve(name)()

        except KeyError:
            raise Exception(f"Unknown recipe {str(recipe)}")