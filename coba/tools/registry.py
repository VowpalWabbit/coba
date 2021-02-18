"""Global class registration and creation.

This functionality both makes it possible for other packages to easily extend COBA,
and for the core functionality to specify classes creation recipes in config files.
"""

from itertools import repeat
from importlib_metadata import entry_points #type: ignore
from typing import Dict, Any, Callable

def coba_registry_class(name:str) -> Callable[[type],type]:

    def registration_decorator(cls: type) -> type:
        CobaRegistry.register(name, cls)
        return cls

    return registration_decorator

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
        method   = "singular"

        if not cls._is_valid_recipe(recipe):
            raise Exception(f"Invalid recipe {str(recipe)}")

        if isinstance(recipe, str):
            name = recipe
    
        if isinstance(recipe, dict):
            mutable_recipe = dict(recipe)

            method = mutable_recipe.pop("method", "singular")
            name   = mutable_recipe.pop("name"  , ""        )
            args   = mutable_recipe.pop("args"  , None      )
            kwargs = mutable_recipe.pop("kwargs", None      )

            if len(mutable_recipe) == 1:
                name, implicit_args = list(mutable_recipe.items())[0]

                if isinstance(implicit_args, dict):
                    kwargs = implicit_args
                else:
                    args = implicit_args

        if method == "singular":
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
            keywords  = ["name", "args", "kwargs", "method"]
            freewords = [ key for key in recipe if key not in keywords ]

            implicit_arg       = None if len(freewords) != 1 else recipe[freewords[0]]
            implicit_is_args   = implicit_arg is not None and not isinstance(implicit_arg,dict)
            implicit_is_kwargs = implicit_arg is not None and     isinstance(implicit_arg,dict)

            no_unknown_words = len(freewords) <= 1 
            contains_name    = "name" in keywords or len(freewords) == 1
            name_collision   = "name" in recipe and len(freewords) == 1
            args_collision   = "args" in recipe and implicit_is_args 
            kwargs_collision = "kwargs" in recipe and implicit_is_kwargs
            no_collisions    = not any([name_collision, args_collision, kwargs_collision])

            return no_unknown_words and contains_name and no_collisions

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