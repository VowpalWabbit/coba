"""Global class registration and creation.

This functionality both makes it possible for other packages to easily extend COBA,
and for the core functionality to specify classes creation recipes in config files.
"""

from itertools import repeat
from importlib import reload
from typing import Dict, Any, Callable, Union, Tuple, Mapping

from coba.backports import entry_points
from coba.exceptions import CobaException

def coba_registration(name:str) -> Callable[[type],type]:

    def registration_decorator(cls: type) -> type:
        CobaRegistry.register(name, cls)
        return cls

    return registration_decorator

class CobaRegistry_meta(type):
    """We use a meta type so that we can have a class level property."""

    def __init__(cls, *args, **kwargs):
        cls._registry: Dict[str,type] = {}
        cls._setstate: set = set()
        cls._endpoints_loaded = False

    def _load_endpoints(cls):
        if not cls._endpoints_loaded:
            cls._endpoints_loaded = True
            for ep in entry_points(group='coba.register'):
                reload(ep.load()) #we use reload in case the registry has been cleared at some point

    @property
    def registry(cls) -> Mapping[str,type]:
        cls._load_endpoints()
        return cls._registry

    @property
    def setstate(cls) -> set:
        cls._load_endpoints()
        return cls._setstate

class CobaRegistry(metaclass=CobaRegistry_meta):

    @classmethod
    def clear(cls) -> None:
        cls._endpoints_loaded = False
        cls._registry.clear()

    @classmethod
    def register(cls, name:str, tipe:type) -> None:
        if name not in cls.registry:
            cls._registry[name] = tipe
            cls._registry[tipe] = name
            if hasattr(tipe,'__getstate__') and hasattr(tipe,'__setstate__'):
                cls._setstate.add(name)
        elif cls._registry[name] != tipe:
            raise CobaException(f"The class `{tipe.__name__}` has already been registered for '{name}'")

class JsonMakerV1:

    def __init__(self, registry:Dict[str,type]) -> None:
        self._registry = registry

    def make(self, recipe:Any) -> Any:
        name   = ""
        args   = None
        kwargs = None
        method = "singular"

        if not self._is_valid_recipe(recipe):
            raise CobaException(f"Invalid recipe {str(recipe)}")

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

                if isinstance(implicit_args, dict) and not self._is_known_recipe(implicit_args):
                    kwargs = implicit_args
                else:
                    args = implicit_args

        if method == "singular":
            return self._construct_single(recipe, name, args, kwargs)
        else:
            if not isinstance(kwargs, list): kwargs = repeat(kwargs)
            if not isinstance(args  , list):   args = repeat(args)
            return [ self._construct_single(recipe, name, a, k) for a,k in zip(args, kwargs) ]

    def _is_valid_recipe(self, recipe:Any) -> bool:

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

    def _is_known_recipe(self, recipe:Any) -> bool:

        name = None

        if isinstance(recipe, str):
            name = recipe

        if isinstance(recipe, dict):
            name = recipe.get('name', None) or [key for key in recipe if key not in ["name", "args", "kwargs", "method"]][0]

        return name in self._registry

    def _construct_or_return(self, item:Any):
        return self.make(item) if self._is_known_recipe(item) else item

    def _construct_single(self, recipe, name, args, kwargs) -> Any:
        try:
            if args is not None and not isinstance(args, list): args = [args]

            if args is not None:
                args = [ self._construct_or_return(a) for a in args ]

            if kwargs is not None:
                kwargs = { k:self._construct_or_return(v) for k,v in kwargs.items() }

            if args is not None and kwargs is not None:
                return self._registry[name](*args, **kwargs)

            elif args is not None and kwargs is None:
                try:
                    return self._registry[name](*args)
                except TypeError as e:
                    return self._registry[name](args)

            elif args is None and kwargs is not None:
                return self._registry[name](**kwargs)
            else:
                return self._registry[name]()

        except KeyError:
            raise CobaException(f"Unknown recipe {str(recipe)}")

        except Exception as e:
            raise CobaException(f"Unable to create recipe {str(recipe)}")

class JsonMakerV2:

    #In Python the distinction between arg, args and kwargs is made using (), [], {}, *, and **
    #In json we only have [] and {} so there is going to be some ambiguity unless we add syntax.

    ##########Construction Syntax#############

    #I think this might be the best... It has low syntax overhead and clear semantic meaning.
    #The biggest problem is that it can have a ton of ambiguity. To work around that we can offer
    #the wordier "*"/"**" notation at the bottom.

    #arg    = <str,float,list,dict,obj>
    #args   = [arg, ... ]
    #kwargs = {<str>:arg, ...}

    #no-arg          = < "class" | {"class":[]} >
    #arg             = { "class": <str|float>           }
    #args            = { "class": args                  }
    #kwargs          = { "class": kwargs                }
    #args and kwargs = { "class": [arg,...,"**",kwargs] }

    #High syntax overhead but no ambiguity. I think ambiguity will be rare, so probably not worth the overhead.
    # {"name":"ClassName", "args":[], "kwargs":{} }

    #Low syntax overhead, no ambiguity but I think the semantic meaning is less clear.
    #["ClassName",args,kwargs]

    #This syntax I am confident I don't like. I don't like that the levels are inconsistent (i.e., outer kw are inner args).
    #Plus, determining which key in the top-level dict is a classname and which is a kw seems tricky.
    #{ class:[], kw:arg, kw:arg }

    ###################Foreach Syntax#############################

    #I think I like this syntax more than the option below.
    #This gives the entire for collection up front and filling
    #in args is only a matter of string checks
    # {"class":"$"       , "for": [1,2,3,4]             }
    # {"class":"$"       , "for": {"zip":[[1,2],[3,4]]} }
    # {"class":{"kw":"$"}, "for": [1,2,3,4]             }

    #This has low syntax overhead but parsing and semantic reading much more difficult.
    # {class: {"for":[1,2,3,4]}             } (arg foreach construction)
    # {class: [{"for":[1,2]},{"for":[3,4]}] } (args foreach construction)
    # {class: {kw:{"for":[1,2]}}            } (kwargs foreach construction)

    def __init__(self, registry:Dict[str,type] = None) -> None:
        self._registry = registry or CobaRegistry.registry

    def make(self, recipe:Union[dict,str], strict:bool = True) -> Any:
        if isinstance(recipe,str) and recipe in self._registry:
            return self._registry[recipe]()

        if isinstance(recipe,dict):
            has_for = 'for' in recipe
            if len(recipe)-(has_for) == 1:
                item_cls = next(iter(recipe.keys()-{'for'}))
                klass    = self._registry.get(item_cls)
                if klass:
                    if has_for:
                        items = []
                        for value in self.make(recipe['for'], strict=False):
                            filled_args = self._fill_template(recipe[item_cls], value)
                            args,kwargs = self._construct_args(filled_args)
                            items.append(klass(*args,**kwargs))
                        return items
                    else:
                        args,kwargs = self._construct_args(recipe[item_cls])
                        return klass(*args,**kwargs)

        if not strict:
            return recipe

        raise CobaException(f"We were unable to make {recipe}.")

    def _construct_args(self, args:Union[list,dict,str,int,float,None]) -> Tuple[list,dict]:
        if isinstance(args,dict):
            return [], {k:self.make(v,False)  for k,v in args.items()}
        if isinstance(args,list):
            if len(args) >= 2 and args[-2] == "**":
               args,kw = args[:-2],self._construct_args(args[-1])[1]
            else:
                kw = {}
            return [ self.make(a,False) for a in args], kw
        else:
            return [ self.make(args,False) ], {}

    def _fill_template(self, temp:Union[list,dict,str,int,float,None], value:Union[list,dict,str,int,float,None]) -> Union[list,dict,str,int,float,None]:
        if isinstance(temp,dict):
            return { self._fill_template(k,value):self._fill_template(v,value) for k,v in temp.items() }
        if isinstance(temp,list):
            return [ self._fill_template(i, value) for i in temp ]
        if isinstance(temp,str) and temp.startswith("$"):
            return value if temp=="$" else value[int(temp[1:])]
        return temp
