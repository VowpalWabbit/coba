import warnings
import importlib.util

from itertools import chain, islice
from collections import defaultdict
from typing import TypeVar, Iterable, Tuple, Union, Sequence

from coba.exceptions import CobaExit

def coba_exit(message:str):
    #we ignore warnings before exiting in order to make jupyter's output a little cleaner
    warnings.filterwarnings("ignore",message="To exit: use 'exit', 'quit', or Ctrl-D.")
    raise CobaExit(message) from None

class PackageChecker:

    @staticmethod
    def matplotlib(caller_name:str = None, strict:bool = True) -> None:
        """Raise ImportError with detailed error message if matplotlib is not installed.

        Functionality requiring matplotlib should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires matplotlib.
        """
        return PackageChecker._check(caller_name, "matplotlib.pyplot", "matplotlib", strict=strict)

    @staticmethod
    def vowpalwabbit(caller_name:str = None, strict:bool = True) -> None:
        """Raise ImportError with detailed error message if vowpalwabbit is not installed.

        Functionality requiring vowpalwabbit should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires vowpalwabbit.
        """
        return PackageChecker._check(caller_name, "vowpalwabbit", strict=strict)

    @staticmethod
    def pandas(caller_name:str = None, strict:bool = True) -> None:
        """Raise ImportError with detailed error message if pandas is not installed.

        Functionality requiring pandas should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires pandas.
        """
        return PackageChecker._check(caller_name, "pandas", strict=strict)

    @staticmethod
    def numpy(caller_name:str = None, strict:bool = True) -> None:
        """Raise ImportError with detailed error message if numpy is not installed.

        Functionality requiring numpy should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires numpy.
        """
        return PackageChecker._check(caller_name, "numpy", strict=strict)

    @staticmethod
    def scipy(caller_name:str = None, strict:bool = True) -> None:
        """Raise ImportError with detailed error message if scipy is not installed.

        Functionality requiring scipy should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires scipy.
        """
        return PackageChecker._check(caller_name, "scipy", strict=strict)

    @staticmethod
    def sklearn(caller_name:str = None, strict:bool = True) -> None:
        """Raise ImportError with detailed error message if sklearn is not installed.

        Functionality requiring sklearn should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires sklearn.
        """
        return PackageChecker._check(caller_name, "sklearn", "scikit-learn", strict=strict)

    @staticmethod
    def torch(caller_name:str = None, strict:bool = True) -> None:
        """Raise ImportError with detailed error message if torch is not installed.

        Functionality requiring torch should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires torch.
        """
        return PackageChecker._check(caller_name, "torch", strict=strict)

    @staticmethod
    def cloudpickle(caller_name:str = None, strict:bool = True) -> None:
        """Raise ImportError with detailed error message if cloudpickle is not installed.

        Functionality requiring cloudpickle should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires torch.
        """
        return PackageChecker._check(caller_name, "cloudpickle", strict=strict)

    def _check(caller_name:str, module_name:str, pkg_name:str = None, strict:bool = True):
        """
        Remarks:
            This pattern borrows heavily from sklearn. As of 6/20/2020 sklearn code could be found
            at https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/__init__.py
        """
        if importlib.util.find_spec(module_name) is not None:
            return True
        elif strict:
            pkg_name = pkg_name or module_name
            coba_exit(f"ERROR: {caller_name} requires the {pkg_name} package. You can install this package via `pip install {pkg_name}`.")
        else:
            return False

class KeyDefaultDict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError( key )
        else:
            value = self.default_factory(key)
            self[key] = value
            return value

_T = TypeVar("_T")
def peek_first(items: Iterable[_T], n:int=1) -> Tuple[Union[_T,Sequence[_T]], Iterable[_T]]:
    items = iter(items)
    first = list(islice(items,n))

    items = [] if not first else chain(first,items)
    first = None if not first else first[0] if n==1 else first

    return first, items
