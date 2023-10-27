import warnings
import importlib

from itertools import chain, islice
from collections import defaultdict
from typing import TypeVar, Iterable, Tuple, Union, Sequence, Any, Optional

from coba import CobaRandom
from coba.exceptions import CobaExit
from coba.random import choice


def coba_exit(message:str):
    #we ignore warnings before exiting in order to make jupyter's output a little cleaner
    warnings.filterwarnings("ignore",message="To exit: use 'exit', 'quit', or Ctrl-D.")
    raise CobaExit(message) from None

class PackageChecker:

    @staticmethod
    def matplotlib(caller_name: str) -> None:
        """Raise ImportError with detailed error message if matplotlib is not installed.

        Functionality requiring matplotlib should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires matplotlib.
        """
        PackageChecker._handle_import_error(caller_name, "matplotlib.pyplot", "matplotlib")

    @staticmethod
    def vowpalwabbit(caller_name: str) -> None:
        """Raise ImportError with detailed error message if vowpalwabbit is not installed.

        Functionality requiring vowpalwabbit should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires vowpalwabbit.
        """

        PackageChecker._handle_import_error(caller_name, "vowpalwabbit")

    @staticmethod
    def pandas(caller_name: str) -> None:
        """Raise ImportError with detailed error message if pandas is not installed.

        Functionality requiring pandas should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires pandas.
        """
        PackageChecker._handle_import_error(caller_name, "pandas")

    @staticmethod
    def numpy(caller_name: str) -> None:
        """Raise ImportError with detailed error message if numpy is not installed.

        Functionality requiring numpy should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires numpy.
        """
        PackageChecker._handle_import_error(caller_name, "numpy")

    @staticmethod
    def scipy(caller_name: str) -> None:
        """Raise ImportError with detailed error message if scipy is not installed.

        Functionality requiring scipy should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires scipy.
        """
        PackageChecker._handle_import_error(caller_name, "scipy")

    @staticmethod
    def sklearn(caller_name: str) -> None:
        """Raise ImportError with detailed error message if sklearn is not installed.

        Functionality requiring sklearn should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires sklearn.
        """
        PackageChecker._handle_import_error(caller_name, "sklearn", "scikit-learn")

    @staticmethod
    def torch(caller_name: str) -> None:
        """Raise ImportError with detailed error message if torch is not installed.

        Functionality requiring torch should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires torch.
        """
        PackageChecker._handle_import_error(caller_name, "torch")

    def _handle_import_error(caller_name:str, module_name:str, pkg_name:str = None):
        """
        Remarks:
            This pattern borrows heavily from sklearn. As of 6/20/2020 sklearn code could be found
            at https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/__init__.py
        """
        try:
            importlib.import_module(module_name)
        except ImportError:
            pkg_name = pkg_name or module_name
            coba_exit(f"ERROR: {caller_name} requires the {pkg_name} package. You can install this package via `pip install {pkg_name}`.")

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


def sample_actions(
    actions: Sequence[Any],
    probabilities: Sequence[float],
    rng: Optional[CobaRandom] = None,
) -> Tuple[Any, float]:
    """
    Sample the actions weighted by their probabilities.
    """
    choice_function = rng.choice if rng else choice
    index = choice_function(range(len(probabilities)), probabilities)

    return actions[index], probabilities[index]
