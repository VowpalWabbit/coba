import warnings
import importlib

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
    def matplotlib(caller_name: str) -> None:
        """Raise ImportError with detailed error message if matplotlib is not installed.

        Functionality requiring matplotlib should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires matplotlib.

        Remarks:
            This pattern borrows heavily from sklearn. As of 6/20/2020 sklearn code could be found
            at https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/__init__.py
        """
        try:
            importlib.import_module('matplotlib.pyplot')
        except ImportError:
            PackageChecker._handle_import_error(caller_name, "matplotlib")

    @staticmethod
    def vowpalwabbit(caller_name: str) -> None:
        """Raise ImportError with detailed error message if vowpalwabbit is not installed.

        Functionality requiring vowpalwabbit should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires vowpalwabbit.

        Remarks:
            This pattern was inspired by sklearn (see `PackageChecker.matplotlib` for more information).
        """

        try:
            importlib.import_module('vowpalwabbit')
        except ImportError as e:
            PackageChecker._handle_import_error(caller_name, "vowpalwabbit")

    @staticmethod
    def pandas(caller_name: str) -> None:
        """Raise ImportError with detailed error message if pandas is not installed.

        Functionality requiring pandas should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires pandas.

        Remarks:
            This pattern was inspired by sklearn (see `PackageChecker.matplotlib` for more information).
        """
        try:
            importlib.import_module('pandas')
        except ImportError:
            PackageChecker._handle_import_error(caller_name, "pandas")

    @staticmethod
    def numpy(caller_name: str) -> None:
        """Raise ImportError with detailed error message if numpy is not installed.

        Functionality requiring numpy should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires numpy.

        Remarks:
            This pattern was inspired by sklearn (see `PackageChecker.matplotlib` for more information).
        """
        try:
            importlib.import_module('numpy')
        except ImportError:
            PackageChecker._handle_import_error(caller_name, "numpy")

    @staticmethod
    def scipy(caller_name: str) -> None:
        """Raise ImportError with detailed error message if scipy is not installed.

        Functionality requiring scipy should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires scipy.

        Remarks:
            This pattern was inspired by sklearn (see `PackageChecker.scipy` for more information).
        """
        try:
            importlib.import_module('scipy')
        except ImportError:
            PackageChecker._handle_import_error(caller_name, "scipy")

    @staticmethod
    def sklearn(caller_name: str) -> None:
        """Raise ImportError with detailed error message if sklearn is not installed.

        Functionality requiring sklearn should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires sklearn.

        Remarks:
            This pattern was inspired by sklearn (see `PackageChecker.matplotlib` for more information).
        """
        try:
            importlib.import_module('sklearn')
        except ImportError:
            PackageChecker._handle_import_error(caller_name, "scikit-learn")

    def _handle_import_error(caller_name:str, pkg_name:str):
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
    action_space,#: list[str],
    probabilities,#: list[float],
    random_seed = None,#: str | None = None,
):# -> tuple[str, float]:
    """
    Sample an action in the action space weighted by the probabilities.
    A random seed can be set to achieve repeatable results. Every time the seed is set the sampling result is
    deterministic. Calling sample_actions(actions, probabilities, "my_random_seed") ten times will result in equal
    results for each call.
    """
    import numpy as np
    from hashlib import sha256
    # Scale to a sum of 1
    probabilities_normalized = np.array(probabilities) / np.sum(probabilities)

    seed = np.frombuffer(sha256(random_seed.encode()).digest(), dtype='uint32') if random_seed is not None else None
    rng = np.random.default_rng(seed=seed)

    action = rng.choice(a=action_space, p=probabilities_normalized)
    probability = probabilities_normalized[action_space.index(action)]

    return action, probability