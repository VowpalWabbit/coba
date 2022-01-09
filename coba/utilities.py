import warnings
import importlib

from itertools import count
from collections import defaultdict
from typing import Sequence, Any, Union

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
            importlib.import_module('matplotlib')
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

class HashableDict(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._hash = hash(tuple(self.items()))

    def __hash__(self) -> int:
        assert self._hash == hash(tuple(self.items()))
        return self._hash

class KeyDefaultDict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError( key )
        else:
            value = self.default_factory(key)
            self[key] = value
            return value

class HeaderList(list):
    def __init__(self, values: Sequence[Any], headers: Sequence[str]) -> None:
        self._headers = dict(zip(headers,count()))
        return super().__init__(values)

    def __getitem__(self, index: Union[str,int]) -> Any:
        return super().__getitem__(self._headers.get(index,index))

    def __setitem__(self, index: Union[str,int], value: Any) -> None:
        super().__setitem__(self._headers.get(index,index), value)

    def pop(self, index: Union[str,int]) -> Any:
        index = self._headers.pop(index,index)
        for k,v in self._headers.items():
            if v > index:
                self._headers[k] = v-1
        return super().pop(index)

class HeaderDict(dict):

    def __init__(self, values, headers: Sequence[str]) -> None:
        self._headers = dict(zip(headers,count()))
        return super().__init__(values)

    def __contains__(self, key: Union[str,int]) -> bool:
        return super().__contains__(self._headers.get(key,key))

    def __getitem__(self, key: Union[str,int]) -> Any:
        return super().__getitem__(self._headers.get(key,key))

    def __setitem__(self, index: Union[str,int], value: Any) -> None:
        super().__setitem__(self._headers.get(index,index), value)

    def get(self, *args) -> Any:
        index = self._headers.get(args[0],args[0])
        return super().get(*(index, *args[1:]))

    def pop(self, *args) -> Any:
        index = self._headers.pop(args[0],args[0])
        return super().pop(*(index, *args[1:]))
