"""Simple one-off utility methods with no clear home."""

import sys
import os

from io import UnsupportedOperation
from contextlib import contextmanager
from typing import IO
from coba.config import CobaException

@contextmanager
def redirect_stderr(to: IO[str]):
    """Redirect stdout for both C and Python.

    Remarks:
        This code comes from https://stackoverflow.com/a/17954769/1066291. Because this modifies
        global pointers this code is not "thread-safe". This limitation is also true of the built-in
        Python modules such as `contextlib.redirect_stdout` and `contextlib.redirect_stderr`. See
        https://docs.python.org/3/library/contextlib.html#contextlib.redirect_stdout for more info.
    """
    try:
        #we assume that this fd is the same
        #one that is used by our C library
        stderr_fd = sys.stderr.fileno()

        def _redirect_stderr(redirect_stderr_fd):
            
            #first we flush Python's stderr. It should be noted that this
            #doesn't close the file descriptor (i.e., sys.stderr.fileno())
            #or Python's wrapper around the stderr_fd.
            sys.stderr.flush()
        
            # next we change the stderr_fd to point to the
            # file contained in the redirect_stderr_fd.
            # If C has anything buffered for stderr it
            # will now go to the new fd. There do appear
            # to be ways to flush C buffers from Python 
            # but I'm not sure it is worth it given the
            # amount of complexity it adds to the code.
            # This change also means that sys.stderr now
            # points to a new file since sys.stderr points
            # to whatever file is at stderr_fd
            os.dup2(redirect_stderr_fd, stderr_fd)

        # when we dup there are now two fd's
        # pointing to the same file. Closing
        # one of these doesn't close the other.
        # therefore it is on us to close the
        # duplicate fd we make here before ending.
        old_stderr_fd = os.dup(stderr_fd)
        new_stderr_fd = to.fileno()

        try:
            _redirect_stderr(new_stderr_fd)
            yield # allow code to be run with the redirected stderr
        finally:
            _redirect_stderr(old_stderr_fd) 
            os.close(old_stderr_fd)
    except UnsupportedOperation:
        #if for some reason we weren't able to redirect
        #then simply move on. No reason to stop working.
        yield

class PackageChecker:

    @staticmethod
    def matplotlib(caller_name: str, silent: bool = False) -> None:
        """Raise ImportError with detailed error message if matplotlib is not installed.

        Functionality requiring matplotlib should call this helper and then lazily import.

        Args:    
            caller_name: The name of the caller that requires matplotlib.

        Remarks:
            This pattern borrows heavily from sklearn. As of 6/20/2020 sklearn code could be found
            at https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/__init__.py
        """
        try:
            import matplotlib # type: ignore
        except ImportError as e:
            _handle_import_error(e, caller_name, "matplotlib", silent)

    @staticmethod
    def vowpalwabbit(caller_name: str, silent: bool = False) -> None:
        """Raise ImportError with detailed error message if vowpalwabbit is not installed.

        Functionality requiring vowpalwabbit should call this helper and then lazily import.

        Args:    
            caller_name: The name of the caller that requires matplotlib.

        Remarks:
            This pattern was inspired by sklearn (see `PackageChecker.matplotlib` for more information).
        """

        try:            
            import vowpalwabbit # type: ignore
        except ImportError as e:
            pass

    @staticmethod
    def pandas(caller_name: str, silent: bool = False) -> None:
        """Raise ImportError with detailed error message if pandas is not installed.

        Functionality requiring pandas should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires pandas.

        Remarks:
            This pattern was inspired by sklearn (see `PackageChecker.matplotlib` for more information).
        """
        try:
            import pandas # type: ignore
        except ImportError as e:
            _handle_import_error(e, caller_name, "pandas", silent)

    @staticmethod
    def numpy(caller_name: str, silent: bool = False) -> None:
        """Raise ImportError with detailed error message if numpy is not installed.

        Functionality requiring numpy should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires numpy.

        Remarks:
            This pattern was inspired by sklearn (see `PackageChecker.matplotlib` for more information).
        """
        try:
            import numpy # type: ignore
        except ImportError as e:
            _handle_import_error(e, caller_name, "numpy", silent)

    @staticmethod
    def sklearn(caller_name: str, silent: bool = False) -> None:
        """Raise ImportError with detailed error message if numpy is not installed.

        Functionality requiring numpy should call this helper and then lazily import.

        Args:
            caller_name: The name of the caller that requires numpy.

        Remarks:
            This pattern was inspired by sklearn (see `PackageChecker.matplotlib` for more information).
        """
        try:
            import sklearn # type: ignore
        except ImportError as e:
            _handle_import_error(e, caller_name, "scikit-learn", silent)

def _handle_import_error(e: Exception, caller_name:str, pkg_name:str, silent:bool):
    if not silent:
        print(
            f"ERROR: {caller_name} requires the {pkg_name} package. You can "
            f"install this package via `pip install {pkg_name}`."
        )

    try:
        exit() #this is my attempt at handling CPython (where this works) vs IPython (e.g., Jupyter Notebooks)
    except NameError:
        raise CobaException() #CobaException has been specially made so that it outputs now TB in Jupyter Notebook 

class HashableDict(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._hash = hash(tuple(self.items()))

    def __hash__(self) -> int:

        assert self._hash == hash(tuple(self.items()))

        return self._hash