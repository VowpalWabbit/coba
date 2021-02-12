"""The data.sources module contains core classes for sources used in data pipelines.

TODO: Add docstrings for all Sources
TODO: Add unit tests for all Sources
"""

import requests

from abc import ABC, abstractmethod
from hashlib import md5
from typing import Generic, Iterable, TypeVar, Any

from coba.tools import ExecutionContext

_T_out = TypeVar("_T_out", bound=Any, covariant=True)

class Source(ABC, Generic[_T_out]):
    @abstractmethod
    def read(self) -> _T_out:
        ...

class DiskSource(Source[Iterable[str]]):
    def __init__(self, filename:str):
        self.filename = filename

    def read(self) -> Iterable[str]:
        with open(self.filename, "r+") as f:
            for line in f:
                yield line

class MemorySource(Source[_T_out]):
    def __init__(self, item: _T_out): #type:ignore
        self._item = item

    def read(self) -> _T_out:
        return self._item

class QueueSource(Source[Iterable[Any]]):
    def __init__(self, source: Any, poison=None) -> None:
        self._queue  = source
        self._poison = poison

    def read(self) -> Iterable[Any]:
        while True:
            item = self._queue.get()

            if item == self._poison:
                return

            yield item

class HttpSource(Source[Iterable[str]]):
    def __init__(self, url: str, file_extension: str = None, checksum: str = None, desc: str = "") -> None:
        self._url       = url
        self._checksum  = checksum
        self._desc      = desc
        self._cachename = f"{md5(self._url.encode('utf-8')).hexdigest()}{file_extension}"

    def read(self) -> Iterable[str]:
        bites = self._get_bytes()

        if self._checksum is not None and md5(bites).hexdigest() != self._checksum:
            message = (
                f"The dataset at {self._url} did not match the expected checksum. This could be the result of "
                "network errors or the file becoming corrupted. Please consider downloading the file again "
                "and if the error persists you may want to manually download and reference the file.")
            raise Exception(message) from None

        if self._cachename not in ExecutionContext.FileCache: ExecutionContext.FileCache.put(self._cachename, bites)

        return bites.decode('utf-8').splitlines()
    
    def _get_bytes(self) -> bytes:
        if self._cachename in ExecutionContext.FileCache:
            with ExecutionContext.Logger.log(f'loading {self._desc} from cache... '.replace('  ', ' ')):
                return ExecutionContext.FileCache.get(self._cachename)
        else:
            with ExecutionContext.Logger.log(f'loading {self._desc} from http... '):
                response = requests.get(self._url)

                if response.status_code == 412 and 'openml' in self._url:
                    if 'please provide api key' in response.text:
                        message = (
                            "An API Key is needed to access openml's rest API. A key can be obtained by creating an "
                            "openml account at openml.org. Once a key has been obtained it should be placed within "
                            "~/.coba as { \"openml_api_key\" : \"<your key here>\", }.")
                        raise Exception(message) from None

                    if 'authentication failed' in response.text:
                        message = (
                            "The API Key you provided no longer seems to be valid. You may need to create a new one"
                            "longing into your openml account and regenerating a key. After regenerating the new key "
                            "should be placed in ~/.coba as { \"openml_api_key\" : \"<your key here>\", }.")
                        raise Exception(message) from None

                return response.content