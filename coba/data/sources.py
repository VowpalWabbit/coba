"""The data.sources module contains core classes for sources used in data pipelines.

TODO: Add docstrings for all Sources
TODO: Add unit tests for all Sources
"""

from coba.data.encoders import StringEncoder
import itertools
import requests
import json

from abc import ABC, abstractmethod
from hashlib import md5
from typing import Generic, Iterable, Sequence, TypeVar, Tuple, Any

from coba.execution import ExecutionContext

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
    def __init__(self, source: _T_out): #type:ignore
        self._source = source

    def read(self) -> _T_out:
        return self._source

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

class OpenmlSource(Source[Tuple[Sequence[Sequence[Any]], Sequence[Any]]]):

    def __init__(self, data_id:int, md5_checksum:str = None):
        self._data_id      = data_id
        self._md5_checksum = md5_checksum

    def read(self) -> Tuple[Sequence[Sequence[Any]], Sequence[Any]]:
        
        from coba.data.pipes import Pipe
        from coba.data.encoders import NumericEncoder, OneHotEncoder
        from coba.data.filters import CsvReader, LabeledCsvCleaner

        data_id        = self._data_id
        md5_checksum   = self._md5_checksum
        openml_api_key = ExecutionContext.Config.openml_api_key

        data_description_url = f'https://www.openml.org/api/v1/json/data/{data_id}'
        type_description_url = f'https://www.openml.org/api/v1/json/data/features/{data_id}'

        if openml_api_key is not None:
            data_description_url += f'?api_key={openml_api_key}'
            type_description_url += f'?api_key={openml_api_key}'

        descr = json.loads(''.join(HttpSource(data_description_url, '.json', None, 'descr').read()))["data_set_description"]

        if descr['status'] == 'deactivated':
            raise Exception(f"Openml {data_id} has been deactivated. This is often due to flags on the data.")

        types = json.loads(''.join(HttpSource(type_description_url, '.json', None, 'types').read()))["data_features"]["feature"]

        headers  = []
        encoders = []
        ignored  = []
        target   = ""

        for tipe in types:

            headers.append(tipe['name'])
            ignored.append(tipe['is_ignore'] == 'true' or tipe['is_row_identifier'] == 'true')

            if tipe['is_target'] == 'true':
                target = tipe['name']

            if tipe['data_type'] == 'numeric':
                encoders.append(NumericEncoder())  
            elif tipe['data_type'] == 'nominal' and tipe['is_target'] == 'false':
                encoders.append(OneHotEncoder(singular_if_binary=True))
            elif tipe['data_type'] == 'nominal' and tipe['is_target'] == 'true':
                encoders.append(OneHotEncoder())
            else:
                encoders.append(StringEncoder())

        csv_url = f"http://www.openml.org/data/v1/get_csv/{descr['file_id']}"

        source  = HttpSource(csv_url, ".csv", md5_checksum, f"openml {data_id}")
        reader  = CsvReader()
        cleaner = LabeledCsvCleaner(target, headers, encoders, ignored, True)

        feature_rows, label_rows = Pipe.join(source, [reader, cleaner]).read()

        return list(feature_rows), list(label_rows)