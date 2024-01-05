import time
import json

from urllib import request
from operator import attrgetter
from typing import Tuple, Sequence, Any, Iterable, Dict, MutableSequence, MutableMapping, Union, overload

from coba.random import random
from coba.pipes import Pipes, HttpSource, ArffReader, DropRows, LabelRows
from coba.context import CobaContext
from coba.primitives import Sparse, Dense, Source
from coba.exceptions import CobaException

from coba.environments.supervised import SupervisedSimulation

class OpenmlSource(Source[Iterable[Tuple[Union[MutableSequence, MutableMapping],Any]]]):
    """Load a source from openml.org"""

    @overload
    def __init__(self, *, data_id:int, drop_missing:bool=True, target:str = None):
        """Instantiate an OpenmlSource.
        Args:
            data_id: The data id uniquely identifying the dataset on openml (i.e., openml.org/d/{id})
            drop_missing: Drop data rows with missing values.
            target: The column that should be marked as the prediction label in the source.
        """
        ...

    @overload
    def __init__(self, *, task_id:int, drop_missing:bool=True, target:str = None):
        """Instantiate an OpenmlSource.
        Args:
            task_id: The openml task id which identifies the dataset to use from openml along with its label
            drop_missing: Drop data rows with missing values.
            target: The column that should be marked as the prediction label in the source
        """
        ...

    def __init__(self, **kwargs):
        """Instantiate an OpenmlSource."""

        self._data_id        = kwargs.get('data_id',None)
        self._task_id        = kwargs.get('task_id',None)
        self._target         = kwargs.get("target", None)
        self._drop_missing   = kwargs.get('drop_missing',True)

    @property
    def params(self) -> Dict[str,Any]:
        """Parameters describing the openml source."""
        return  { "openml_data": self._data_id, "openml_task": self._task_id, "openml_target": self._target, "drop_missing": self._drop_missing }

    def read(self) -> Iterable[Union[Dense,Sparse]]:
        """Read and parse the openml source."""

        try:

            # we only allow three paralellel request, an attempt at being "considerate" to openml
            # openml_semaphore will be None if we aren't multiprocessing otherwise it'll have a sema
            openml_semaphore = CobaContext.store.get("openml_semaphore")
            semaphore_acquired = False

            if openml_semaphore and not self._source_already_cached():
                openml_semaphore.acquire()
                if self._source_already_cached(): #pragma: no cover
                    openml_semaphore.release() #in-case another process cached everything needed while we were waiting
                else:
                    semaphore_acquired = True

            if self._data_id:
                data_descr   = self._get_data_descr(self._data_id)
                self._target = self._target or self._clean_name(data_descr.get("default_target_attribute",None))
                task_type    = None

            if self._task_id:
                task_descr = self._get_task_descr(self._task_id)
                task_type  = task_descr['type']

                if task_descr['type'] not in [1,2]:
                    raise CobaException(f"Openml task {self._task_id} does not appear to be a regression or classification task.")

                if not task_descr['data']:
                    raise CobaException(f"Openml task {self._task_id} does not appear to have an associated data source.")

                self._data_id = task_descr['data']
                self._target  = self._clean_name(task_descr.get('target'))
                data_descr    = self._get_data_descr(self._data_id)

            if not self._target:
                raise CobaException(f"We were unable to find an appropriate target column for the given openml source.")

            if data_descr.get('status') == 'deactivated':
                raise CobaException(f"Openml {self._data_id} has been deactivated. This is often due to flags on the data.")

            is_ignore = lambda feat_descr:(
                feat_descr['is_ignore'        ] == 'true' or
                feat_descr['is_row_identifier'] == 'true' or
                feat_descr['data_type'        ] not in ['numeric', 'nominal']
            )

            ignore = [ self._clean_name(f['name']) for f in self._get_feat_descr(self._data_id) if is_ignore(f)]

            if self._target in ignore: ignore.pop(ignore.index(self._target))

            label_type = 'c' if task_type==1 else 'r' if task_type==2 else None
            drop_row = attrgetter('missing') if self._drop_missing else None

            lines = self._get_arff_lines(data_descr["file_id"], None)
            reader= ArffReader()
            drop  = DropRows(drop_cols=ignore, drop_row=drop_row)
            label = LabelRows(self._target, label_type)

            yield from Pipes.join(reader, drop, label).filter(lines)

        except KeyboardInterrupt:
            #we don't want to clear the cache in the case of a KeyboardInterrupt
            raise

        except CobaException:
            #we don't want to clear the cache if it is an error we know about (the original raise should clear if needed)
            raise

        except Exception:
            #if something unexpected went wrong clear the cache just in case it was corrupted somehow
            self._clear_cache()
            raise

        finally:
            if semaphore_acquired:
                openml_semaphore.release()

    def _clean_name(self, name: str) -> str:
        return name.strip().strip('\'"').replace('\\','') if name else name

    def _get_data(self, url:str, key:str, checksum:str=None) -> Iterable[str]:
        try:
            with CobaContext.cacher.get_set(key, lambda: self._http_request(url)) as out:
                yield from out
        except Exception:
            self._clear_cache()
            raise

    def _http_request(self, url: str, tries: int = 0) -> Iterable[str]:
        api_key   = CobaContext.api_keys.get('openml')
        semaphore = CobaContext.store.get("openml_semaphore")

        # In an attempt to be considerate we stagger/limit our hits of the REST API.
        # Openml doesn't publish any rate-limiting guidelines, so this is just a guess.
        # if semaphore is not None it indictes that we are in a CobaMultiprocessor.
        if semaphore: time.sleep(2*random())

        try:
            KB = 1024
            MB = 1024*KB
            if api_key: url = f"{url}?api_key={api_key}"
            yield from HttpSource(url, timeout=20, chunk_size=10*MB).read()

        except TimeoutError:
            if tries >= 3: raise
            yield from self._http_request(url, tries+1)
            return

        except request.HTTPError as e:
            status, content = e.code, e.fp.read()

            if status == 412 and 'please provide api key' in content.lower():
                raise CobaException(
                    "Openml has requested an API key to access openml's rest API. A key can be obtained by creating "
                    "an openml account at openml.org. Once a key has been obtained it should be placed within "
                    "~/.coba as { \"api_keys\" : { \"openml\" : \"<your key here>\", } }.")

            if status == 412 and 'authentication failed' in content.lower():
                raise CobaException(
                    "The API key you provided no longer seems to be valid. You may need to create a new one by "
                    "logging into your openml account and regenerating a key. After regenerating the new key "
                    "should be placed in ~/.coba as { \"api_keys\" : { \"openml\" : \"<your key here>\", } }.")

            if status == 404:
                raise CobaException("We're sorry but we were unable to find the requested dataset on openml.")

            raise CobaException(f"An error was returned by openml: {content}")


    def _get_data_descr(self, data_id:int) -> Dict[str,Any]:
        descr_txt = " ".join(self._get_data(f'https://openml.org/api/v1/json/data/{data_id}', self._cache_keys['data']))
        descr_obj = json.loads(descr_txt)["data_set_description"]
        return descr_obj

    def _get_feat_descr(self, data_id:int) -> Sequence[Dict[str,Any]]:
        descr_txt = " ".join(self._get_data(f'https://openml.org/api/v1/json/data/features/{data_id}', self._cache_keys['feat']))
        descr_obj = json.loads(descr_txt)["data_features"]["feature"]
        return descr_obj

    def _get_task_descr(self, task_id) -> Dict[str,Any]:
        descr_txt = " ".join(self._get_data(f'https://openml.org/api/v1/json/task/{task_id}', self._cache_keys['task']))
        descr_obj = json.loads(descr_txt)['task']

        task_type   = int(descr_obj.get('task_type_id',0))
        task_source = ([i for i in descr_obj.get("input",[]) if i.get('name',None) == 'source_data'] + [{}])[0]
        data_id     = int(task_source.get("data_set",{}).get("data_set_id",0)) or None
        target      = task_source.get("data_set",{}).get("target_feature","") or None

        return { 'id': task_id, 'type': task_type, 'data': data_id, 'target': target}

    def _get_arff_lines(self, file_id:str, md5_checksum:str) -> Iterable[str]:
            arff_url = f"https://openml.org/data/v1/download/{file_id}"
            arff_key = self._cache_keys['arff']
            return self._get_data(arff_url, arff_key, md5_checksum)

    def _clear_cache(self) -> None:
        for key in self._cache_keys.values():
            CobaContext.cacher.rmv(key)

    def _source_already_cached(self) -> bool:
        old_data_id = self._data_id
        all_cached  = False

        if self._task_id and self._cache_keys['task'] in CobaContext.cacher:
            task_descr = self._get_task_descr(self._task_id)

            if not task_descr['data']: return True #this will fall into an exception so no more caching is needed

            self._data_id = task_descr['data']

        all_cached = bool(self._data_id and all(self._cache_keys[k] in CobaContext.cacher for k in ['data','feat','arff']))

        self._data_id = old_data_id
        return all_cached

    @property
    def _cache_keys(self):
        return {
            'task': f"openml_{self._task_id or 0:0>6}_task",
            'data': f"openml_{self._data_id or 0:0>6}_data",
            'feat': f"openml_{self._data_id or 0:0>6}_feat",
            'arff': f"openml_{self._data_id or 0:0>6}_arff",
        }

class OpenmlSimulation(SupervisedSimulation):
    """A supervised simulation created from an openml dataset.
    Download a dataset from openml.org and create a SupervisedSimulation.
    """

    @overload
    def __init__(self, data_id: int, drop_missing: bool = True, take: int = None, *, target:str = None, ):
        """Instantiate an OpenmlSimulation.
        Args:
            data_id: The data id uniquely identifying the dataset on openml (i.e., openml.org/d/{id})
            drop_missing: Drop data rows with missing values.
            take: The number of interactions we'd like the simulation to have (these will be selected at random).
            target: The column that should be marked as the prediction label in the source.
        """
        ...

    @overload
    def __init__(self, *, task_id: int, drop_missing: bool = True, take: int=None, target:str = None):
        """Instantiate an OpenmlSimulation.
        Args:
            task_id: The openml task id which identifies the dataset to use from openml along with its label
            drop_missing: Drop data rows with missing values.
            take: The number of interactions we'd like the simulation to have (these will be selected at random).
            target: The column that should be marked as the prediction label in the source.
        """
        ...

    def __init__(self, *args, **kwargs) -> None:
        """Instantiate an OpenmlSimulation."""
        kwargs.update(zip(['data_id','drop_missing','take'], args))
        super().__init__(OpenmlSource(**kwargs), None, kwargs.get('label_type',None), kwargs.get('take',None))

    @property
    def params(self) -> Dict[str, Any]:
        return {**super().params, "env_type": "OpenmlSimulation" }

    def __str__(self) -> str:

        params = [
            f"data={self.params['openml_data']}" if self.params.get('openml_data') else f"task={self.params['openml_task']}",
            f"target={self.params['openml_target']}" if self.params.get('openml_target') else "",
        ]

        return f"Openml({str.join(', ', filter(None,params))})"
