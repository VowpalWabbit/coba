import time
import json

from typing import Tuple, Sequence, Any, Iterable, Dict, MutableSequence, MutableMapping, Union, overload

from coba.random import random
from coba.pipes import Pipes, Source, HttpSource, Drop, ArffReader, ListSource, Structure
from coba.contexts import CobaContext, CobaContext
from coba.exceptions import CobaException

from coba.environments.simulated.supervised import SupervisedSimulation
from coba.pipes.readers import SparseWithMeta

class OpenmlSource(Source[Iterable[Tuple[Union[MutableSequence, MutableMapping],Any]]]):
    """Load a source (local if cached) from openml.

    This is primarily used by OpenmlSimulation to create Environments for Experiments.
    """

    @overload
    def __init__(self, *, data_id:int, cat_as_str:bool=False, drop_missing:bool=False):
        """Instantiate an OpenmlSource.

        Args:
            data_id: The data id uniquely identifying the dataset on openml (i.e., openml.org/d/{id})
            cat_as_str: Categorical features should be encoded as a string rather than one hot encoded.
            drop_missing: Drop data rows with missing values.
        """
        ...

    @overload
    def __init__(self, *, task_id:int, cat_as_str:bool=False, drop_missing:bool=False):
        """Instantiate an OpenmlSource.

        Args:
            task_id: The openml task id which identifies the dataset to use from openml along with its label
            cat_as_str: Categorical features should be encoded as a string rather than one hot encoded.
            drop_missing: Drop data rows with missing values.
        """
        ...

    def __init__(self, **kwargs):
        """Instantiate an OpenmlSource."""

        self._data_id      = kwargs.get('data_id',None)
        self._task_id      = kwargs.get('task_id',None)
        self._target       = None
        self._cat_as_str   = kwargs.get('cat_as_str',False)
        self._drop_missing = kwargs.get('drop_missing',True)

    @property
    def params(self) -> Dict[str,Any]:
        """Parameters describing the openml source."""
        return  { "openml_data": self._data_id, "openml_task": self._task_id, "openml_target": self._target, "cat_as_str": self._cat_as_str, "drop_missing": self._drop_missing }

    def read(self) -> Iterable[Tuple[Any, Any]]:
        """Read and parse the openml source."""

        try:

            # we only allow three paralellel request, an attempt at being "considerate" to openml
            srcsema = CobaContext.store.get("srcsema")
            acquired = False
            
            if srcsema and not self._all_cached():
                srcsema.acquire()
                if self._all_cached(): #pragma: no cover
                    srcsema.release() #in-case another process cached everything needed while we were waiting
                else:
                    acquired = True

            if self._data_id:
                data_descr   = self._get_data_descr(self._data_id)
                self._target = self._clean_name(data_descr.get("default_target_attribute",None))
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

            is_ignore = lambda feat_descr: (
                feat_descr['is_ignore'        ] == 'true' or
                feat_descr['is_row_identifier'] == 'true' or
                feat_descr['data_type'        ] not in ['numeric', 'nominal']
            )

            ignore = [ self._clean_name(f['name']) for f in self._get_feat_descr(self._data_id) if is_ignore(f)]

            if self._target in ignore: ignore.pop(ignore.index(self._target))

            def drop_row(row):
                row_values = row._values.values() if isinstance(row,SparseWithMeta) else row._values
                has_missing = "?" in row_values or "" in row_values
                return self._drop_missing and has_missing

            source    = ListSource(self._get_arff_lines(data_descr["file_id"], None))
            reader    = ArffReader(cat_as_str=self._cat_as_str)
            drop      = Drop(drop_cols=ignore, drop_row=drop_row)
            structure = Structure([None, self._target])

            for features,label in Pipes.join(source, reader, drop, structure).read():
                #ensure that SupervisedSimulation will interpret the label as a class
                if task_type == 1 and isinstance(label,(int,float)): 
                    label = str(int(label) if float(label).is_integer() else label)
                
                yield features, label

        except KeyboardInterrupt:
            #we don't want to clear the cache in the case of a KeyboardInterrupt
            raise

        except CobaException:
            #we don't want to clear the cache if it is an error we know about (the original raise should clear if needed)
            raise

        except Exception as e:
            #if something unexpected went wrong clear the cache just in case it was corrupted somehow
            self._clear_cache()
            raise
    
        finally:
            if acquired:
                srcsema.release()

    def _clean_name(self, name: str) -> str:
        return name.strip().strip('\'"').replace('\\','') if name else name

    def _get_data(self, url:str, key:str, checksum:str=None) -> Iterable[str]:

        # This can't be done in a streaming manner unless there is a persistent cacher.
        # Because we don't require cacher this means this may not be possible with streaming.
        # if checksum is not None and md5(bites).hexdigest() != checksum:
        #     #if the cache has become corrupted we need to clear it
        #     CobaContext.cacher.rmv(key)
        #     message = (
        #         f"The response from {url} did not match the given checksum {checksum}. This could be the result "
        #         "of network errors or the file becoming corrupted. Please consider downloading the file again. "
        #         "If the error persists you may want to manually download and reference the file.")
        #     raise CobaException(message) from None
        try:
            for b in CobaContext.cacher.get_put(key, lambda: self._http_request(url)):
                yield b.decode('utf-8')
        except Exception:
            self._clear_cache()
            raise

    def _http_request(self, url:str) -> Iterable[bytes]:
        api_key = CobaContext.api_keys['openml']
        srcsema = CobaContext.store.get("srcsema")

        # An attempt to be considerate of how often we hit their REST api.
        # They don't publish any rate-limiting guidelines so this is just a guess.
        # we check for srcsema because this indicates whether we are multiprocessing
        if srcsema: time.sleep(2*random())

        with HttpSource(url + (f'?api_key={api_key}' if api_key else '')).read() as response:

            if response.status_code == 412:
                if 'please provide api key' in response.text:
                    message = (
                        "Openml has requested an API Key to access openml's rest API. A key can be obtained by creating "
                        "an openml account at openml.org. Once a key has been obtained it should be placed within "
                        "~/.coba as { \"api_keys\" : { \"openml\" : \"<your key here>\", } }.")
                    raise CobaException(message)

                if 'authentication failed' in response.text:
                    message = (
                        "The API Key you provided no longer seems to be valid. You may need to create a new one by "
                        "logging into your openml account and regenerating a key. After regenerating the new key "
                        "should be placed in ~/.coba as { \"api_keys\" : { \"openml\" : \"<your key here>\", } }.")
                    raise CobaException(message)

            if response.status_code == 404:
                raise CobaException("We're sorry but we were unable to find the requested dataset on openml.")

            if response.status_code != 200:
                raise CobaException(f"An error was returned by openml: {response.text}")

            # NOTE: These two checks need to be gated with a status code failure
            # NOTE: otherwise this will cause the data to be downloaded all at once
            # NOTE: unfortunately I don't know the appropriate status code so commenting out for now
            # if "Usually due to high server load" in response.text:
            #     message = (
            #         "Openml has experienced an error that they believe is the result of high server loads. "
            #         "Openml recommends that you try again in a few seconds. Additionally, if not already "
            #         "done, consider setting up a DiskCache in a coba config file to reduce the number of "
            #         "openml calls in the future.")
            #     raise CobaException(message) from None

            # if '' == response.text:
            #     raise CobaException("Openml experienced an unexpected error. Please try requesting the data again.") from None

            for b in response.iter_lines(decode_unicode=False):
                yield b

    def _get_data_descr(self, data_id:int) -> Dict[str,Any]:

        descr_txt = " ".join(self._get_data(f'https://www.openml.org/api/v1/json/data/{data_id}', self._cache_keys['data']))
        descr_obj = json.loads(descr_txt)["data_set_description"]

        return descr_obj

    def _get_feat_descr(self, data_id:int) -> Sequence[Dict[str,Any]]:

        descr_txt = " ".join(self._get_data(f'https://www.openml.org/api/v1/json/data/features/{data_id}', self._cache_keys['feat']))
        descr_obj = json.loads(descr_txt)["data_features"]["feature"]

        return descr_obj

    def _get_task_descr(self, task_id) -> Dict[str,Any]:

        descr_txt = " ".join(self._get_data(f'https://www.openml.org/api/v1/json/task/{task_id}', self._cache_keys['task']))
        descr_obj = json.loads(descr_txt)['task']

        task_type   = int(descr_obj.get('task_type_id',0))
        task_source = ([i for i in descr_obj.get("input",[]) if i.get('name',None) == 'source_data'] + [{}])[0]
        data_id     = int(task_source.get("data_set",{}).get("data_set_id",0)) or None
        target      = task_source.get("data_set",{}).get("target_feature","") or None

        return { 'id': task_id, 'type': task_type, 'data': data_id, 'target': target}

    def _get_arff_lines(self, file_id:str, md5_checksum:str) -> Iterable[str]:

            arff_url = f"https://www.openml.org/data/v1/download/{file_id}"
            arff_key = self._cache_keys['arff']

            return self._get_data(arff_url, arff_key, md5_checksum)

    def _clear_cache(self) -> None:
        for key in self._cache_keys.values():
            CobaContext.cacher.release(key) #to make sure we don't get stuck in a race condition
            CobaContext.cacher.rmv(key)

    def _all_cached(self) -> bool:

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
    def __init__(self, data_id: int, cat_as_str: bool = False, keep_missing: bool = False, take: int = None):
        """Instantiate an OpenmlSimulation.

        Args:
            data_id: The data id uniquely identifying the dataset on openml (i.e., openml.org/d/{id})
            cat_as_str: Indicates if categorical features should be encoded as a string rather than one hot encoded.
            drop_missing: Drop data rows with missing values.
            take: The number of interactions we'd like the simulation to have (these will be selected at random).
        """
        ...

    @overload
    def __init__(self, *, task_id: int, cat_as_str: bool = False, keep_missing: bool = False, take: int=None):
        """Instantiate an OpenmlSimulation.

        Args:
            task_id: The openml task id which identifies the dataset to use from openml along with its label
            cat_as_str: Indicates if categorical features should be encoded as a string rather than one hot encoded.
            drop_missing: Drop data rows with missing values.
            take: The number of interactions we'd like the simulation to have (these will be selected at random).
        """
        ...

    def __init__(self, *args, **kwargs) -> None:
        """Instantiate an OpenmlSimulation."""
        kwargs.update(zip(['data_id','cat_as_str','drop_missing','take'], args))
        super().__init__(OpenmlSource(**kwargs), None, None, kwargs.get('take',None))

    @property
    def params(self) -> Dict[str, Any]:
        return {**super().params, "type": "OpenmlSimulation" }

    def __str__(self) -> str:

        params = [
            f"data={self.params['openml_data']}" if self.params.get('openml_data') else f"task={self.params['openml_task']}",
            f"target={self.params['openml_target']}" if self.params.get('openml_target') else "",
            f"cat_as_str={self.params['cat_as_str']}",
            f"drop_missing={self.params['drop_missing']}"
        ]

        return f"Openml({str.join(', ', filter(None,params))})"
