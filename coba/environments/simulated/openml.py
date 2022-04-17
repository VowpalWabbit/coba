import time
import json

from typing import Tuple, Sequence, Any, Iterable, Dict, MutableSequence, MutableMapping, Union, Optional
from coba.backports import Literal

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

    def __init__(self, data_id:int, label_type:Literal["C","R"] = None, cat_as_str:bool=False):
        """Instantiate an OpenmlSource.
        
        Args:
            data_id: The data id uniquely identifying the data source on openml (i.e., openml.org/d/{id})
            label_type: Indicates if a regression or classification label should be used on the dataset.
            cat_as_str: Indicates if categorical features should be encoded as a string rather than one hot encoded. 
        """
        self._data_id    = data_id
        self._label_type = label_type
        self._cat_as_str = cat_as_str
        self._cache_keys = {
            'descr': f"openml_{data_id:0>6}_descr",
            'feats': f"openml_{data_id:0>6}_feats",
            'csv'  : f"openml_{data_id:0>6}_csv",
            'arff' : f"openml_{data_id:0>6}_arff",
            'tasks': f"openml_{data_id:0>6}_tasks",
        }

    @property
    def params(self) -> Dict[str,Any]:
        """Parameters describing the openml source."""
        return  { "openml": self._data_id, "cat_as_str": self._cat_as_str }

    def read(self) -> Iterable[Tuple[Any, Any]]:
        """Read and parse the openml source."""
        try:

            dataset_description  = self._get_dataset_description(self._data_id)

            if dataset_description['status'] == 'deactivated':
                raise CobaException(f"Openml {self._data_id} has been deactivated. This is often due to flags on the data.")

            is_ignore = lambda r: (
                r['is_ignore'        ] == 'true' or
                r['is_row_identifier'] == 'true' or
                r['data_type'        ] not in ['numeric', 'nominal']
            )

            ignore = [ self._clean_name(f['name']) for f in self._get_feature_descriptions(self._data_id) if is_ignore(f)]
            
            if self._label_type is not None:
                target = self._clean_name(self._get_target_for_problem_type(self._get_task_descriptions(self._data_id)))
            else:
                target = dataset_description["default_target_attribute"]

            if target in ignore: ignore.pop(ignore.index(target))

            def row_has_missing_values(row):
                row_values = row._values.values() if isinstance(row,SparseWithMeta) else row._values
                return "?" in row_values or "" in row_values

            source    = ListSource(self._get_dataset_lines(dataset_description["file_id"], None))
            reader    = ArffReader(cat_as_str=self._cat_as_str)
            drop      = Drop(drop_cols=ignore, drop_row=row_has_missing_values)
            structure = Structure([None, target])

            return Pipes.join(source, reader, drop, structure).read()

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

    def _clean_name(self, name: str) -> str:
        return name.strip().strip('\'"').replace('\\','')

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

        # we only allow three paralellel request, another attempt at being more "considerate".
        if srcsema:srcsema.acquire() 

        # An attempt to be considerate of how often we hit their REST api. 
        # They don't publish any rate-limiting guidelines so this is just a guess.
        
        if srcsema: time.sleep(2*random())

        try:
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
        finally:
            if srcsema: srcsema.release()

    def _get_dataset_description(self, data_id:int) -> Dict[str,Any]:

        description_txt = " ".join(self._get_data(f'https://www.openml.org/api/v1/json/data/{data_id}', self._cache_keys['descr']))
        description_obj = json.loads(description_txt)["data_set_description"]

        return description_obj

    def _get_feature_descriptions(self, data_id:int) -> Sequence[Dict[str,Any]]:

        types_txt = " ".join(self._get_data(f'https://www.openml.org/api/v1/json/data/features/{data_id}', self._cache_keys['feats']))
        types_obj = json.loads(types_txt)["data_features"]["feature"]

        return types_obj

    def _get_task_descriptions(self, data_id) -> Sequence[Dict[str,Any]]:
        
        try:
            text  = " ".join(self._get_data(f'https://www.openml.org/api/v1/json/task/list/data_id/{data_id}', self._cache_keys['tasks']))
            return json.loads(text).get("tasks",{}).get("task",[])
        except CobaException:
            return []

    def _get_dataset_lines(self, file_id:str, md5_checksum:str) -> Iterable[str]:

            arff_url = f"https://www.openml.org/data/v1/download/{file_id}"

            csv_key  = self._cache_keys['csv']
            arff_key = self._cache_keys['arff']

            if csv_key in CobaContext.cacher: CobaContext.cacher.rmv(csv_key)
            return self._get_data(arff_url, arff_key, md5_checksum)

    def _get_target_for_problem_type(self, tasks: Sequence[Dict[str,Any]]):

        task_type_id = 1 if self._label_type == "C" else 2

        for task in tasks:
            if task["task_type_id"] == task_type_id:
                for input in task['input']:
                    if input['name'] == 'target_feature':
                        return input['value'] # just take the first one

        raise CobaException(f"Openml {self._data_id} does not appear to be a {self._label_type} dataset")

    def _clear_cache(self) -> None:
        for key in self._cache_keys.values():
            CobaContext.cacher.release(key) #to make sure we don't get stuck in a race condition
            CobaContext.cacher.rmv(key)

class OpenmlSimulation(SupervisedSimulation):
    """A supervised simulation created from an openml dataset.

    Download a dataset from openml.org and create a SupervisedSimulation.
    """

    def __init__(self, 
        data_id: int, 
        take: Union[int, Tuple[Optional[int], Optional[int]]] = None, 
        label_type: Literal["C","R"] = None, 
        cat_as_str: bool = False) -> None:
        """Instantiate an OpenmlSimulation.

        Args:
            data_id: The id for an openml dataset. This can be found, for example, in the url https://openml.org/d/<id>.
            take: The number of interactions we'd like the simulation to have (these will be selected at random).
            label_type: Whether classification or regression openml tasks should be used to create the simulation.
            cat_as_str: True if categorical features should be left as strings, false if they should be one hot encoded.
        """
        super().__init__(OpenmlSource(data_id, label_type, cat_as_str), None, label_type, take)

    @property
    def params(self) -> Dict[str, Any]:
        return {**super().params, "type": "OpenmlSimulation" }

    def __str__(self) -> str:
        return f"Openml(id={self.params['openml']}, cat_as_str={self.params['cat_as_str']})"
