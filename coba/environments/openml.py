import json

from hashlib import md5
from numbers import Number
from collections import defaultdict
from typing_extensions import Literal
from typing import Tuple, Sequence, Any, Iterable, Dict, Union

from coba.pipes import Pipe, Source, HttpIO, Default, Drop, Encode, _T_Data, Structure, ArffReader, CsvReader, Take
from coba.config import CobaConfig
from coba.exceptions import CobaException
from coba.encodings import NumericEncoder, OneHotEncoder, StringEncoder

from coba.environments.primitives import SimulatedEnvironment, SimulatedInteraction
from coba.environments.simulations import  ClassificationSimulation, RegressionSimulation

class OpenmlSource(Source[Union[Iterable[Tuple[_T_Data, str]], Iterable[Tuple[_T_Data, Number]]]]):

    def __init__(self, id:int, problem_type:str = "classification", cat_as_str:bool=False, take:int = None, md5_checksum:str = None):

        assert problem_type in ["classification", "regression"]

        self._data_id      = id
        self._md5_checksum = md5_checksum
        self._problem_type = problem_type 
        self._cat_as_str   = cat_as_str
        self._take         = take
        self._cached_urls  = []

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the environment."""

        if self._take is not None:
            return { "openml": self._data_id, "cat_as_str": self._cat_as_str, "openml_type": self._problem_type, "openml_take": self._take,  }
        else:
            return { "openml": self._data_id, "cat_as_str": self._cat_as_str, "openml_type": self._problem_type, }

    def read(self) -> Union[Iterable[Tuple[_T_Data, str]], Iterable[Tuple[_T_Data, Number]]]:

        try:
            data_id        = self._data_id
            md5_checksum   = self._md5_checksum

            dataset_description  = self._get_dataset_description(data_id)

            if dataset_description['status'] == 'deactivated':
                raise CobaException(f"Openml {data_id} has been deactivated. This is often due to flags on the data.")
            
            feature_descriptions = self._get_feature_descriptions(data_id)

            encoders = defaultdict(lambda:StringEncoder())
            ignored = []
            target  = ""

            for description in feature_descriptions:

                header = description['name'].strip().strip('\'"')
                
                is_ignored = (
                    description['is_ignore'        ] == 'true' or 
                    description['is_row_identifier'] == 'true' or
                    description['data_type'        ] not in ['numeric', 'nominal']
                )

                if is_ignored:
                    ignored.append(header)

                if description['is_target'] == 'true':
                    target = header
                    
                if description['data_type'] == 'numeric':
                    encoders[header] = NumericEncoder()
                elif description['data_type'] == 'nominal' and self._cat_as_str:
                    encoders[header] = StringEncoder()
                elif description['data_type'] == 'nominal' and not self._cat_as_str:
                    # it happens moderately often that these values are wrong, #description["nominal_value"]
                    encoders[header] = OneHotEncoder() 

            if target != "":
                target_encoder = encoders[target]
                required_encoder = NumericEncoder if self._problem_type == "regression" else (OneHotEncoder,StringEncoder)

            if target == "" or not isinstance(target_encoder, required_encoder):
                target = self._get_target_for_problem_type(data_id)
            
            if target in ignored:
                ignored.pop(ignored.index(target))

            if self._problem_type == "classification":
                encoders[target] = StringEncoder() if self._cat_as_str else OneHotEncoder()

            file_rows = self._get_dataset_rows(dataset_description["file_id"], target, md5_checksum)

            def row_has_missing_values(row):
                row_values = row.values() if isinstance(row,dict) else row
                return "?" in row_values or "" in row_values
            
            drops     = Drop(drop_cols=ignored, drop_row=row_has_missing_values)
            takes     = Take(self._take, keep_first=True, seed=1)
            defaults  = Default({target:"0"})
            encodes   = Encode(encoders)
            structure = Structure([None, target])

            return Pipe.join([drops, takes, defaults, encodes, structure]).filter(file_rows)

        except KeyboardInterrupt:
            #we don't want to clear the cache in the case of a KeyboardInterrupt
            raise
        
        except CobaException:
            #we don't want to clear the cache if it is an error we know about (we the original raise would clear if needed)
            raise

        except Exception:
            #if something unexpected went wrong clear the
            #cache just in case it was corrupted somehow
            
            for key in self._cached_urls:
                CobaConfig.cacher.rmv(key)

            raise

    def _get_url(self, url:str, checksum:str=None) -> Iterable[str]:
        
        if url in CobaConfig.cacher:
            self._cached_urls.append(url)
            for b in CobaConfig.cacher.get(url):
                yield b.decode('utf-8')
        else:

            api_key  = CobaConfig.api_keys['openml']
            with HttpIO(url + (f'?api_key={api_key}' if api_key else '')).read() as response:

                if response.status_code == 412: # pragma: no cover
                    
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

                if response.status_code == 404: # pragma: no cover
                    message = (
                        "We're sorry but we were unable to find the requested dataset on openml.")
                    raise CobaException(message)

                # NOTE: These two checks need to be gated with a status code failure 
                # NOTE: otherwise this will cause the data to be downloaded all at once
                # NOTE: unfortunately I don't know the appropriate status code so commenting out for now
                # if "Usually due to high server load" in response.text:
                #     message = (
                #         "Openml has experienced an error that they believe is the result of high server loads."
                #         "Openml recommends that you try again in a few seconds. Additionally, if not already "
                #         "done, consider setting up a DiskCache in CobaConfig to reduce the number of openml "
                #         "calls in the future.")
                #     raise CobaException(message) from None

                # if '' == response.text: # pragma: no cover
                #     raise CobaException("Openml experienced an unexpected error. Please try requesting the data again.") from None

                bites = response.iter_lines(decode_unicode=False)

                if url not in CobaConfig.cacher:
                    CobaConfig.cacher.put(url,bites)
                    
                    if url in CobaConfig.cacher:
                        self._cached_urls.append(url)
                        bites = CobaConfig.cacher.get(url)

            # This can't reasonably be done in a streaming manner unless caching to disk so commenting out for now. 
            # if checksum is not None and md5(bites).hexdigest() != checksum:

            #     #if the cache has become corrupted we need to clear it
            #     CobaConfig.cacher.rmv(url)

            #     message = (
            #         f"The response from {url} did not match the given checksum {checksum}. This could be the result "
            #         "of network errors or the file becoming corrupted. Please consider downloading the file again. "
            #         "If the error persists you may want to manually download and reference the file.")
            #     raise CobaException(message) from None

                for b in bites:
                    yield b.decode('utf-8')

    def _get_dataset_description(self, data_id:int) -> Dict[str,Any]:

        description_txt = " ".join(list(self._get_url(f'https://www.openml.org/api/v1/json/data/{data_id}')))
        description_obj = json.loads(description_txt)["data_set_description"]

        return description_obj

    def _get_feature_descriptions(self, data_id:int) -> Sequence[Dict[str,Any]]:

        types_txt = " ".join(self._get_url(f'https://www.openml.org/api/v1/json/data/features/{data_id}'))
        types_obj = json.loads(types_txt)["data_features"]["feature"]

        return types_obj

    def _get_dataset_rows(self, file_id:str, target:str, md5_checksum:str) -> Any:

            csv_url  = f"http://www.openml.org/data/v1/get_csv/{file_id}"
            arff_url = f"http://www.openml.org/data/v1/download/{file_id}"

            openml_dialect = dict(quotechar="'", escapechar="\\", doublequote=False)

            if arff_url in CobaConfig.cacher:
                return ArffReader(skip_encoding=True, **openml_dialect).filter(self._get_url(arff_url, md5_checksum))
            
            if csv_url in CobaConfig.cacher:
                return CsvReader(True, **openml_dialect).filter(self._get_url(csv_url, md5_checksum))
            
            #we lock on downloading a dataset in case a user is running multithreaded and chunked by task
            #in this case they could download the same data set repeatedly. This also makes requests a little
            #nicer in terms of the load we place on openml's file server. srcsema should only be set when a
            #user is running an experiment in a multithreaded environment.
            
            srcsema = CobaConfig.store.get("srcsema")
            if srcsema: srcsema.acquire()

            try:
                return CsvReader(True, **openml_dialect).filter(self._get_url(csv_url, md5_checksum))
            except:
                return ArffReader(skip_encoding=True, **openml_dialect).filter(self._get_url(arff_url, md5_checksum))
            finally:
                if srcsema: srcsema.release()

    def _get_target_for_problem_type(self, data_id:int):

        text  = " ".join(self._get_url(f'https://www.openml.org/api/v1/json/task/list/data_id/{data_id}'))
        tasks = json.loads(text).get("tasks",{}).get("task",[])

        task_type = 1 if self._problem_type == "classification" else 2

        for task in tasks:
            if task["task_type_id"] == task_type: #aka, classification task
                for input in task['input']:
                    if input['name'] == 'target_feature':
                        return input['value'].strip().strip('\'"') #just take the first one

        raise CobaException(f"Openml {data_id} does not appear to be a {self._problem_type} dataset")

    def __repr__(self) -> str:
        return f'{{"OpenmlSource":{self._data_id}}}'

class OpenmlSimulation(SimulatedEnvironment):
    """A simulation created from openml data with features and labels.

    OpenmlSimulation turns labeled observations from a classification data set,
    into interactions. For each interaction the feature set becomes the context and 
    all possible labels become the actions. Rewards for each interaction are created by 
    assigning a reward of 1 for taking the correct action (i.e., choosing the correct
    label) and a reward of 0 for taking any other action (i.e., choosing any of the
    incorrect lables).
    """

    def __init__(self, id: int, take:int = None, simulation_type:Literal["classification","regression"] = "classification", cat_as_str:bool = False, md5_checksum: str = None) -> None:
        self._sim_type = simulation_type
        self._source = OpenmlSource(id, simulation_type, cat_as_str, take, md5_checksum)

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the simulation."""
        return self._source.params

    def read(self) -> Iterable[SimulatedInteraction]:
        """Read the interactions in this simulation."""

        if self._sim_type == "classification":
            return ClassificationSimulation(self._source.read()).read()
        else:
            return RegressionSimulation(self._source.read()).read()

    def __repr__(self) -> str:
        return f"OpenmlSimulation(id={self.params['openml']}, cat_as_str={self.params['cat_as_str']}, take={self.params.get('openml_take')})"

    def __str__(self) -> str:
        return self.__repr__()
