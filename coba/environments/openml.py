import json

from math import isnan
from itertools import compress
from hashlib import md5
from numbers import Number
from typing import Optional, Tuple, Sequence, Any, List, Iterable, Dict

from coba.pipes import Source, HttpIO
from coba.config import CobaConfig
from coba.exceptions import CobaException

from coba.environments.core import Context, Action, SimulatedInteraction, Simulation, ClassificationSimulation, RegressionSimulation

class OpenmlSource(Source[Tuple[Sequence[Context], Sequence[Action]]]):

    def __init__(self, id:int, problem_type:str = "classification", nominal_as_str:bool=False, md5_checksum:str = None):
        
        assert problem_type in ["classification", "regression"]
        
        self._data_id        = id
        self._md5_checksum   = md5_checksum
        self._problem_type   = problem_type 
        self._nominal_as_str = nominal_as_str
        self._cached_urls    = []

        #add flag to allow regression simulations
        #if a known exception happens still cache the query results

    def read(self) -> Tuple[Sequence[Sequence[Any]], Sequence[Any]]:
        
        #placing some of these at the top would cause circular references
        from coba.encodings import Encoder, NumericEncoder, OneHotEncoder, StringEncoder
        from coba.pipes     import Encode, Transpose

        try:
            data_id        = self._data_id
            md5_checksum   = self._md5_checksum

            dataset_description  = self._get_dataset_description(data_id)

            if dataset_description['status'] == 'deactivated':
                raise CobaException(f"Openml {data_id} has been deactivated. This is often due to flags on the data.")
            
            feature_descriptions = self._get_feature_descriptions(data_id)

            headers : List[str]     = []
            encoders: List[Encoder] = []
            ignored : List[bool]    = []
            target  : str           = ""

            for tipe in feature_descriptions:

                headers.append(tipe['name'].lower())
                ignored.append(tipe['is_ignore'] == 'true' or tipe['is_row_identifier'] == 'true')

                if tipe['is_target'] == 'true':
                    target = tipe['name'].lower()
                    
                if tipe['data_type'] == 'numeric':
                    encoders.append(NumericEncoder())  
                elif tipe['data_type'] == 'nominal':
                    if self._nominal_as_str:
                        encoders.append(StringEncoder())
                    else:
                        encoders.append(OneHotEncoder(singular_if_binary=True))
                else:
                    ignored[-1] = True
                    encoders.append(StringEncoder())

            if target != "":
                target_encoder = encoders[headers.index(target)]
                problem_encoder = NumericEncoder if self._problem_type == "regression" else (OneHotEncoder,StringEncoder)

            if target == "" or not isinstance(target_encoder, problem_encoder):
                target = self._get_target_for_problem_type(data_id)
            
            ignored[headers.index(target)] = False

            if self._problem_type == "classification":
                target_encoder = StringEncoder() if self._nominal_as_str else OneHotEncoder(singular_if_binary=False)
                encoders[headers.index(target)] = target_encoder

            file_rows = self._get_dataset_rows(dataset_description["file_id"], target, md5_checksum)
            is_sparse = isinstance(file_rows[0], tuple) and len(file_rows[0]) == 2

            if is_sparse:
                file_headers  = [ header.lower() for header in file_rows.pop(0)[1]]
            else:
                file_headers  = [ header.lower() for header in file_rows.pop(0)]

            file_cols = list(Transpose().filter(file_rows))

            for ignored_header in compress(headers, ignored):
                if ignored_header in file_headers:
                    file_cols.pop(file_headers.index(ignored_header))
                    file_headers.remove(ignored_header)

            file_encoders = [ encoders[headers.index(file_header)] for file_header in file_headers]

            if is_sparse:
                label_col = file_cols[file_headers.index(target)]
                
                dense_label_col = ['0']*len(file_rows)
                
                for index, value in zip(label_col[0], label_col[1]):
                    dense_label_col[index] = value

                file_cols[file_headers.index(target)] = (tuple(range(len(file_rows))), tuple(dense_label_col))

            file_cols    = list(Encode(file_encoders).filter(file_cols))
            label_col    = file_cols.pop(file_headers.index(target))
            feature_rows = list(Transpose().filter(file_cols))

            if is_sparse:
                label_col = label_col[1]

            no_missing_values = [ not any(isinstance(val,Number) and isnan(val) for val in row) for row in feature_rows ]

            if not any(no_missing_values):
                raise CobaException(f"Every example in openml {data_id} has a missing value. The simulation is being ignored.")

            feature_rows = list(compress(feature_rows, no_missing_values))
            label_col    = list(compress(label_col, no_missing_values)) 
            
            return feature_rows, label_col

        except KeyboardInterrupt:
            #we don't want to clear the cache in the case of a KeyboardInterrupt
            raise
        
        except CobaException:
            raise

        except Exception:
            #if something unexpected went wrong clear the
            #cache just in case it was corrupted somehow
            
            for key in self._cached_urls:
                CobaConfig.cacher.rmv(key)
            
            raise

    def _get_url(self, url:str, checksum:str=None) -> bytes:
        
        if url in CobaConfig.cacher:
            bites = CobaConfig.cacher.get(url)
        else:

            api_key  = CobaConfig.api_keys['openml']
            response = HttpIO(url + (f'?api_key={api_key}' if api_key else '')).read()

            if response.status_code == 412:
                if 'please provide api key' in response.text:
                    message = (
                        "An API Key is needed to access openml's rest API. A key can be obtained by creating an "
                        "openml account at openml.org. Once a key has been obtained it should be placed within "
                        "~/.coba as { \"api_keys\" : { \"openml\" : \"<your key here>\", } }.")
                    raise CobaException(message) from None

                if 'authentication failed' in response.text:
                    message = (
                        "The API Key you provided no longer seems to be valid. You may need to create a new one"
                        "longing into your openml account and regenerating a key. After regenerating the new key "
                        "should be placed in ~/.coba as { \"api_keys\" : { \"openml\" : \"<your key here>\", } }.")
                    raise CobaException(message) from None

            if response.status_code == 404:
                message = (
                    "We're sorry but we were unable to find the requested dataset on openml. The most likely cause "
                    "for this is openml not providing the requested dataset in a format that COBA can process.")
                raise CobaException(message) from None

            if "Usually due to high server load" in response.text:
                message = (
                    "Openml has experienced an error that they believe is the result of high server loads."
                    "Openml recommends that you try again in a few seconds. Additionally, if not already "
                    "done, consider setting up a DiskCache in coba config to reduce the number of openml "
                    "calls in the future.")
                raise CobaException(message) from None

            if '' == response.text:
                raise CobaException("Openml experienced an unexpected error. Please try requesting the data again.") from None

            bites = response.content

            if url not in CobaConfig.cacher:
                self._cached_urls.append(url)
                CobaConfig.cacher.put(url,bites)

        if checksum is not None and md5(bites).hexdigest() != checksum:
            
            #if the cache has become corrupted we need to clear it
            CobaConfig.cacher.rmv(url)

            message = (
                f"The response from {url} did not match the given checksum {checksum}. This could be the result "
                "of network errors or the file becoming corrupted. Please consider downloading the file again. "
                "If the error persists you may want to manually download and reference the file.")
            raise CobaException(message) from None

        return bites.decode('utf-8')

    def _get_dataset_description(self, data_id:int) -> Dict[str,Any]:

        description_txt = self._get_url(f'https://www.openml.org/api/v1/json/data/{data_id}')
        description_obj = json.loads(description_txt)["data_set_description"]

        return description_obj

    def _get_feature_descriptions(self, data_id:int) -> Sequence[Dict[str,Any]]:
        
        types_txt = self._get_url(f'https://www.openml.org/api/v1/json/data/features/{data_id}')
        types_obj = json.loads(types_txt)["data_features"]["feature"]

        return types_obj

    def _get_dataset_rows(self, file_id:str, target:str, md5_checksum:str) -> Any:
            from coba.pipes import ArffReader, CsvReader, Encode, Transpose
            
            csv_url  = f"http://www.openml.org/data/v1/get_csv/{file_id}"
            arff_url = f"http://www.openml.org/data/v1/download/{file_id}"

            if arff_url in CobaConfig.cacher:
                text = self._get_url(arff_url, md5_checksum)
                return list(ArffReader(skip_encoding=[target]).filter(text.splitlines()))

            try:
                text = self._get_url(csv_url, md5_checksum)
                return list(CsvReader().filter(text.splitlines()))
            
            except:
                text = self._get_url(arff_url, md5_checksum)
                return list(ArffReader(skip_encoding=[target]).filter(text.splitlines()))

    def _get_target_for_problem_type(self, data_id:int):

        text  = self._get_url(f'https://www.openml.org/api/v1/json/task/list/data_id/{data_id}')
        tasks = json.loads(text).get("tasks",{}).get("task",[])

        task_type = 1 if self._problem_type == "classification" else 2

        for task in tasks:
            if task["task_type_id"] == task_type: #aka, classification task
                for input in task['input']:
                    if input['name'] == 'target_feature':
                        return input['value'] #just take the first one

        raise CobaException(f"Openml {data_id} does not appear to be a {self._problem_type} dataset")

class OpenmlSimulation(Simulation):
    """A simulation created from openml data with features and labels.

    OpenmlSimulation turns labeled observations from a classification data set,
    into interactions. For each interaction the feature set becomes the context and 
    all possible labels become the actions. Rewards for each interaction are created by 
    assigning a reward of 1 for taking the correct action (i.e., choosing the correct
    label) and a reward of 0 for taking any other action (i.e., choosing any of the
    incorrect lables).
    """

    def __init__(self, id: int, simulation_type:str = "classification", nominal_as_str:bool = False, md5_checksum: str = None) -> None:
        self._source = OpenmlSource(id, simulation_type, nominal_as_str, md5_checksum)
        self._interactions: Optional[Sequence[SimulatedInteraction]] = None

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the simulation."""
        return { "openml": self._source._data_id}

    def read(self) -> Iterable[SimulatedInteraction]:
        """Read the interactions in this simulation."""

        features,labels = self._source.read()

        if isinstance(labels[0],Number):
            return RegressionSimulation(features,labels).read()
        else:
            return ClassificationSimulation(features,labels).read()

    def __repr__(self) -> str:
        return f'{{"OpenmlSimulation":{self._source._data_id}}}'
