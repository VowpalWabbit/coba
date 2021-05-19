import json

from itertools import compress, count, repeat
from hashlib import md5
from typing import Tuple, Sequence, Any, List, cast

from coba.pipes import Source, HttpSource
from coba.utilities import CobaConfig

from coba.simulations.core import Context, Action, Interaction, ClassificationReward, MemorySimulation, Simulation

class OpenmlSource(Source[Tuple[Sequence[Context], Sequence[Action]]]):

    def __init__(self, id:int, md5_checksum:str = None):
        self._data_id      = id
        self._md5_checksum = md5_checksum

    def read(self) -> Tuple[Sequence[Sequence[Any]], Sequence[Any]]:
        
        #placing some of these at the top would cause circular references
        from coba.encodings import Encoder, NumericEncoder, OneHotEncoder, StringEncoder
        from coba.pipes     import ArffReader, CsvReader, Encode, Flatten, Transpose

        d_key = None
        t_key = None
        o_key = None

        try:
            data_id        = self._data_id
            md5_checksum   = self._md5_checksum

            d_key = f'https://www.openml.org/api/v1/json/data/{data_id}'
            t_key = f'https://www.openml.org/api/v1/json/data/features/{data_id}'

            d_bytes  = self._query(d_key, "descr")
            d_object = json.loads(d_bytes.decode('utf-8'))["data_set_description"]

            if d_object['status'] == 'deactivated':
                raise Exception(f"Openml {data_id} has been deactivated. This is often due to flags on the data.")

            t_bytes  = self._query(t_key, "types")
            t_object = json.loads(t_bytes.decode('utf-8'))["data_features"]["feature"]

            headers : List[str]     = []
            encoders: List[Encoder] = []
            ignored : List[bool]    = []
            target  : str           = ""

            for tipe in t_object:

                headers.append(tipe['name'].lower())
                ignored.append(tipe['is_ignore'] == 'true' or tipe['is_row_identifier'] == 'true')

                if tipe['is_target'] == 'true':
                    target = tipe['name'].lower()

                if tipe['data_type'] == 'numeric':
                    encoders.append(NumericEncoder())  
                elif tipe['data_type'] == 'nominal' and tipe['is_target'] == 'false':
                    encoders.append(OneHotEncoder(singular_if_binary=True))
                elif tipe['data_type'] == 'nominal' and tipe['is_target'] == 'true':
                    encoders.append(OneHotEncoder())
                else:
                    encoders.append(StringEncoder())

            if isinstance(encoders[headers.index(target)], NumericEncoder):
                target = self._get_classification_target(data_id)
                ignored[headers.index(target)] = False
                if isinstance(encoders[headers.index(target)], OneHotEncoder):
                    cast(OneHotEncoder,encoders[headers.index(target)])._singular_if_binary = False
                else:
                    encoders[headers.index(target)] = OneHotEncoder()

            csv_url = f"http://www.openml.org/data/v1/get_csv/{d_object['file_id']}"
            arff_url = f"http://www.openml.org/data/v1/download/{d_object['file_id']}"

            try:
                if csv_url in CobaConfig.Cacher or arff_url not in CobaConfig.Cacher:
                    o_key     = csv_url
                    o_bytes   = self._query(o_key, "obser", md5_checksum)
                    file_rows = list(CsvReader().filter(o_bytes.decode('utf-8').splitlines()))
                else:
                    o_key     = arff_url
                    o_bytes   = self._query(o_key, "obser", md5_checksum)
                    file_rows = list(ArffReader().filter(o_bytes.decode('utf-8').splitlines()))
            except:
                if o_key == csv_url:
                    o_key     = arff_url
                    o_bytes   = self._query(o_key, "obser", md5_checksum)
                    file_rows = list(ArffReader().filter(o_bytes.decode('utf-8').splitlines()))
                else:
                    o_key     = csv_url
                    o_bytes   = self._query(o_key, "obser", md5_checksum)
                    file_rows = list(CsvReader().filter(o_bytes.decode('utf-8').splitlines()))

            is_sparse_data = isinstance(file_rows[0], tuple) and len(file_rows[0]) == 2

            if is_sparse_data:
                file_headers  = [ header.lower() for header in file_rows.pop(0)[1]]
            else:
                file_headers  = [ header.lower() for header in file_rows.pop(0)]

            file_encoders = [ encoders[file_headers.index(header)] for header in headers]

            file_cols = list(Transpose().filter(file_rows))

            if is_sparse_data:
                target_index = file_headers.index(target)
                
                target_col   = file_cols[target_index]
                missing_rows = list(set(range(len(file_rows))) - set(target_col[0]))
                
                #we densify the target column to make sure classes of value '0' are encoded with the rest of the target column
                file_cols[target_index] = ( target_col[0] + tuple(missing_rows),  target_col[1] + tuple(['0']*len(missing_rows)) )

            file_cols = list(Encode(file_encoders).filter(file_cols))

            for ignored_header in compress(headers, ignored):
                file_cols.pop(file_headers.index(ignored_header))
                file_headers.remove(ignored_header)

            label_col    = file_cols.pop(file_headers.index(target))
            feature_rows = list(Transpose().filter(Flatten().filter(file_cols)))

            if is_sparse_data:
                label_rows = label_col[0]
                label_col  = label_col[1]

                #A quick sanity check just in case the label column is missing values
                feature_rows = [feature_rows[i] for i in label_rows]

            assert len(feature_rows) == len(label_col),  "Mismatched lengths of features and labels"

            #we only cache after all the data has been successfully loaded
            for key,bytes in [ (d_key, d_bytes), (t_key, t_bytes), (o_key, o_bytes) ]:
                if key not in CobaConfig.Cacher: CobaConfig.Cacher.put(key,bytes)

            return feature_rows, label_col

        except KeyboardInterrupt:
            raise
        
        except Exception as e:
            #if something went wrong we want to clear the 
            #cache just in case it was corrupted somehow
            for k in [d_key, t_key, o_key]:
                if k is not None: CobaConfig.Cacher.rmv(k)
            
            raise

    def _query(self, url:str, description:str, checksum:str=None) -> bytes:
        
        if url in CobaConfig.Cacher:
            #with CobaConfig.Logger.time(f'loading {description} from cache... '):
            bites = CobaConfig.Cacher.get(url)
        else:

            api_key = CobaConfig.Api_Keys['openml']

            #with CobaConfig.Logger.time(f'loading {description} from http... '):
            response = HttpSource(url + (f'?api_key={api_key}' if api_key else '')).read()

            if response.status_code == 412:
                if 'please provide api key' in response.text:
                    message = (
                        "An API Key is needed to access openml's rest API. A key can be obtained by creating an "
                        "openml account at openml.org. Once a key has been obtained it should be placed within "
                        "~/.coba as { \"api_keys\" : { \"openml\" : \"<your key here>\", } }.")
                    raise Exception(message) from None

                if 'authentication failed' in response.text:
                    message = (
                        "The API Key you provided no longer seems to be valid. You may need to create a new one"
                        "longing into your openml account and regenerating a key. After regenerating the new key "
                        "should be placed in ~/.coba as { \"api_keys\" : { \"openml\" : \"<your key here>\", } }.")
                    raise Exception(message) from None

            if response.status_code == 404:
                message = (
                    "We're sorry but we were unable to find the requested dataset on openml. The most likely cause "
                    "for this is openml not providing the requested dataset in a format that COBA can process.")
                raise Exception(message) from None

            if "Usually due to high server load" in response.text:
                message = (
                    "Openml reported an error that they believe is likely caused by high server loads ."
                    "Openml recommends that you try again in a few seconds. Additionally, if not already "
                    "done, consider setting up a DiskCache in coba config to reduce the number of openml "
                    "calls in the future.")
                raise Exception(message) from None

            if '' == response.text:
                raise Exception("The http response was empty. Try re-running the benchmark.") from None

            bites = response.content

        if checksum is not None and md5(bites).hexdigest() != checksum:
            
            #if the cache has become corrupted we need to clear it
            CobaConfig.Cacher.rmv(url)

            message = (
                f"The response from {url} did not match the given checksum {checksum}. This could be the result "
                "of network errors or the file becoming corrupted. Please consider downloading the file again. "
                "If the error persists you may want to manually download and reference the file.")
            raise Exception(message) from None

        return bites

    def _get_classification_target(self, data_id):

        t_key = f'https://www.openml.org/api/v1/json/task/list/data_id/{data_id}'
        t_bites = self._query(t_key, "tasks")

        tasks = json.loads(t_bites.decode('utf-8'))["tasks"]["task"]
        
        if t_key not in CobaConfig.Cacher:
            CobaConfig.Cacher.put(t_key,t_bites)

        for task in tasks:
            if task["task_type_id"] == 1: #aka, classification task
                for input in task['input']:
                    if input['name'] == 'target_feature':
                        return input['value'] #just take the first one

        

        raise Exception(f"Openml {data_id} does not appear to be a classification dataset")

class OpenmlSimulation(Source[Simulation]):
    """A simulation created from openml data with features and labels.

    OpenmlSimulation turns labeled observations from a classification data set
    set, into interactions. For each interaction the feature set becomes the context and 
    all possible labels become the actions. Rewards for each interaction are created by 
    assigning a reward of 1 for taking the correct action (i.e., choosing the correct
    label) and a reward of 0 for taking any other action (i.e., choosing any of the
    incorrect lables).

    Remark:
        This class when created from a data set will load all data into memory. Be careful when 
        doing this if you are working with a large dataset. To reduce memory usage you can provide
        meta information upfront that will allow features to be correctly encoded while the
        dataset is being streamed instead of waiting until the end of the data to train an encoder.
    """

    def __init__(self, id: int, md5_checksum: str = None) -> None:
        self._source = OpenmlSource(id, md5_checksum)

    def read(self) -> Simulation:        
        feature_rows, label_col = self._source.read()

        actions      = list(set(label_col))
        reward       = ClassificationReward()
        interactions = cast(List[Interaction],[])
        
        for key, context, label in zip(count(), feature_rows, label_col):
            reward.add(key, label)
            interactions.append(Interaction(key, context, actions))

        return MemorySimulation(interactions, reward)

    def __repr__(self) -> str:
        return f'{{"OpenmlSimulation":{self._source._data_id}}}'