import requests
import unittest.mock
import unittest
import json

from typing import cast, Tuple

from coba.exceptions   import CobaException
from coba.contexts     import CobaContext, CobaContext, NullLogger, MemoryCacher, NullCacher
from coba.environments import OpenmlSimulation

from coba.environments.simulated.openml import OpenmlSource

CobaContext.logger = NullLogger()

class PutOnceCacher(MemoryCacher):

    def put(self, key, value) -> None:

        if key in self: raise Exception("Writing data again without reason.")
        return super().put(key, value)

class ExceptionCacher(MemoryCacher):

    def __init__(self, failure_key, failure_exception) -> None:
        self._failure_key = failure_key
        self._failure_exception = failure_exception

        super().__init__()

    def get(self, key):

        if key == self._failure_key:
            raise self._failure_exception
        
        return super().get(key)

class MockResponse:
    def __init__(self, status_code, text, iter_lines):
        self.text        = text
        self.status_code = status_code
        self._iter_lines = iter_lines

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def iter_lines(self, *args, **kwargs):
        return self._iter_lines

class OpenmlSource_Tests(unittest.TestCase):
    
    def setUp(self) -> None:
        CobaContext.api_keys = {'openml': None}
        CobaContext.cacher   = MemoryCacher()
        CobaContext.logger   = NullLogger()

    def test_already_cached_values_are_not_cached_again(self):

        CobaContext.cacher = PutOnceCacher()

        data_set_description = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
            }
        }

        data_set_features = {
            "data_features":{
                "feature":[
                    {"index":"0","name":"pH"          ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"1","name":"temperature" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"2","name":"conductivity","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"3","name":"coli"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"4","name":"play"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"}
                ]
            }
        }

        data_set_arff = """
            @relation weather
            
            @attribute pH real
            @attribute temperature real
            @attribute conductivity real
            @attribute coli {2, 1}
            @attribute play {no, yes}
            
            @data
            8.1,27,1410,2,no
            8.2,29,1180,2,no
            8.2,28,1410,2,yes
            8.3,27,1020,1,yes
            7.6,23,4700,1,yes
        """

        data_set_tasks = {
            "tasks":{
                "task":[
                    { "task_id":338754, "task_type_id":1, "status":"active", "input": [{"name":"target_feature", "value":"play"}]}, 
                    { "task_id":359909, "task_type_id":5, "status":"active" } 
                ]
            }
        }

        CobaContext.cacher.put('openml_042693_descr', json.dumps(data_set_description).encode().splitlines())
        CobaContext.cacher.put('openml_042693_feats', json.dumps(data_set_features).encode().splitlines())
        CobaContext.cacher.put('openml_042693_arff' , data_set_arff.encode().splitlines() )
        CobaContext.cacher.put('openml_042693_tasks', json.dumps(data_set_tasks).encode().splitlines() )

        features,labels = list(zip(*OpenmlSource(42693).read()))

        self.assertEqual(len(features), 5)
        self.assertEqual(len(labels), 5)

        self.assertEqual([8.1, 27, 1410, (1,0)], features[0])
        self.assertEqual([8.2, 29, 1180, (1,0)], features[1])
        self.assertEqual([8.2, 28, 1410, (1,0)], features[2])
        self.assertEqual([8.3, 27, 1020, (0,1)], features[3])
        self.assertEqual([7.6, 23, 4700, (0,1)], features[4])

        self.assertEqual((1,0), labels[0])
        self.assertEqual((1,0), labels[1])
        self.assertEqual((0,1), labels[2])
        self.assertEqual((0,1), labels[3])
        self.assertEqual((0,1), labels[4])

    def test_dataset_deactivated(self):

        data_set_description = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"deactivated",
            }
        }


        CobaContext.cacher.put('openml_042693_descr', json.dumps(data_set_description).encode().splitlines())

        with self.assertRaises(CobaException) as e:
            feature_rows, label_col = OpenmlSource(42693).read()

        self.assertTrue("has been deactivated" in str(e.exception))

    def test_classification_type_classification_dataset_with_missing_values(self):

        data_set_description = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
            }
        }

        data_set_features = {
            "data_features":{
                "feature":[
                    {"index":"0","name":"pH"          ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"1","name":"temperature" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"2","name":"conductivity","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"3","name":"coli"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"4","name":"play"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"}
                ]
            }
        }

        data_set_arff = """
            @relation weather
            
            @attribute pH real
            @attribute temperature real
            @attribute conductivity real
            @attribute coli {2, 1}
            @attribute play {no, yes}
            
            @data
            ?,27,1410,2,no
            8.2,29,1180,2,no
            8.2,,1410,2,yes
            8.3,27,1020,1,yes
            7.6,23,4700,1,yes
        """

        data_set_tasks = {
            "tasks":{
                "task":[
                    { "task_id":338754, "task_type_id":1, "status":"active", "input": [{"name":"target_feature", "value":"play"}]}, 
                    { "task_id":359909, "task_type_id":5, "status":"active" } 
                ]
            }
        }

        CobaContext.cacher.put('openml_042693_descr', json.dumps(data_set_description).encode().splitlines())
        CobaContext.cacher.put('openml_042693_feats', json.dumps(data_set_features).encode().splitlines())
        CobaContext.cacher.put('openml_042693_arff' , data_set_arff.encode().splitlines() )
        CobaContext.cacher.put('openml_042693_tasks', json.dumps(data_set_tasks).encode().splitlines() )

        feature_rows, label_col = list(zip(*OpenmlSource(42693).read()))

        self.assertEqual(len(feature_rows), 3)
        self.assertEqual(len(label_col), 3)

        self.assertEqual([8.2, 29, 1180, (1,0)], feature_rows[0])
        self.assertEqual([8.3, 27, 1020, (0,1)], feature_rows[1])
        self.assertEqual([7.6, 23, 4700, (0,1)], feature_rows[2])

        self.assertEqual((1,0), label_col[0])
        self.assertEqual((0,1), label_col[1])
        self.assertEqual((0,1), label_col[2])

    def test_classification_type_classification_dataset_cat_as_str(self):

        data_set_description = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
            }
        }

        data_set_features = {
            "data_features":{
                "feature":[
                    {"index":"0","name":"pH"          ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"1","name":"temperature" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"2","name":"conductivity","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"3","name":"coli"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"4","name":"play"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"}
                ]
            }
        }

        data_set_arff = """
            @relation weather
            
            @attribute pH real
            @attribute temperature real
            @attribute conductivity real
            @attribute coli {2, 1}
            @attribute play {no, yes}
            
            @data
            8.1,27,1410,2,no
            8.2,29,1180,2,no
            8.2,28,1410,2,yes
            8.3,27,1020,1,yes
            7.6,23,4700,1,yes
        """

        data_set_tasks = {
            "tasks":{
                "task":[
                    { "task_id":338754, "task_type_id":1, "status":"active", "input": [{"name":"target_feature", "value":"play"}]}, 
                    { "task_id":359909, "task_type_id":5, "status":"active" } 
                ]
            }
        }

        CobaContext.cacher.put('openml_042693_descr', json.dumps(data_set_description).encode().splitlines())
        CobaContext.cacher.put('openml_042693_feats', json.dumps(data_set_features).encode().splitlines())
        CobaContext.cacher.put('openml_042693_arff' , data_set_arff.encode().splitlines() )
        CobaContext.cacher.put('openml_042693_tasks', json.dumps(data_set_tasks).encode().splitlines() )

        feature_rows, label_col = list(zip(*OpenmlSource(42693, cat_as_str=True).read()))

        self.assertEqual(len(feature_rows), 5)
        self.assertEqual(len(label_col), 5)

        self.assertEqual([8.1, 27, 1410, '2'], feature_rows[0])
        self.assertEqual([8.2, 29, 1180, '2'], feature_rows[1])
        self.assertEqual([8.2, 28, 1410, '2'], feature_rows[2])
        self.assertEqual([8.3, 27, 1020, '1'], feature_rows[3])
        self.assertEqual([7.6, 23, 4700, '1'], feature_rows[4])

        self.assertEqual('no', label_col[0])
        self.assertEqual('no', label_col[1])
        self.assertEqual('yes', label_col[2])
        self.assertEqual('yes', label_col[3])
        self.assertEqual('yes', label_col[4])

    def test_regression_type_regression_dataset(self):

        data_set_description = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
            }
        }

        data_set_features = {
            "data_features":{
                "feature":[
                    {"index":"0","name":"pH"          ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"1","name":"temperature" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"2","name":"conductivity","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"3","name":"coli"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"4","name":"play"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"}
                ]
            }
        }

        data_set_arff = """
            @relation weather
            
            @attribute pH real
            @attribute temperature real
            @attribute conductivity real
            @attribute coli {2, 1}
            @attribute play {no, yes}
            
            @data
            8.1,27,1410,2,no
            8.2,29,1180,2,no
            8.2,28,1410,2,yes
            8.3,27,1020,1,yes
            7.6,23,4700,1,yes
        """

        data_set_tasks = {
            "tasks":{
                "task":[
                    { "task_id":338754, "task_type_id":2, "status":"active", "input": [{"name":"target_feature", "value":"pH"}]}, 
                    { "task_id":359909, "task_type_id":5, "status":"active" } 
                ]
            }
        }

        CobaContext.cacher.put('openml_042693_descr', json.dumps(data_set_description).encode().splitlines())
        CobaContext.cacher.put('openml_042693_feats', json.dumps(data_set_features).encode().splitlines())
        CobaContext.cacher.put('openml_042693_arff' , data_set_arff.encode().splitlines() )
        CobaContext.cacher.put('openml_042693_tasks', json.dumps(data_set_tasks).encode().splitlines() )

        feature_rows, label_col = list(zip(*OpenmlSource(42693, problem_type="regression").read()))

        self.assertEqual(len(feature_rows), 5)
        self.assertEqual(len(label_col), 5)

        self.assertEqual([27, 1410, (1,0), (1,0)], feature_rows[0])
        self.assertEqual([29, 1180, (1,0), (1,0)], feature_rows[1])
        self.assertEqual([28, 1410, (1,0), (0,1)], feature_rows[2])
        self.assertEqual([27, 1020, (0,1), (0,1)], feature_rows[3])
        self.assertEqual([23, 4700, (0,1), (0,1)], feature_rows[4])

        self.assertEqual(8.1, label_col[0])
        self.assertEqual(8.2, label_col[1])
        self.assertEqual(8.2, label_col[2])
        self.assertEqual(8.3, label_col[3])
        self.assertEqual(7.6, label_col[4])

    def test_classification_type_regression_dataset_cache_exception(self):

        data_set_description = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
            }
        }

        data_set_features = {
            "data_features":{
                "feature":[
                    {"index":"0","name":"pH"          ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"1","name":"temperature" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"2","name":"conductivity","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"3","name":"coli"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"4","name":"play"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"}
                ]
            }
        }

        data_set_tasks = {
            "tasks":{
                "task":[
                    { "task_id":338754, "task_type_id":5, "status":"active" }, 
                    { "task_id":359909, "task_type_id":5, "status":"active" } 
                ]
            }
        }

        CobaContext.cacher = ExceptionCacher('openml_042693_tasks', CobaException())

        CobaContext.cacher.put('openml_042693_descr', json.dumps(data_set_description).encode().splitlines())
        CobaContext.cacher.put('openml_042693_feats', json.dumps(data_set_features).encode().splitlines())
        CobaContext.cacher.put('openml_042693_tasks', json.dumps(data_set_tasks).encode().splitlines() )

        with self.assertRaises(Exception) as e:
            feature_rows, label_col = OpenmlSource(42693).read()

        self.assertTrue("does not appear" in str(e.exception))

    def test_classification_type_regression_dataset_no_classification_tasks(self):

        data_set_description = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
            }
        }

        data_set_features = {
            "data_features":{
                "feature":[
                    {"index":"0","name":"pH"          ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"1","name":"temperature" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"2","name":"conductivity","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"3","name":"coli"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"4","name":"play"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"}
                ]
            }
        }

        data_set_tasks = {
            "tasks":{
                "task":[
                    { "task_id":338754, "task_type_id":5, "status":"active" }, 
                    { "task_id":359909, "task_type_id":5, "status":"active" } 
                ]
            }
        }

        CobaContext.cacher.put('openml_042693_descr', json.dumps(data_set_description).encode().splitlines())
        CobaContext.cacher.put('openml_042693_feats', json.dumps(data_set_features).encode().splitlines())
        CobaContext.cacher.put('openml_042693_tasks', json.dumps(data_set_tasks).encode().splitlines() )

        with self.assertRaises(Exception) as e:
            feature_rows, label_col = OpenmlSource(42693).read()

        self.assertTrue("does not appear" in str(e.exception))

    def test_classification_type_regression_dataset_no_tasks(self):

        data_set_description = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
            }
        }

        data_set_features = {
            "data_features":{
                "feature":[
                    {"index":"0","name":"pH"          ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"1","name":"temperature" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"2","name":"conductivity","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"3","name":"coli"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"4","name":"play"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"}
                ]
            }
        }

        data_set_tasks = { }

        CobaContext.cacher.put('openml_042693_descr', json.dumps(data_set_description).encode().splitlines())
        CobaContext.cacher.put('openml_042693_feats', json.dumps(data_set_features).encode().splitlines())
        CobaContext.cacher.put('openml_042693_tasks', json.dumps(data_set_tasks).encode().splitlines() )

        with self.assertRaises(CobaException) as e:
            feature_rows, label_col = OpenmlSource(42693).read()

        self.assertTrue("does not appear" in str(e.exception))

    def test_classification_type_regression_dataset_classification_tasks(self):

        data_set_description = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
            }
        }

        data_set_features = {
            "data_features":{
                "feature":[
                    {"index":"0","name":"pH"          ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"1","name":"temperature" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"2","name":"conductivity","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"3","name":"coli"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"4","name":"play"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"}
                ]
            }
        }

        data_set = """
            @relation weather
            
            @attribute pH real
            @attribute temperature real
            @attribute conductivity real
            @attribute coli {2, 1}
            @attribute play {no, yes}
            
            @data
            8.1,27,1410,2,no
            8.2,29,1180,2,no
            8.2,28,1410,2,yes
            8.3,27,1020,1,yes
            7.6,23,4700,1,yes
        """

        data_set_tasks = {
            "tasks":{
                "task":[
                    { "task_id":338754, "task_type_id":1, "status":"active", "input": [{"name":"target_feature", "value":"coli"}]}, 
                    { "task_id":359909, "task_type_id":5, "status":"active" },
                ]
            }
        }

        CobaContext.cacher.put('openml_042693_descr', json.dumps(data_set_description).encode().splitlines())
        CobaContext.cacher.put('openml_042693_feats', json.dumps(data_set_features).encode().splitlines())
        CobaContext.cacher.put('openml_042693_arff' , data_set.encode().splitlines() )
        CobaContext.cacher.put('openml_042693_tasks', json.dumps(data_set_tasks).encode().splitlines() )

        feature_rows, label_col = list(zip(*OpenmlSource(42693).read()))

        self.assertEqual(len(feature_rows), 5)
        self.assertEqual(len(label_col), 5)

        self.assertEqual([8.1, 27, 1410, (1,0)], feature_rows[0])
        self.assertEqual([8.2, 29, 1180, (1,0)], feature_rows[1])
        self.assertEqual([8.2, 28, 1410, (0,1)], feature_rows[2])
        self.assertEqual([8.3, 27, 1020, (0,1)], feature_rows[3])
        self.assertEqual([7.6, 23, 4700, (0,1)], feature_rows[4])

        self.assertEqual((1,0), label_col[0])
        self.assertEqual((1,0), label_col[1])
        self.assertEqual((1,0), label_col[2])
        self.assertEqual((0,1), label_col[3])
        self.assertEqual((0,1), label_col[4])

    def test_classification_type_regression_dataset_classification_tasks_targeting_ignored_col(self):

        data_set_description = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
            }
        }

        data_set_features = {
            "data_features":{
                "feature":[
                    {"index":"0","name":"pH"          ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"1","name":"temperature" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"2","name":"conductivity","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"3","name":"coli"        ,"data_type":"nominal","is_ignore":"true" ,"is_row_identifier":"false"},
                    {"index":"4","name":"play"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"}
                ]
            }
        }

        data_set_arff = """
            @relation weather
            
            @attribute pH real
            @attribute temperature real
            @attribute conductivity real
            @attribute coli {2, 1}
            @attribute play {no, yes}
            
            @data
            8.1,27,1410,2,no
            8.2,29,1180,2,no
            8.2,28,1410,2,yes
            8.3,27,1020,1,yes
            7.6,23,4700,1,yes
        """

        data_set_tasks = {
            "tasks":{
                "task":[
                    { "task_id":338754, "task_type_id":1, "status":"active", "input": [{"name":"target_feature", "value":"coli"}]}, 
                    { "task_id":359909, "task_type_id":5, "status":"active" },
                ]
            }
        }

        CobaContext.cacher.put('openml_042693_descr', json.dumps(data_set_description).encode().splitlines())
        CobaContext.cacher.put('openml_042693_feats', json.dumps(data_set_features).encode().splitlines())
        CobaContext.cacher.put('openml_042693_arff' , data_set_arff.encode().splitlines() )
        CobaContext.cacher.put('openml_042693_tasks', json.dumps(data_set_tasks).encode().splitlines() )

        feature_rows, label_col = list(zip(*OpenmlSource(42693).read()))

        self.assertEqual(len(feature_rows), 5)
        self.assertEqual(len(label_col), 5)

        self.assertEqual([8.1, 27, 1410, (1,0)], feature_rows[0])
        self.assertEqual([8.2, 29, 1180, (1,0)], feature_rows[1])
        self.assertEqual([8.2, 28, 1410, (0,1)], feature_rows[2])
        self.assertEqual([8.3, 27, 1020, (0,1)], feature_rows[3])
        self.assertEqual([7.6, 23, 4700, (0,1)], feature_rows[4])

        self.assertEqual((1,0), label_col[0])
        self.assertEqual((1,0), label_col[1])
        self.assertEqual((1,0), label_col[2])
        self.assertEqual((0,1), label_col[3])
        self.assertEqual((0,1), label_col[4])

    def test_classification_type_regression_dataset_classification_tasks_targeting_numeric_col(self):

        data_set_description = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
            }
        }

        data_set_features = {
            "data_features":{
                "feature":[
                    {"index":"0","name":"pH"          ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"1","name":"temperature" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"2","name":"conductivity","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"3","name":"coli"        ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"4","name":"play"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"}
                ]
            }
        }

        data_set = """
            @relation weather
            
            @attribute pH real
            @attribute temperature real
            @attribute conductivity real
            @attribute coli {2, 1}
            @attribute play {no, yes}
            
            @data
            8.1,27,1410,2,no
            8.2,29,1180,2,no
            8.2,28,1410,2,yes
            8.3,27,1020,1,yes
            7.6,23,4700,1,yes
        """

        data_set_tasks = {
            "tasks":{
                "task":[
                    { "task_id":338754, "task_type_id":1, "status":"active", "input": [{"name":"target_feature", "value":"coli"}]}, 
                    { "task_id":359909, "task_type_id":5, "status":"active" },
                ]
            }
        }

        CobaContext.cacher.put('openml_042693_descr', json.dumps(data_set_description).encode().splitlines())
        CobaContext.cacher.put('openml_042693_feats', json.dumps(data_set_features).encode().splitlines())
        CobaContext.cacher.put('openml_042693_arff' , data_set.encode().splitlines() )
        CobaContext.cacher.put('openml_042693_tasks', json.dumps(data_set_tasks).encode().splitlines() )

        feature_rows, label_col = list(zip(*OpenmlSource(42693).read()))

        self.assertEqual(len(feature_rows), 5)
        self.assertEqual(len(label_col), 5)

        self.assertEqual([8.1, 27, 1410, (1,0)], feature_rows[0])
        self.assertEqual([8.2, 29, 1180, (1,0)], feature_rows[1])
        self.assertEqual([8.2, 28, 1410, (0,1)], feature_rows[2])
        self.assertEqual([8.3, 27, 1020, (0,1)], feature_rows[3])
        self.assertEqual([7.6, 23, 4700, (0,1)], feature_rows[4])

        self.assertEqual((1,0), label_col[0])
        self.assertEqual((1,0), label_col[1])
        self.assertEqual((1,0), label_col[2])
        self.assertEqual((0,1), label_col[3])
        self.assertEqual((0,1), label_col[4])

    def test_classification_type_unsupervised_dataset_classification_tasks(self):

        data_set_description = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
            }
        }

        data_set_features = {
            "data_features":{
                "feature":[
                    {"index":"0","name":"pH"          ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"1","name":"temperature" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"2","name":"conductivity","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"3","name":"coli"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"4","name":"play"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"}
                ]
            }
        }

        data_set = """
            @relation weather
            
            @attribute pH real
            @attribute temperature real
            @attribute conductivity real
            @attribute coli {2, 1}
            @attribute play {no, yes}
            
            @data
            8.1,27,1410,2,no
            8.2,29,1180,2,no
            8.2,28,1410,2,yes
            8.3,27,1020,1,yes
            7.6,23,4700,1,yes
        """

        data_set_tasks = {
            "tasks":{
                "task":[
                    { "task_id":338754, "task_type_id":1, "status":"active", "input": [{"name":"target_feature", "value":"coli"}]}, 
                    { "task_id":359909, "task_type_id":5, "status":"active" },
                ]
            }
        }

        CobaContext.cacher.put('openml_042693_descr', json.dumps(data_set_description).encode().splitlines())
        CobaContext.cacher.put('openml_042693_feats', json.dumps(data_set_features).encode().splitlines())
        CobaContext.cacher.put('openml_042693_arff' , data_set.encode().splitlines() )
        CobaContext.cacher.put('openml_042693_tasks', json.dumps(data_set_tasks).encode().splitlines() )

        feature_rows, label_col = list(zip(*OpenmlSource(42693).read()))

        self.assertEqual(len(feature_rows), 5)
        self.assertEqual(len(label_col), 5)

        self.assertEqual([8.1, 27, 1410, (1,0)], feature_rows[0])
        self.assertEqual([8.2, 29, 1180, (1,0)], feature_rows[1])
        self.assertEqual([8.2, 28, 1410, (0,1)], feature_rows[2])
        self.assertEqual([8.3, 27, 1020, (0,1)], feature_rows[3])
        self.assertEqual([7.6, 23, 4700, (0,1)], feature_rows[4])

        self.assertEqual((1,0), label_col[0])
        self.assertEqual((1,0), label_col[1])
        self.assertEqual((1,0), label_col[2])
        self.assertEqual((0,1), label_col[3])
        self.assertEqual((0,1), label_col[4])

    def test_sparse_classification_type_classification_dataset(self):

        data_set_description = {
            "data_set_description":{
                "id":"1594",
                "name":"news20_test",
                "version":"2",
                "file_id":"1595696",
                "visibility":"public",
                "status":"active",
            }
        }

        data_set_features = {
            "data_features":{
                "feature":[
                    {"index":"0","name":"att_1","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"1","name":"att_2","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"2","name":"att_3","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"3","name":"att_4","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"4","name":"att_5","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"5","name":"att_6","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"6","name":"att_7","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"7","name":"att_8","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"8","name":"att_9","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"9","name":"att_10","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"10","name":"class","data_type":"nominal","nominal_value":["A","B","C","D"],"is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"}
                ]
            }
        }

        data_set_arff = """
            @relation news20
            
            @attribute att_1 numeric
            @attribute att_2 numeric
            @attribute att_3 numeric
            @attribute att_4 numeric
            @attribute att_5 numeric
            @attribute att_6 numeric
            @attribute att_7 numeric
            @attribute att_8 numeric
            @attribute att_9 numeric
            @attribute att_10 numeric
            @attribute class {B, C, D}
            
            @data
            {0 2,1 3}
            {2 1,3 1,4 1,6 1,8 1,10 B}
            {0 3,1 1,2 1,3 9,4 1,5 1,6 1,10 C}
            {0 1,3 1,6 1,7 1,8 1,9 2,10 D}
        """

        data_set_tasks = {
            "tasks":{
                "task":[
                    { "task_id":338754, "task_type_id":1, "status":"active", "input": [{"name":"target_feature", "value":"class"}]}, 
                    { "task_id":359909, "task_type_id":5, "task_type":"Clustering", "status":"active" }
                ]
            }
        }

        CobaContext.cacher.put('openml_001594_descr', json.dumps(data_set_description).encode().splitlines())
        CobaContext.cacher.put('openml_001594_feats', json.dumps(data_set_features).encode().splitlines())
        CobaContext.cacher.put('openml_001594_arff', data_set_arff.encode().splitlines())
        CobaContext.cacher.put('openml_001594_tasks', json.dumps(data_set_tasks).encode().splitlines())

        feature_rows, label_col = list(zip(*OpenmlSource(1594).read()))

        self.assertEqual(len(feature_rows), 4)
        self.assertEqual(len(label_col   ), 4)

        self.assertEqual(dict(zip( (0,1)          , (2,3)          )), feature_rows[0])
        self.assertEqual(dict(zip( (2,3,4,6,8)    , (1,1,1,1,1)    )), feature_rows[1])
        self.assertEqual(dict(zip( (0,1,2,3,4,5,6), (3,1,1,9,1,1,1))), feature_rows[2])
        self.assertEqual(dict(zip( (0,3,6,7,8,9)  , (1,1,1,1,1,2)  )), feature_rows[3])

        self.assertEqual((1,0,0,0), label_col[0])
        self.assertEqual((0,1,0,0), label_col[1])
        self.assertEqual((0,0,1,0), label_col[2])
        self.assertEqual((0,0,0,1), label_col[3])

    def test_cache_cleared_on_unexpected_exception(self):

        data_set_description = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
            }
        }

        data_set_features = {
            "data_features":{
                "feature":[
                    {"index":"0","name":"pH"          ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"1","name":"temperature" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"2","name":"conductivity","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"3","name":"coli"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"4","name":"play"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"}
                ]
            }
        }

        data_set_tasks = {
            "tasks":{
                "task":[
                    { "task_id":338754, "task_type_id":1, "status":"active", "input": [{"name":"target_feature", "value":"class"}]}, 
                    { "task_id":359909, "task_type_id":5, "task_type":"Clustering", "status":"active" }
                ]
            }
        }

        CobaContext.cacher = ExceptionCacher('openml_042693_arff', Exception())

        CobaContext.cacher.put('openml_042693_descr', json.dumps(data_set_description).encode().splitlines())
        CobaContext.cacher.put('openml_042693_feats', json.dumps(data_set_features).encode().splitlines())
        CobaContext.cacher.put('openml_042693_tasks', json.dumps(data_set_tasks).encode().splitlines())
        CobaContext.cacher.put('openml_042693_arff' , b"" )

        with self.assertRaises(Exception) as e:
            feature_rows, label_col = list(zip(*OpenmlSource(42693).read()))

        self.assertNotIn('openml_042693_descr', CobaContext.cacher)
        self.assertNotIn('openml_042693_feats', CobaContext.cacher)
        self.assertNotIn('openml_042693_arff' , CobaContext.cacher)

    def test_cache_not_cleared_on_keyboard_interrupt(self):

        data_set_description = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
            }
        }

        data_set_features = {
            "data_features":{
                "feature":[
                    {"index":"0","name":"pH"          ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"1","name":"temperature" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"2","name":"conductivity","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"3","name":"coli"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"4","name":"play"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"}
                ]
            }
        }

        data_set = """
            @relation weather
            
            @attribute pH real
            @attribute temperature real
            @attribute conductivity real
            @attribute coli {2, 1}
            @attribute play {no, yes}
            
            @data
            8.1,27,1410,2,no
            8.2,29,1180,2,no
            8.2,28,1410,2,yes
            8.3,27,1020,1,yes
            7.6,23,4700,1,yes
        """

        data_set_tasks = {
            "tasks": {
                "task": [
                    { "task_id":338754, "task_type_id":1, "status":"active", "input": [{"name":"target_feature", "value":"class"}]}, 
                    { "task_id":359909, "task_type_id":5, "status":"active" } 
                ]
            }
        }

        CobaContext.cacher = ExceptionCacher('openml_042693_arff', KeyboardInterrupt())

        CobaContext.cacher.put('openml_042693_descr', json.dumps(data_set_description).encode().splitlines())
        CobaContext.cacher.put('openml_042693_feats', json.dumps(data_set_features).encode().splitlines())
        CobaContext.cacher.put('openml_042693_arff' , data_set.encode().splitlines() )
        CobaContext.cacher.put('openml_042693_tasks', json.dumps(data_set_tasks).encode().splitlines() )

        with self.assertRaises(KeyboardInterrupt) as e:
            feature_rows, label_col = list(zip(*OpenmlSource(42693).read()))

        self.assertIn('openml_042693_descr', CobaContext.cacher)
        self.assertIn('openml_042693_feats', CobaContext.cacher)
        self.assertIn('openml_042693_arff' , CobaContext.cacher)

    def test_cache_cleared_on_cache_coba_exception(self):

        data_set_description = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
            }
        }

        data_set_features = {
            "data_features":{
                "feature":[
                    {"index":"0","name":"pH"          ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"1","name":"temperature" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"2","name":"conductivity","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"3","name":"coli"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"4","name":"play"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"}
                ]
            }
        }

        data_set = """
            @relation weather
            
            @attribute pH real
            @attribute temperature real
            @attribute conductivity real
            @attribute coli {2, 1}
            @attribute play {no, yes}
            
            @data
            8.1,27,1410,2,no
            8.2,29,1180,2,no
            8.2,28,1410,2,yes
            8.3,27,1020,1,yes
            7.6,23,4700,1,yes
        """

        data_set_tasks = {
            "tasks":{
                "task":[
                    { "task_id":338754, "task_type_id":1, "status":"active", "input": [{"name":"target_feature", "value":"class"}]}, 
                    { "task_id":359909, "task_type_id":5, "status":"active" } 
                ]
            }
        }

        CobaContext.cacher = ExceptionCacher('openml_042693_arff', CobaException())

        CobaContext.cacher.put('openml_042693_descr', json.dumps(data_set_description).encode().splitlines())
        CobaContext.cacher.put('openml_042693_feats', json.dumps(data_set_features).encode().splitlines())
        CobaContext.cacher.put('openml_042693_arff' , data_set.encode().splitlines() )
        CobaContext.cacher.put('openml_042693_tasks', json.dumps(data_set_tasks).encode().splitlines() )

        with self.assertRaises(Exception) as e:
            feature_rows, label_col = list(zip(*OpenmlSource(42693).read()))

        self.assertNotIn('openml_042693_descr', CobaContext.cacher)
        self.assertNotIn('openml_042693_feats', CobaContext.cacher)
        self.assertNotIn('openml_042693_arff' , CobaContext.cacher)

    def test_arff_classification_type_classification_dataset_from_http_with_sparse_csv_cache(self):

        data_set_csv = """
            "att_1","att_2","att_3","att_4","att_5","att_6","att_7","att_8","att_9","att_10","class"
            {0 2,1 3}
            {2 1,3 1,4 1,6 1,8 1,10 B}
            {0 3,1 1,2 1,3 9,4 1,5 1,6 1}
            {0 1,3 1,6 1,7 1,8 1,9 2,10 D}
        """

        CobaContext.cacher.put('openml_042693_csv'  , data_set_csv.encode().splitlines())

        data_set_description = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
            }
        }

        data_set_features = {
            "data_features":{
                "feature":[
                    {"index":"0","name":"pH"          ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"1","name":"temperature" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"2","name":"conductivity","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"3","name":"coli"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"4","name":"play"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"}
                ]
            }
        }

        data_set_tasks = {
            "tasks":{
                "task":[
                    { "task_id":338754, "task_type_id":1, "status":"active", "input": [{"name":"target_feature", "value":"play"}]}, 
                    { "task_id":359909, "task_type_id":5, "status":"active" } 
                ]
            }
        }

        data_set_arff = """
            @relation weather
            
            @attribute pH real
            @attribute temperature real
            @attribute conductivity real
            @attribute coli {2, 1}
            @attribute play {no, yes}
            
            @data
            8.1,27,1410,2,no
            8.2,29,1180,2,no
            8.2,28,1410,2,yes
            8.3,27,1020,1,yes
            7.6,23,4700,1,yes
        """
   
        def mocked_requests_get(*args, **kwargs):

            if args[0] == 'https://www.openml.org/api/v1/json/data/42693':
                return MockResponse(200, "", json.dumps(data_set_description).encode().splitlines())

            if args[0] == 'https://www.openml.org/api/v1/json/data/features/42693':
                return MockResponse(200, "", json.dumps(data_set_features).encode().splitlines())

            if args[0] == 'https://www.openml.org/api/v1/json/task/list/data_id/42693':
                return MockResponse(200, "", json.dumps(data_set_tasks).encode().splitlines())

            if args[0] == 'https://www.openml.org/data/v1/download/22044555':
                return MockResponse(200, "", data_set_arff.encode().splitlines())

            return MockResponse(None, 404, [])

        with unittest.mock.patch.object(requests, 'get', side_effect=mocked_requests_get):
            feature_rows, label_col = list(zip(*OpenmlSource(42693).read()))

        self.assertEqual(len(feature_rows), 5)
        self.assertEqual(len(label_col), 5)

        self.assertEqual([8.1, 27, 1410, (1,0)], feature_rows[0])
        self.assertEqual([8.2, 29, 1180, (1,0)], feature_rows[1])
        self.assertEqual([8.2, 28, 1410, (1,0)], feature_rows[2])
        self.assertEqual([8.3, 27, 1020, (0,1)], feature_rows[3])
        self.assertEqual([7.6, 23, 4700, (0,1)], feature_rows[4])

        self.assertEqual((1,0), label_col[0])
        self.assertEqual((1,0), label_col[1])
        self.assertEqual((0,1), label_col[2])
        self.assertEqual((0,1), label_col[3])
        self.assertEqual((0,1), label_col[4])

        self.assertIn   ('openml_042693_descr', CobaContext.cacher)
        self.assertIn   ('openml_042693_feats', CobaContext.cacher)
        self.assertIn   ('openml_042693_arff' , CobaContext.cacher)
        self.assertNotIn('openml_042693_csv'  , CobaContext.cacher)

    def test_read_twice_http_request_put_once_cache_once(self):

        data_set_description = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
            }
        }

        data_set_features = {
            "data_features":{
                "feature":[
                    {"index":"0","name":"pH"          ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"1","name":"temperature" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"2","name":"conductivity","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"3","name":"coli"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"4","name":"play"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"}
                ]
            }
        }

        data_set_tasks = {
            "tasks":{
                "task":[
                    { "task_id":338754, "task_type_id":1, "status":"active", "input": [{"name":"target_feature", "value":"play"}]}, 
                    { "task_id":359909, "task_type_id":5, "status":"active" } 
                ]
            }
        }

        data_set_arff = """
            @relation weather
            
            @attribute pH real
            @attribute temperature real
            @attribute conductivity real
            @attribute coli {2, 1}
            @attribute play {no, yes}
            
            @data
            8.1,27,1410,2,no
            8.2,29,1180,2,no
            8.2,28,1410,2,yes
            8.3,27,1020,1,yes
            7.6,23,4700,1,yes
        """
   
        request_dict = {
            'https://www.openml.org/api/v1/json/data/42693'             : MockResponse(200, "", json.dumps(data_set_description).encode().splitlines()),
            'https://www.openml.org/api/v1/json/data/features/42693'    : MockResponse(200, "", json.dumps(data_set_features).encode().splitlines()),
            'https://www.openml.org/data/v1/download/22044555'          : MockResponse(200, "", data_set_arff.encode().splitlines()),
            'https://www.openml.org/api/v1/json/task/list/data_id/42693': MockResponse(200, "", json.dumps(data_set_tasks).encode().splitlines())
        }

        def mocked_requests_get(*args, **kwargs):
            return request_dict.pop(args[0])

        CobaContext.cacher = PutOnceCacher()

        with unittest.mock.patch.object(requests, 'get', side_effect=mocked_requests_get):
            for _ in range(2):
                feature_rows, label_col = list(zip(*OpenmlSource(42693).read()))

                self.assertEqual(len(feature_rows), 5)
                self.assertEqual(len(label_col), 5)

                self.assertEqual([8.1, 27, 1410, (1,0)], feature_rows[0])
                self.assertEqual([8.2, 29, 1180, (1,0)], feature_rows[1])
                self.assertEqual([8.2, 28, 1410, (1,0)], feature_rows[2])
                self.assertEqual([8.3, 27, 1020, (0,1)], feature_rows[3])
                self.assertEqual([7.6, 23, 4700, (0,1)], feature_rows[4])

                self.assertEqual((1,0), label_col[0])
                self.assertEqual((1,0), label_col[1])
                self.assertEqual((0,1), label_col[2])
                self.assertEqual((0,1), label_col[3])
                self.assertEqual((0,1), label_col[4])

                self.assertIn('openml_042693_descr', CobaContext.cacher)
                self.assertIn('openml_042693_feats', CobaContext.cacher)
                self.assertIn('openml_042693_arff' , CobaContext.cacher)

    def test_status_code_412_request_api_key(self):
        with unittest.mock.patch.object(requests, 'get', return_value=MockResponse(412, "please provide api key", [])):
            with self.assertRaises(CobaException) as e:
                feature_rows, label_col = list(zip(*OpenmlSource(42693).read()))

    def test_status_code_412_rejected_api_key(self):
        with unittest.mock.patch.object(requests, 'get', return_value=MockResponse(412, "authentication failed", [])):
            with self.assertRaises(CobaException) as e:
                feature_rows, label_col = list(zip(*OpenmlSource(42693).read()))

    def test_status_code_404(self):
        with unittest.mock.patch.object(requests, 'get', return_value=MockResponse(404, "authentication failed", [])):
            with self.assertRaises(CobaException) as e:
                feature_rows, label_col = list(zip(*OpenmlSource(42693).read()))

    def test_status_code_405(self):
        with unittest.mock.patch.object(requests, 'get', return_value=MockResponse(405, "authentication failed", [])):
            with self.assertRaises(CobaException) as e:
                feature_rows, label_col = list(zip(*OpenmlSource(42693).read()))

class OpenmlSimulation_Tests(unittest.TestCase):

    @unittest.skip("While it is nice to test this functionality, in practice it is fairly slow.")
    def test_simple_openml_source_classification_online(self) -> None:
        #this test requires interet acess to download the data

        CobaContext.api_keys = {'openml':None}
        CobaContext.cacher   = NullCacher()
        CobaContext.logger   = NullLogger()

        interactions = list(OpenmlSimulation(1116).read())

        self.assertEqual(len(interactions), 6598)

        for rnd in interactions:

            hash(rnd.context)    #make sure these are hashable
            hash(rnd.actions[0]) #make sure these are hashable
            hash(rnd.actions[1]) #make sure these are hashable

            self.assertEqual(len(cast(Tuple,rnd.context)), 167)
            self.assertIn((0,1), rnd.actions)
            self.assertIn((1,0), rnd.actions)
            self.assertEqual(len(rnd.actions),2)
            self.assertIn(1, rnd.kwargs["rewards"])
            self.assertIn(0, rnd.kwargs["rewards"])

    def test_simple_openml_source_classification_offline(self) -> None:

        CobaContext.api_keys = {'openml':None}
        CobaContext.cacher   = MemoryCacher()
        CobaContext.logger   = NullLogger()

        data_set_description = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
            }
        }

        data_set_features = {
            "data_features":{
                "feature":[
                    {"index":"0","name":"pH"          ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"1","name":"temperature" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"2","name":"conductivity","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"3","name":"coli"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"4","name":"play"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"}
                ]
            }
        }

        data_set = """
            @relation weather
            
            @attribute pH real
            @attribute temperature real
            @attribute conductivity real
            @attribute coli {2, 1}
            @attribute play {no, yes}
            
            @data
            8.1,27,1410,2,no
            8.2,29,1180,2,no
            8.3,27,1020,1,yes
        """

        data_set_tasks = {
            "tasks":{
                "task":[
                    { "task_id":338754, "task_type_id":1, "status":"active", "input": [{"name":"target_feature", "value":"coli"}]}, 
                    { "task_id":359909, "task_type_id":5, "status":"active" } 
                ]
            }
        }

        CobaContext.cacher.put('openml_042693_descr', json.dumps(data_set_description).encode().splitlines())
        CobaContext.cacher.put('openml_042693_feats', json.dumps(data_set_features).encode().splitlines())
        CobaContext.cacher.put('openml_042693_arff' , data_set.encode().splitlines() )
        CobaContext.cacher.put('openml_042693_tasks', json.dumps(data_set_tasks).encode().splitlines() )

        interactions = list(OpenmlSimulation(42693, cat_as_str=True, label_type="C").read())

        self.assertEqual(len(interactions), 3)

        for rnd in interactions:

            hash(rnd.context)    #make sure these are hashable
            hash(rnd.actions[0]) #make sure these are hashable
            hash(rnd.actions[1]) #make sure these are hashable

        self.assertEqual((8.1,27,1410,'no'), interactions[0].context)
        self.assertEqual((8.2,29,1180,'no'), interactions[1].context)
        self.assertEqual((8.3,27,1020,'yes'), interactions[2].context)
        
        self.assertEqual(["1","2"], interactions[0].actions)
        self.assertEqual(["1","2"], interactions[1].actions)
        self.assertEqual(["1","2"], interactions[2].actions)

        self.assertEqual([0,1], interactions[0].kwargs["rewards"])
        self.assertEqual([0,1], interactions[1].kwargs["rewards"])
        self.assertEqual([1,0], interactions[2].kwargs["rewards"])

    def test_simple_openml_source_regression_offline(self) -> None:

        CobaContext.api_keys = {'openml':None}
        CobaContext.cacher   = MemoryCacher()
        CobaContext.logger   = NullLogger()

        data_set_description = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
            }
        }

        data_set_features = {
            "data_features":{
                "feature":[
                    {"index":"0","name":"pH"          ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"1","name":"temperature" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"2","name":"conductivity","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"3","name":"coli"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"4","name":"play"        ,"data_type":"nominal","is_ignore":"false","is_row_identifier":"false"}
                ]
            }
        }

        data_set = """
            @relation weather
            
            @attribute pH real
            @attribute temperature real
            @attribute conductivity real
            @attribute coli {2, 1}
            @attribute play {no, yes}
            
            @data
            8.1,27,1410,2,no
            8.2,29,1180,2,no
            8.3,27,1020,1,yes
        """

        data_set_tasks = {
            "tasks":{
                "task":[
                    { "task_id":338754, "task_type_id":2, "status":"active", "input": [{"name":"target_feature", "value":"pH"}]}, 
                    { "task_id":359909, "task_type_id":5, "status":"active" } 
                ]
            }
        }

        CobaContext.cacher.put('openml_042693_descr', json.dumps(data_set_description).encode().splitlines())
        CobaContext.cacher.put('openml_042693_feats', json.dumps(data_set_features).encode().splitlines())
        CobaContext.cacher.put('openml_042693_arff' , data_set.encode().splitlines() )
        CobaContext.cacher.put('openml_042693_tasks', json.dumps(data_set_tasks).encode().splitlines() )

        interactions = list(OpenmlSimulation(42693, label_type="R").read())

        self.assertEqual(len(interactions), 3)

        for rnd in interactions:

            hash(rnd.context)    #make sure these are hashable
            hash(rnd.actions[0]) #make sure these are hashable
            hash(rnd.actions[1]) #make sure these are hashable

        self.assertEqual((27,1410,(1,0),(1,0)), interactions[0].context)
        self.assertEqual((29,1180,(1,0),(1,0)), interactions[1].context)
        self.assertEqual((27,1020,(0,1),(0,1)), interactions[2].context)
        
        self.assertEqual([(0,0,1), (1,0,0), (0,1,0)], interactions[0].actions)
        self.assertEqual([(0,0,1), (1,0,0), (0,1,0)], interactions[1].actions)
        self.assertEqual([(0,0,1), (1,0,0), (0,1,0)], interactions[2].actions)

        self.assertEqual([0 ,  1, .5], [ round(r,2) for r in interactions[0].kwargs["rewards"]])
        self.assertEqual([.5, .5,  1], [ round(r,2) for r in interactions[1].kwargs["rewards"]])
        self.assertEqual([1 ,  0, .5], [ round(r,2) for r in interactions[2].kwargs["rewards"]])

    def test_str(self):
        self.assertEqual('Openml(id=150, cat_as_str=False)', str(OpenmlSimulation(150)))

if __name__ == '__main__':
    unittest.main()
