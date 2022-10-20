import requests
import unittest.mock
import unittest
import json

from math import isnan
from threading import Semaphore, Event, Thread
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
        CobaContext.store    = {}

    def test_already_cached_values_are_not_cached_again(self):

        CobaContext.cacher = PutOnceCacher()

        data = {
            "data_set_description":{
                "id":"42693",
                "file_id":"22044555",
                "status":"active",
                "default_target_attribute":"play"
            }
        }

        feat = {
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

        arff = """
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

        CobaContext.cacher.put('openml_042693_data', json.dumps(data).splitlines())
        CobaContext.cacher.put('openml_042693_feat', json.dumps(feat).splitlines())
        CobaContext.cacher.put('openml_042693_arff', arff.splitlines() )

        features,labels = zip(*[ (r.feats,r.label) for r in OpenmlSource(data_id=42693).read()])

        self.assertEqual(len(features), 5)
        self.assertEqual(len(labels  ), 5)

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

    def test_data_deactivated(self):

        data = {
            "data_set_description":{
                "id":"42693",
                "file_id":"22044555",
                "status":"deactivated",
                "default_target_attribute": "play"
            }
        }


        CobaContext.cacher.put('openml_042693_data', json.dumps(data).splitlines())

        with self.assertRaises(CobaException) as e:
            feature_rows, label_col = OpenmlSource(data_id=42693).read()

        self.assertTrue("has been deactivated" in str(e.exception))

    def test_missing_values(self):

        data = {
            "data_set_description":{
                "id":"42693",
                "file_id":"22044555",
                "status":"active",
                "default_target_attribute":"play"
            }
        }

        feat = {
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

        arff = """
            @relation weather

            @attribute pH real
            @attribute temperature real
            @attribute conductivity real
            @attribute coli {2, 1}
            @attribute play {no, yes}

            @data
            ?,27,1410,2,no
            8.2,29,1180,2,no
            8.2,?,1410,2,yes
            8.3,27,1020,1,yes
            7.6,23,4700,1,yes
        """

        CobaContext.cacher.put('openml_042693_data', json.dumps(data).splitlines())
        CobaContext.cacher.put('openml_042693_feat', json.dumps(feat).splitlines())
        CobaContext.cacher.put('openml_042693_arff', arff.splitlines() )

        features,labels = zip(*[(r.feats,r.label) for r in OpenmlSource(data_id=42693).read()])

        self.assertEqual(len(features), 3)
        self.assertEqual(len(labels  ), 3)

        self.assertEqual([8.2, 29, 1180, (1,0)], features[0])
        self.assertEqual([8.3, 27, 1020, (0,1)], features[1])
        self.assertEqual([7.6, 23, 4700, (0,1)], features[2])

        self.assertEqual((1,0), labels[0])
        self.assertEqual((0,1), labels[1])
        self.assertEqual((0,1), labels[2])

    def test_skip_structure_openmlreader(self):

        data = {
            "data_set_description":{
                "id":"42693",
                "file_id":"22044555",
                "status":"active",
                "default_target_attribute":"play"
            }
        }

        feat = {
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

        arff = """
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

        CobaContext.cacher.put('openml_042693_data', json.dumps(data).splitlines())
        CobaContext.cacher.put('openml_042693_feat', json.dumps(feat).splitlines())
        CobaContext.cacher.put('openml_042693_arff', arff.splitlines() )

        raw = list(OpenmlSource(data_id=42693, drop_missing=False, skip_structure=True).read())
        
        self.assertNotEqual(raw[0][0],raw[0][0])
        
        self.assertEqual([27, 1410, (1,0),(1,0)], list(raw[0])[1:])
        self.assertEqual([8.2, 29, 1180, (1,0),(1,0)], raw[1])
        
        self.assertEqual(8.2, raw[2][0])
        self.assertNotEqual(raw[2][1],raw[2][1])
        self.assertEqual([1410, (1,0),(0,1)], list(raw[2])[2:])
        
        self.assertEqual([8.3, 27, 1020, (0,1),(0,1)], raw[3])
        self.assertEqual([7.6, 23, 4700, (0,1),(0,1)], raw[4])

    def test_cat_as_str(self):

        data = {
            "data_set_description":{
                "id":"42693",
                "file_id":"22044555",
                "status":"active",
                "default_target_attribute":"play"
            }
        }

        feat = {
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

        arff = """
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

        CobaContext.cacher.put('openml_042693_data', json.dumps(data).splitlines())
        CobaContext.cacher.put('openml_042693_feat', json.dumps(feat).splitlines())
        CobaContext.cacher.put('openml_042693_arff', arff.splitlines() )

        features,labels = zip(*[ (r.feats,r.label) for r in OpenmlSource(data_id=42693,cat_as_str=True).read()])

        self.assertEqual(len(features), 5)
        self.assertEqual(len(labels  ), 5)

        self.assertEqual([8.1, 27, 1410, '2'], features[0])
        self.assertEqual([8.2, 29, 1180, '2'], features[1])
        self.assertEqual([8.2, 28, 1410, '2'], features[2])
        self.assertEqual([8.3, 27, 1020, '1'], features[3])
        self.assertEqual([7.6, 23, 4700, '1'], features[4])

        self.assertEqual('no' , labels[0])
        self.assertEqual('no' , labels[1])
        self.assertEqual('yes', labels[2])
        self.assertEqual('yes', labels[3])
        self.assertEqual('yes', labels[4])

    def test_data_classification(self):

        data = {
            "data_set_description":{
                "id":"42693",
                "file_id":"22044555",
                "status":"active",
                "default_target_attribute":"play"
            }
        }

        feat = {
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

        arff = """
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

        CobaContext.cacher.put('openml_042693_data', json.dumps(data).splitlines())
        CobaContext.cacher.put('openml_042693_feat', json.dumps(feat).splitlines())
        CobaContext.cacher.put('openml_042693_arff' , arff.splitlines() )

        features,labels = zip(*[ (r.feats,r.label) for r in OpenmlSource(data_id=42693).read()])

        self.assertEqual(len(features), 5)
        self.assertEqual(len(labels  ), 5)

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

    def test_data_regression(self):

        data = {
            "data_set_description":{
                "id":"42693",
                "file_id":"22044555",
                "status":"active",
                "default_target_attribute":"pH"
            }
        }

        feat = {
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

        arff = """
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

        CobaContext.cacher.put('openml_042693_data', json.dumps(data).splitlines())
        CobaContext.cacher.put('openml_042693_feat', json.dumps(feat).splitlines())
        CobaContext.cacher.put('openml_042693_arff' , arff.splitlines() )

        features,labels = zip(*[ (r.feats,r.label) for r in OpenmlSource(data_id=42693).read()])

        self.assertEqual(len(features), 5)
        self.assertEqual(len(labels  ), 5)

        self.assertEqual([27, 1410, (1,0), (1,0)], features[0])
        self.assertEqual([29, 1180, (1,0), (1,0)], features[1])
        self.assertEqual([28, 1410, (1,0), (0,1)], features[2])
        self.assertEqual([27, 1020, (0,1), (0,1)], features[3])
        self.assertEqual([23, 4700, (0,1), (0,1)], features[4])

        self.assertEqual(8.1, labels[0])
        self.assertEqual(8.2, labels[1])
        self.assertEqual(8.2, labels[2])
        self.assertEqual(8.3, labels[3])
        self.assertEqual(7.6, labels[4])

    def test_no_default_target_attribute(self):

        data = {
            "data_set_description":{
                "id":"42693",
                "file_id":"22044555",
                "status":"active",
            }
        }

        CobaContext.cacher.put('openml_042693_data', json.dumps(data).splitlines())

        with self.assertRaises(Exception) as e:
            feature_rows, label_col = OpenmlSource(data_id=42693).read()

        self.assertTrue("We were unable to find" in str(e.exception))

    def test_targeting_ignored_col(self):

        data = {
            "data_set_description":{
                "id":"42693",
                "file_id":"22044555",
                "status":"active",
                "default_target_attribute":"coli"
            }
        }

        feat = {
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

        arff = """
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

        CobaContext.cacher.put('openml_042693_data', json.dumps(data).splitlines())
        CobaContext.cacher.put('openml_042693_feat', json.dumps(feat).splitlines())
        CobaContext.cacher.put('openml_042693_arff', arff.splitlines() )

        features,labels = zip(*[ (r.feats,r.label) for r in OpenmlSource(data_id=42693).read()])

        self.assertEqual(len(features), 5)
        self.assertEqual(len(labels  ), 5)

        self.assertEqual([8.1, 27, 1410, (1,0)], features[0])
        self.assertEqual([8.2, 29, 1180, (1,0)], features[1])
        self.assertEqual([8.2, 28, 1410, (0,1)], features[2])
        self.assertEqual([8.3, 27, 1020, (0,1)], features[3])
        self.assertEqual([7.6, 23, 4700, (0,1)], features[4])

        self.assertEqual((1,0), labels[0])
        self.assertEqual((1,0), labels[1])
        self.assertEqual((1,0), labels[2])
        self.assertEqual((0,1), labels[3])
        self.assertEqual((0,1), labels[4])

    def test_sparse_classification_target(self):

        data = {
            "data_set_description":{
                "id":"1594",
                "file_id":"1595696",
                "status":"active",
                "default_target_attribute":"class"
            }
        }

        feat = {
            "data_features":{
                "feature":[
                    {"index":"0","name":"0","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"1","name":"1","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"2","name":"2","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"3","name":"3","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"4","name":"4","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"5","name":"5","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"6","name":"6","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"7","name":"7","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"8","name":"8","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"9","name":"9","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"10","name":"class","data_type":"nominal","nominal_value":["A","B","C","D"],"is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"}
                ]
            }
        }

        arff = """
            @relation news20

            @attribute 0 numeric
            @attribute 1 numeric
            @attribute 2 numeric
            @attribute 3 numeric
            @attribute 4 numeric
            @attribute 5 numeric
            @attribute 6 numeric
            @attribute 7 numeric
            @attribute 8 numeric
            @attribute 9 numeric
            @attribute class {B, C, D}

            @data
            {0 2,1 3}
            {2 1,3 1,4 1,6 1,8 1,10 B}
            {0 3,1 1,2 1,3 9,4 1,5 1,6 1,10 C}
            {0 1,3 1,6 1,7 1,8 1,9 2,10 D}
        """

        CobaContext.cacher.put('openml_001594_data', json.dumps(data).splitlines())
        CobaContext.cacher.put('openml_001594_feat', json.dumps(feat).splitlines())
        CobaContext.cacher.put('openml_001594_arff', arff.splitlines())

        features,labels = zip(*[ (r.feats,r.label) for r in OpenmlSource(data_id=1594).read()])

        self.assertEqual(len(features), 4)
        self.assertEqual(len(labels  ), 4)

        self.assertEqual(dict(zip(map(str,(0,1))          , (2,3)          )), features[0])
        self.assertEqual(dict(zip(map(str,(2,3,4,6,8))    , (1,1,1,1,1)    )), features[1])
        self.assertEqual(dict(zip(map(str,(0,1,2,3,4,5,6)), (3,1,1,9,1,1,1))), features[2])
        self.assertEqual(dict(zip(map(str,(0,3,6,7,8,9))  , (1,1,1,1,1,2)  )), features[3])

        self.assertEqual((1,0,0,0), labels[0])
        self.assertEqual((0,1,0,0), labels[1])
        self.assertEqual((0,0,1,0), labels[2])
        self.assertEqual((0,0,0,1), labels[3])

    def test_task(self):

        task = {
            "task":{
                "task_type_id":"1",
                "input":[
                    {"name":"source_data","data_set":{"data_set_id":"1594","target_feature":"class"}}
                ]
            }
        }

        data = {
            "data_set_description":{
                "id":"1594",
                "file_id":"1595696",
                "status":"active",
            }
        }

        feat = {
            "data_features":{
                "feature":[
                    {"index":"0","name":"0","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"1","name":"1","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"2","name":"2","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"3","name":"3","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"4","name":"4","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"5","name":"5","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"6","name":"5","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"7","name":"7","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"8","name":"8","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"9","name":"9","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"10","name":"class","data_type":"nominal","nominal_value":["A","B","C","D"],"is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"}
                ]
            }
        }

        arff = """
            @relation news20

            @attribute 0 numeric
            @attribute 1 numeric
            @attribute 2 numeric
            @attribute 3 numeric
            @attribute 4 numeric
            @attribute 5 numeric
            @attribute 6 numeric
            @attribute 7 numeric
            @attribute 8 numeric
            @attribute 9 numeric
            @attribute class {B, C, D}

            @data
            {0 2,1 3}
            {2 1,3 1,4 1,6 1,8 1,10 B}
            {0 3,1 1,2 1,3 9,4 1,5 1,6 1,10 C}
            {0 1,3 1,6 1,7 1,8 1,9 2,10 D}
        """

        CobaContext.cacher.put('openml_001111_task', json.dumps(task).splitlines())
        CobaContext.cacher.put('openml_001594_data', json.dumps(data).splitlines())
        CobaContext.cacher.put('openml_001594_feat', json.dumps(feat).splitlines())
        CobaContext.cacher.put('openml_001594_arff', arff.splitlines())

        features,labels = zip(*[ (r.feats,r.label) for r in OpenmlSource(task_id=1111).read()])

        self.assertEqual(len(features), 4)
        self.assertEqual(len(labels  ), 4)

        self.assertEqual(dict(zip( map(str,(0,1))          , (2,3)          )), features[0])
        self.assertEqual(dict(zip( map(str,(2,3,4,6,8))    , (1,1,1,1,1)    )), features[1])
        self.assertEqual(dict(zip( map(str,(0,1,2,3,4,5,6)), (3,1,1,9,1,1,1))), features[2])
        self.assertEqual(dict(zip( map(str,(0,3,6,7,8,9))  , (1,1,1,1,1,2)  )), features[3])

        self.assertEqual((1,0,0,0), labels[0])
        self.assertEqual((0,1,0,0), labels[1])
        self.assertEqual((0,0,1,0), labels[2])
        self.assertEqual((0,0,0,1), labels[3])

    def test_task_without_source_data(self):

        task = {
            "task":{
                "task_type_id":"1",
                "input":[ ]
            }
        }

        data = {
            "data_set_description":{
                "id":"1594",
                "file_id":"1595696",
                "status":"active",
            }
        }

        feat = {
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

        arff = """
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

        CobaContext.cacher.put('openml_001111_task', json.dumps(task).splitlines())
        CobaContext.cacher.put('openml_001594_data', json.dumps(data).splitlines())
        CobaContext.cacher.put('openml_001594_feat', json.dumps(feat).splitlines())
        CobaContext.cacher.put('openml_001594_arff', arff.splitlines())

        with self.assertRaises(CobaException) as e:
            feature_rows, label_col = list(zip(*OpenmlSource(task_id=1111).read()))

        self.assertIn("does not appear to have an associated", str(e.exception))

    def test_task_classification_with_numeric_target(self):

        task = {
            "task":{
                "task_type_id":"1",
                "input":[
                    {"name":"source_data","data_set":{"data_set_id":"1594","target_feature":"class"}}
                ]
            }
        }

        data = {
            "data_set_description":{
                "id":"1594",
                "file_id":"1595696",
                "status":"active",
            }
        }

        feat = {
            "data_features":{
                "feature":[
                    {"index":"0","name":"0" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"1","name":"1" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"2","name":"2" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"3","name":"3" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"4","name":"4" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"5","name":"5" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"6","name":"6" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"7","name":"7" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"8","name":"8" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"9","name":"9","data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                    {"index":"9","name":"class" ,"data_type":"numeric","is_ignore":"false","is_row_identifier":"false"},
                ]
            }
        }

        arff = """
            @relation test

            @attribute 0 numeric
            @attribute 1 numeric
            @attribute 2 numeric
            @attribute 3 numeric
            @attribute 4 numeric
            @attribute 5 numeric
            @attribute 6 numeric
            @attribute 7 numeric
            @attribute 8 numeric
            @attribute 9 numeric
            @attribute class numeric

            @data
            {0 2,1 3}
            {2 1,3 1,4 1,6 1,8 1,10 1}
            {0 3,1 1,2 1,3 9,4 1,5 1,6 1,10 2}
            {0 1,3 1,6 1,7 1,8 1,9 2,10 3}
        """

        CobaContext.cacher.put('openml_001111_task', json.dumps(task).splitlines())
        CobaContext.cacher.put('openml_001594_data', json.dumps(data).splitlines())
        CobaContext.cacher.put('openml_001594_feat', json.dumps(feat).splitlines())
        CobaContext.cacher.put('openml_001594_arff', arff.splitlines())
        
        features,labels = zip(*[ (r.feats,r.label) for r in OpenmlSource(task_id=1111).read()])

        self.assertEqual(len(features), 4)
        self.assertEqual(len(labels  ), 4)

        self.assertEqual(dict(zip(map(str,(0,1))          , (2,3)          )), features[0])
        self.assertEqual(dict(zip(map(str,(2,3,4,6,8))    , (1,1,1,1,1)    )), features[1])
        self.assertEqual(dict(zip(map(str,(0,1,2,3,4,5,6)), (3,1,1,9,1,1,1))), features[2])
        self.assertEqual(dict(zip(map(str,(0,3,6,7,8,9))  , (1,1,1,1,1,2)  )), features[3])

        self.assertEqual('0', labels[0])
        self.assertEqual('1', labels[1])
        self.assertEqual('2', labels[2])
        self.assertEqual('3', labels[3])

    def test_task_id_no_source(self):

        task_description = {
            "task":{
                "input":[ ]
            }
        }

        CobaContext.cacher.put('openml_001111_task', json.dumps(task_description).splitlines())

        with self.assertRaises(CobaException):
            list(OpenmlSource(task_id=1111).read())

    def test_cache_cleared_on_unexpected_exception(self):

        data = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
                "default_target_attribute":"play"
            }
        }

        feat = {
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

        CobaContext.cacher = ExceptionCacher('openml_042693_arff', Exception())

        CobaContext.cacher.put('openml_042693_data', json.dumps(data).splitlines())
        CobaContext.cacher.put('openml_042693_feat', json.dumps(feat).splitlines())
        CobaContext.cacher.put('openml_042693_arff', b"" )

        with self.assertRaises(Exception) as e:
            feature_rows, label_col = list(zip(*OpenmlSource(data_id=42693).read()))

        self.assertNotIn('openml_042693_data', CobaContext.cacher)
        self.assertNotIn('openml_042693_feat', CobaContext.cacher)
        self.assertNotIn('openml_042693_arff', CobaContext.cacher)

    def test_cache_not_cleared_on_keyboard_interrupt(self):

        data = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
                "default_target_attribute":"play"
            }
        }

        feat = {
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

        CobaContext.cacher = ExceptionCacher('openml_042693_arff', KeyboardInterrupt())

        CobaContext.cacher.put('openml_042693_data', json.dumps(data).splitlines())
        CobaContext.cacher.put('openml_042693_feat', json.dumps(feat).splitlines())
        CobaContext.cacher.put('openml_042693_arff', data_set.splitlines() )

        with self.assertRaises(KeyboardInterrupt) as e:
            feature_rows, label_col = list(zip(*OpenmlSource(data_id=42693).read()))

        self.assertIn('openml_042693_data', CobaContext.cacher)
        self.assertIn('openml_042693_feat', CobaContext.cacher)
        self.assertIn('openml_042693_arff', CobaContext.cacher)

    def test_cache_cleared_on_cache_coba_exception(self):

        data = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
                "default_target_attribute":"play"
            }
        }

        feat = {
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

        arff = """
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

        CobaContext.cacher = ExceptionCacher('openml_042693_arff', CobaException())

        CobaContext.cacher.put('openml_042693_data', json.dumps(data).splitlines())
        CobaContext.cacher.put('openml_042693_feat', json.dumps(feat).splitlines())
        CobaContext.cacher.put('openml_042693_arff', arff.splitlines() )

        with self.assertRaises(Exception) as e:
            feature_rows, label_col = list(zip(*OpenmlSource(data_id=42693).read()))

        self.assertNotIn('openml_042693_data', CobaContext.cacher)
        self.assertNotIn('openml_042693_feat', CobaContext.cacher)
        self.assertNotIn('openml_042693_arff', CobaContext.cacher)

    def test_read_twice_http_request_put_once_cache_once(self):

        data = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
                "default_target_attribute":"play"
            }
        }

        feat = {
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

        arff = """
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
            'https://openml.org/api/v1/json/data/42693'         : MockResponse(200, "", json.dumps(data).splitlines()),
            'https://openml.org/api/v1/json/data/features/42693': MockResponse(200, "", json.dumps(feat).splitlines()),
            'https://openml.org/data/v1/download/22044555'      : MockResponse(200, "", arff.splitlines()),
        }

        def mocked_requests_get(*args, **kwargs):
            return request_dict.pop(args[0])

        CobaContext.cacher = PutOnceCacher()

        with unittest.mock.patch.object(requests, 'get', side_effect=mocked_requests_get):
            for _ in range(2):
                features,labels = zip(*[ (r.feats,r.label) for r in OpenmlSource(data_id=42693).read()])

                self.assertEqual(len(features), 5)
                self.assertEqual(len(labels  ), 5)

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

                self.assertIn('openml_042693_data', CobaContext.cacher)
                self.assertIn('openml_042693_feat', CobaContext.cacher)
                self.assertIn('openml_042693_arff', CobaContext.cacher)

    def test_semaphore_locked_and_released(self):

        semaphore = Semaphore(1)
        block_1   = Event()
        block_2   = Event()

        task = {
            "task":{
                "task_type_id":"1",
                "input":[
                    {"name":"source_data","data_set":{"data_set_id":"42693","target_feature":"play"}}
                ]
            }
        }

        data = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
                "default_target_attribute":"play"
            }
        }

        feat = {
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

        arff = """
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
            'https://openml.org/api/v1/json/task/123'           : MockResponse(200, "", json.dumps(task).splitlines()),
            'https://openml.org/api/v1/json/data/42693'         : MockResponse(200, "", json.dumps(data).splitlines()),
            'https://openml.org/api/v1/json/data/features/42693': MockResponse(200, "", json.dumps(feat).splitlines()),
            'https://openml.org/data/v1/download/22044555'      : MockResponse(200, "", arff.splitlines()),
        }

        def mocked_requests_get(*args, **kwargs):
            block_2.set()
            block_1.wait()
            return request_dict.pop(args[0])

        CobaContext.store['openml_semaphore'] = semaphore
        CobaContext.cacher = PutOnceCacher()

        with unittest.mock.patch.object(requests, 'get', side_effect=mocked_requests_get):
            def thread_1():
                features,labels = zip(*[ (r.feats,r.label) for r in OpenmlSource(task_id=123).read()])

            t1 = Thread(None, thread_1)
            t1.start()

            #make sure t1 has time to acuire lock
            block_2.wait()

            #we shouldn't be able to acquire if openml correctly locked 
            self.assertFalse(semaphore.acquire(blocking=False))
            block_1.set() # now we release t1 to finish
            t1.join()

            #now we can acquire because openml should release when done
            self.assertTrue(semaphore.acquire(blocking=False))

            #this should complete despite us acquiring above 
            #because it doesn't lock since everything is cached
            features,labels = zip(*[ (r.feats,r.label) for r in OpenmlSource(task_id=123).read()])

            self.assertEqual(len(features), 5)
            self.assertEqual(len(labels  ), 5)

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

            self.assertIn('openml_042693_data', CobaContext.cacher)
            self.assertIn('openml_042693_feat', CobaContext.cacher)
            self.assertIn('openml_042693_arff', CobaContext.cacher)

    def test_status_code_412_request_api_key(self):
        with unittest.mock.patch.object(requests, 'get', return_value=MockResponse(412, "please provide api key", [])):
            with self.assertRaises(CobaException) as e:
                feature_rows, label_col = list(zip(*OpenmlSource(data_id=42693).read()))

    def test_status_code_412_rejected_api_key(self):
        with unittest.mock.patch.object(requests, 'get', return_value=MockResponse(412, "authentication failed", [])):
            with self.assertRaises(CobaException) as e:
                feature_rows, label_col = list(zip(*OpenmlSource(data_id=42693).read()))

    def test_status_code_404(self):
        with unittest.mock.patch.object(requests, 'get', return_value=MockResponse(404, "authentication failed", [])):
            with self.assertRaises(CobaException) as e:
                feature_rows, label_col = list(zip(*OpenmlSource(data_id=42693).read()))

    def test_status_code_405(self):
        with unittest.mock.patch.object(requests, 'get', return_value=MockResponse(405, "authentication failed", [])):
            with self.assertRaises(CobaException) as e:
                feature_rows, label_col = list(zip(*OpenmlSource(data_id=42693).read()))

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

        data = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
                "default_target_attribute": "coli"
            }
        }

        feat = {
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

        arff = """
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

        CobaContext.cacher.put('openml_042693_data', json.dumps(data).splitlines())
        CobaContext.cacher.put('openml_042693_feat', json.dumps(feat).splitlines())
        CobaContext.cacher.put('openml_042693_arff', arff.splitlines() )

        interactions = list(OpenmlSimulation(data_id=42693, cat_as_str=True).read())

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

        self.assertEqual([0,1], interactions[0].rewards)
        self.assertEqual([0,1], interactions[1].rewards)
        self.assertEqual([1,0], interactions[2].rewards)

    def test_simple_openml_source_regression_offline(self) -> None:

        CobaContext.api_keys = {'openml':None}
        CobaContext.cacher   = MemoryCacher()
        CobaContext.logger   = NullLogger()

        data = {
            "data_set_description":{
                "id":"42693",
                "name":"testdata",
                "version":"2",
                "format":"ARFF",
                "licence":"CC0",
                "file_id":"22044555",
                "visibility":"public",
                "status":"active",
                "default_target_attribute":"pH"
            }
        }

        feat = {
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

        arff = """
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

        CobaContext.cacher.put('openml_042693_data', json.dumps(data).splitlines())
        CobaContext.cacher.put('openml_042693_feat', json.dumps(feat).splitlines())
        CobaContext.cacher.put('openml_042693_arff', arff.splitlines() )

        interactions = list(OpenmlSimulation(data_id=42693).read())

        self.assertEqual(len(interactions), 3)

        for rnd in interactions:

            hash(rnd.context)    #make sure these are hashable
            hash(rnd.actions[0]) #make sure these are hashable
            hash(rnd.actions[1]) #make sure these are hashable

        self.assertEqual((27,1410,(1,0),(1,0)), interactions[0].context)
        self.assertEqual((29,1180,(1,0),(1,0)), interactions[1].context)
        self.assertEqual((27,1020,(0,1),(0,1)), interactions[2].context)

        self.assertEqual([(1,0,0), (0,1,0), (0,0,1)], interactions[0].actions)
        self.assertEqual([(1,0,0), (0,1,0), (0,0,1)], interactions[1].actions)
        self.assertEqual([(1,0,0), (0,1,0), (0,0,1)], interactions[2].actions)

        self.assertEqual([ 1, .5,  0], [ round(r,2) for r in interactions[0].rewards])
        self.assertEqual([.5,  1, .5], [ round(r,2) for r in interactions[1].rewards])
        self.assertEqual([ 0, .5,  1], [ round(r,2) for r in interactions[2].rewards])

    def test_str(self):
        self.assertEqual('Openml(data=150)', str(OpenmlSimulation(150)))
        self.assertEqual('Openml(task=150)', str(OpenmlSimulation(task_id=150)))

if __name__ == '__main__':
    unittest.main()
