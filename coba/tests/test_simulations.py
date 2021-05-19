import unittest

import timeit

from itertools import repeat
from typing import List, Sequence, Tuple, cast, Optional

from coba.utilities import CobaConfig, NoneCacher, NoneLogger, MemoryCacher
from coba.simulations import (
    Key, Action, Context, Interaction, ClassificationSimulation, MemoryReward, MemorySimulation, 
    LambdaSimulation, OpenmlSimulation, Shuffle, Take, Batch, PCA, Sort, OpenmlSource
)

CobaConfig.Logger = NoneLogger()

def _choices(interaction: Interaction) -> Sequence[Tuple[Key, Optional[Context], Action]]:
    return [  (interaction.key, interaction.context, a) for a in interaction.actions]

class PutOnceCacher(MemoryCacher):

    def put(self, key, value) -> None:

        if key in self: raise Exception("Writing data again without reason.")
        return super().put(key, value)

class OpenmlSource_Tests(unittest.TestCase):
    
    def test_put_once_cache(self):

        CobaConfig.Api_Keys['openml'] = None
        CobaConfig.Cacher = PutOnceCacher()

        #data description query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/data/42693', b'{"data_set_description":{"id":"42693","name":"testdata","version":"2","description":"this is test data","format":"ARFF","upload_date":"2020-10-01T20:47:23","licence":"CC0","url":"https:\\/\\/www.openml.org\\/data\\/v1\\/download\\/22044555\\/testdata.arff","file_id":"22044555","visibility":"public","status":"active","processing_date":"2020-10-01 20:48:03","md5_checksum":"6656a444676c309dd8143aa58aa796ad"}}')
        #data types query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/data/features/42693', b'{"data_features":{"feature":[{"index":"0","name":"pH","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"1","name":"temperature","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"2","name":"conductivity","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"3","name":"coli","data_type":"nominal","nominal_value":[1,2],"is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"4","name":"play","data_type":"nominal","nominal_value":["no","yes"],"is_target":"true","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"}]}}')
        #data content query
        CobaConfig.Cacher.put('http://www.openml.org/data/v1/get_csv/22044555', b'"pH","temperature","conductivity","coli","play"\n8.1,27,1410,2,no\r\n8.2,29,1180,2,no\r\n8.2,28,1410,2,yes\r\n8.3,27,1020,1,yes\r\n7.6,23,4700,1,yes\r\n\r\n')
        #trials query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/task/list/data_id/42693', b'{"tasks":{"task":[\n    { "task_id":338754,\n    "task_type_id":5,\n    "task_type":"Clustering",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n,  { "task_id":359909,\n    "task_type_id":5,\n    "task_type":"Clustering",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n  ]}\n}\n')

        feature_rows, label_col = OpenmlSource(42693).read()

        self.assertEqual(len(feature_rows), 5)
        self.assertEqual(len(label_col), 5)

        self.assertEqual((8.1, 27, 1410, 0), feature_rows[0])
        self.assertEqual((8.2, 29, 1180, 0), feature_rows[1])
        self.assertEqual((8.2, 28, 1410, 0), feature_rows[2])
        self.assertEqual((8.3, 27, 1020, 1), feature_rows[3])
        self.assertEqual((7.6, 23, 4700, 1), feature_rows[4])

        self.assertEqual((1,0), label_col[0])
        self.assertEqual((1,0), label_col[1])
        self.assertEqual((0,1), label_col[2])
        self.assertEqual((0,1), label_col[3])
        self.assertEqual((0,1), label_col[4])

    def test_csv_default_classification(self):

        CobaConfig.Api_Keys['openml'] = None
        CobaConfig.Cacher = MemoryCacher()

        #data description query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/data/42693', b'{"data_set_description":{"id":"42693","name":"testdata","version":"2","description":"this is test data","format":"ARFF","upload_date":"2020-10-01T20:47:23","licence":"CC0","url":"https:\\/\\/www.openml.org\\/data\\/v1\\/download\\/22044555\\/testdata.arff","file_id":"22044555","visibility":"public","status":"active","processing_date":"2020-10-01 20:48:03","md5_checksum":"6656a444676c309dd8143aa58aa796ad"}}')
        #data types query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/data/features/42693', b'{"data_features":{"feature":[{"index":"0","name":"pH","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"1","name":"temperature","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"2","name":"conductivity","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"3","name":"coli","data_type":"nominal","nominal_value":[1,2],"is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"4","name":"play","data_type":"nominal","nominal_value":["no","yes"],"is_target":"true","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"}]}}')
        #data content query
        CobaConfig.Cacher.put('http://www.openml.org/data/v1/get_csv/22044555', b'"pH","temperature","conductivity","coli","play"\n8.1,27,1410,2,no\r\n8.2,29,1180,2,no\r\n8.2,28,1410,2,yes\r\n8.3,27,1020,1,yes\r\n7.6,23,4700,1,yes\r\n\r\n')
        #trials query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/task/list/data_id/42693', b'{"tasks":{"task":[\n    { "task_id":338754,\n    "task_type_id":5,\n    "task_type":"Clustering",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n,  { "task_id":359909,\n    "task_type_id":5,\n    "task_type":"Clustering",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n  ]}\n}\n')

        feature_rows, label_col = OpenmlSource(42693).read()

        self.assertEqual(len(feature_rows), 5)
        self.assertEqual(len(label_col), 5)

        self.assertEqual((8.1, 27, 1410, 0), feature_rows[0])
        self.assertEqual((8.2, 29, 1180, 0), feature_rows[1])
        self.assertEqual((8.2, 28, 1410, 0), feature_rows[2])
        self.assertEqual((8.3, 27, 1020, 1), feature_rows[3])
        self.assertEqual((7.6, 23, 4700, 1), feature_rows[4])

        self.assertEqual((1,0), label_col[0])
        self.assertEqual((1,0), label_col[1])
        self.assertEqual((0,1), label_col[2])
        self.assertEqual((0,1), label_col[3])
        self.assertEqual((0,1), label_col[4])

    def test_csv_not_classification(self):

        CobaConfig.Api_Keys['openml'] = None
        CobaConfig.Cacher = MemoryCacher()

        #data description query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/data/42693', b'{"data_set_description":{"id":"42693","name":"testdata","version":"2","description":"this is test data","format":"ARFF","upload_date":"2020-10-01T20:47:23","licence":"CC0","url":"https:\\/\\/www.openml.org\\/data\\/v1\\/download\\/22044555\\/testdata.arff","file_id":"22044555","visibility":"public","status":"active","processing_date":"2020-10-01 20:48:03","md5_checksum":"6656a444676c309dd8143aa58aa796ad"}}')
        #data types query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/data/features/42693', b'{"data_features":{"feature":[{"index":"0","name":"pH","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"1","name":"temperature","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"2","name":"conductivity","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"3","name":"coli","data_type":"nominal","nominal_value":[1,2],"is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"4","name":"play","data_type":"numeric","nominal_value":["no","yes"],"is_target":"true","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"}]}}')
        #data content query
        CobaConfig.Cacher.put('http://www.openml.org/data/v1/get_csv/22044555', b'"pH","temperature","conductivity","coli","play"\n8.1,27,1410,2,1\r\n8.2,29,1180,2,2\r\n8.2,28,1410,2,3\r\n8.3,27,1020,1,4\r\n7.6,23,4700,1,5\r\n\r\n')
        #trials query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/task/list/data_id/42693', b'{"tasks":{"task":[\n    { "task_id":338754,\n    "task_type_id":5,\n    "task_type":"Clustering",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n,  { "task_id":359909,\n    "task_type_id":5,\n    "task_type":"Clustering",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n  ]}\n}\n')

        with self.assertRaises(Exception) as e:
            feature_rows, label_col = OpenmlSource(42693).read()

        self.assertTrue("does not appear" in str(e.exception))

    def test_csv_not_default_classification(self):

        CobaConfig.Api_Keys['openml'] = None
        CobaConfig.Cacher = MemoryCacher()

        #data description query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/data/42693', b'{"data_set_description":{"id":"42693","name":"testdata","version":"2","description":"this is test data","format":"ARFF","upload_date":"2020-10-01T20:47:23","licence":"CC0","url":"https:\\/\\/www.openml.org\\/data\\/v1\\/download\\/22044555\\/testdata.arff","file_id":"22044555","visibility":"public","status":"active","processing_date":"2020-10-01 20:48:03","md5_checksum":"6656a444676c309dd8143aa58aa796ad"}}')
        #data types query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/data/features/42693', b'{"data_features":{"feature":[{"index":"0","name":"pH","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"1","name":"temperature","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"2","name":"conductivity","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"3","name":"coli","data_type":"nominal","nominal_value":[1,2],"is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"4","name":"play","data_type":"numeric","is_target":"true","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"}]}}')
        #data content query
        CobaConfig.Cacher.put('http://www.openml.org/data/v1/get_csv/22044555', b'"pH","temperature","conductivity","coli","play"\n8.1,27,1410,2,1\r\n8.2,29,1180,2,2\r\n8.2,28,1410,2,3\r\n8.3,27,1020,1,4\r\n7.6,23,4700,1,5\r\n\r\n')
        #trials query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/task/list/data_id/42693', b'{"tasks":{"task":[\n    { "task_id":338754,\n    "task_type_id":1,\n    "task_type":"Classification",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ,              {"name":"target_feature", "value":"coli"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n,  { "task_id":359909,\n    "task_type_id":5,\n    "task_type":"Clustering",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n  ]}\n}\n')

        feature_rows, label_col = OpenmlSource(42693).read()

        self.assertEqual(len(feature_rows), 5)
        self.assertEqual(len(label_col), 5)

        self.assertEqual((8.1, 27, 1410, 1), feature_rows[0])
        self.assertEqual((8.2, 29, 1180, 2), feature_rows[1])
        self.assertEqual((8.2, 28, 1410, 3), feature_rows[2])
        self.assertEqual((8.3, 27, 1020, 4), feature_rows[3])
        self.assertEqual((7.6, 23, 4700, 5), feature_rows[4])

        self.assertEqual((0,1), label_col[0])
        self.assertEqual((0,1), label_col[1])
        self.assertEqual((0,1), label_col[2])
        self.assertEqual((1,0), label_col[3])
        self.assertEqual((1,0), label_col[4])

    def test_arff_default_arff_classification(self):

        CobaConfig.Api_Keys['openml'] = None
        CobaConfig.Cacher = MemoryCacher()

        #data description query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/data/42693', b'{"data_set_description":{"id":"42693","name":"testdata","version":"2","description":"this is test data","format":"ARFF","upload_date":"2020-10-01T20:47:23","licence":"CC0","url":"https:\\/\\/www.openml.org\\/data\\/v1\\/download\\/22044555\\/testdata.arff","file_id":"22044555","visibility":"public","status":"active","processing_date":"2020-10-01 20:48:03","md5_checksum":"6656a444676c309dd8143aa58aa796ad"}}')
        #data types query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/data/features/42693', b'{"data_features":{"feature":[{"index":"0","name":"pH","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"1","name":"temperature","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"2","name":"conductivity","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"3","name":"coli","data_type":"nominal","nominal_value":[1,2],"is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"4","name":"play","data_type":"nominal","nominal_value":["no","yes"],"is_target":"true","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"}]}}')
        #data content query
        CobaConfig.Cacher.put('http://www.openml.org/data/v1/download/22044555', b'@relation weather\r\n\r\n@attribute pH real\r\n@attribute temperature real\r\n@attribute conductivity real\r\n@attribute coli {2, 1}\r\n@attribute play {yes, no}\r\n\r\n@data\r\n8.1,27,1410,2,no\r\n8.2,29,1180,2,no\r\n8.2,28,1410,2,yes\r\n8.3,27,1020,1,yes\r\n7.6,23,4700,1,yes\r\n\r\n')
        #trials query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/task/list/data_id/42693', b'{"tasks":{"task":[\n    { "task_id":338754,\n    "task_type_id":5,\n    "task_type":"Clustering",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n,  { "task_id":359909,\n    "task_type_id":5,\n    "task_type":"Clustering",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n  ]}\n}\n')

        feature_rows, label_col = OpenmlSource(42693).read()

        self.assertEqual(len(feature_rows), 5)
        self.assertEqual(len(label_col), 5)

        self.assertEqual((8.1, 27.0, 1410.0, 0), feature_rows[0])
        self.assertEqual((8.2, 29.0, 1180.0, 0), feature_rows[1])
        self.assertEqual((8.2, 28.0, 1410.0, 0), feature_rows[2])
        self.assertEqual((8.3, 27.0, 1020.0, 1), feature_rows[3])
        self.assertEqual((7.6, 23.0, 4700.0, 1), feature_rows[4])

        self.assertEqual((1,0), label_col[0])
        self.assertEqual((1,0), label_col[1])
        self.assertEqual((0,1), label_col[2])
        self.assertEqual((0,1), label_col[3])
        self.assertEqual((0,1), label_col[4])

    def test_arff_sparse_arff_classification(self):

        CobaConfig.Api_Keys['openml'] = None
        CobaConfig.Cacher = MemoryCacher()

        #data description query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/data/1594', b'{"data_set_description":{"id":"1594","name":"news20_test","version":"2","description":"this is test data","format":"Sparse_ARFF","upload_date":"2015-06-18T12:22:35","licence":"CC0","url":"https:\\/\\/www.openml.org\\/data\\/v1\\/download\\/1595696\\/news20.sparse_arff","file_id":"1595696","visibility":"public","status":"active","processing_date":"2020-10-01 20:48:03","md5_checksum":"79f56a6d9b73f90b6209199589fb2018"}}')

        #data types query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/data/features/1594', b'{"data_features":{"feature":[{"index":"0","name":"att_1","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"1","name":"att_2","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"2","name":"att_3","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"3","name":"att_4","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"4","name":"att_5","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"5","name":"att_6","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"6","name":"att_7","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"7","name":"att_8","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"8","name":"att_9","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"9","name":"att_10","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"10","name":"class","data_type":"nominal","nominal_value":["class_A","class_B","class_C","class_D"],"is_target":"true","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"}]}}')

        #data content query
        CobaConfig.Cacher.put('http://www.openml.org/data/v1/download/1595696', b'@relation news20\r\n\r\n@attribute att_1 numeric\r\n@attribute att_2 numeric\r\n@attribute att_3 numeric\r\n@attribute att_4 numeric\r\n@attribute att_5 numeric\r\n@attribute att_6 numeric\r\n@attribute att_7 numeric\r\n@attribute att_8 numeric\r\n@attribute att_9 numeric\r\n@attribute att_10 numeric\r\n@attribute class {class_A, class_B, class_C, class_D}\r\n\r\n@data\r\n{0 2,1 3,10 class_A}\r\n{2 1,3 1,4 1,6 1,8 1,10 class_B}\r\n{0 3,1 1,2 1,3 9,4 1,5 1,6 1,10 class_C}\r\n{0 1,3 1,6 1,7 1,8 1,9 2,10 class_D}\r\n\r\n')

        #trials query -- didn't modify yet
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/task/list/data_id/1594', b'{"tasks":{"task":[\n    { "task_id":338754,\n    "task_type_id":5,\n    "task_type":"Clustering",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n,  { "task_id":359909,\n    "task_type_id":5,\n    "task_type":"Clustering",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n  ]}\n}\n')

        feature_rows, label_col = OpenmlSource(1594).read()

        self.assertEqual(len(feature_rows), 4)
        self.assertEqual(len(label_col), 4)

        self.assertEqual(( (0,1)          , (2,3)          ), feature_rows[0])
        self.assertEqual(( (2,3,4,6,8)    , (1,1,1,1,1)    ), feature_rows[1])
        self.assertEqual(( (0,1,2,3,4,5,6), (3,1,1,9,1,1,1)), feature_rows[2])
        self.assertEqual(( (0,3,6,7,8,9)  , (1,1,1,1,1,2)  ), feature_rows[3])

        self.assertEqual((1,0,0,0), label_col[0])
        self.assertEqual((0,1,0,0), label_col[1])
        self.assertEqual((0,0,1,0), label_col[2])
        self.assertEqual((0,0,0,1), label_col[3])

    def test_arff_sparse_arff_missing_labels(self):

        CobaConfig.Api_Keys['openml'] = None
        CobaConfig.Cacher = MemoryCacher()

        #data description query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/data/1594', b'{"data_set_description":{"id":"1594","name":"news20_test","version":"2","description":"this is test data","format":"Sparse_ARFF","upload_date":"2015-06-18T12:22:35","licence":"CC0","url":"https:\\/\\/www.openml.org\\/data\\/v1\\/download\\/1595696\\/news20.sparse_arff","file_id":"1595696","visibility":"public","status":"active","processing_date":"2020-10-01 20:48:03","md5_checksum":"79f56a6d9b73f90b6209199589fb2018"}}')

        #data types query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/data/features/1594', b'{"data_features":{"feature":[{"index":"0","name":"att_1","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"1","name":"att_2","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"2","name":"att_3","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"3","name":"att_4","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"4","name":"att_5","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"5","name":"att_6","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"6","name":"att_7","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"7","name":"att_8","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"8","name":"att_9","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"9","name":"att_10","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"10","name":"class","data_type":"nominal","nominal_value":["class_A","class_B","class_C","class_D"],"is_target":"true","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"}]}}')

        #data content query
        CobaConfig.Cacher.put('http://www.openml.org/data/v1/download/1595696', b'@relation news20\r\n\r\n@attribute att_1 numeric\r\n@attribute att_2 numeric\r\n@attribute att_3 numeric\r\n@attribute att_4 numeric\r\n@attribute att_5 numeric\r\n@attribute att_6 numeric\r\n@attribute att_7 numeric\r\n@attribute att_8 numeric\r\n@attribute att_9 numeric\r\n@attribute att_10 numeric\r\n@attribute class {0, class_B, class_C, class_D}\r\n\r\n@data\r\n{0 2,1 3}\r\n{2 1,3 1,4 1,6 1,8 1,10 class_B}\r\n{0 3,1 1,2 1,3 9,4 1,5 1,6 1}\r\n{0 1,3 1,6 1,7 1,8 1,9 2,10 class_D}\r\n\r\n')

        #trials query -- didn't modify yet
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/task/list/data_id/1594', b'{"tasks":{"task":[\n    { "task_id":338754,\n    "task_type_id":5,\n    "task_type":"Clustering",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n,  { "task_id":359909,\n    "task_type_id":5,\n    "task_type":"Clustering",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n  ]}\n}\n')

        feature_rows, label_col = OpenmlSource(1594).read()

        self.assertEqual(len(feature_rows), 4)
        self.assertEqual(len(label_col)   , 4)

        self.assertEqual(( (2,3,4,6,8)    , (1,1,1,1,1)    ), feature_rows[0])
        self.assertEqual(( (0,3,6,7,8,9)  , (1,1,1,1,1,2)  ), feature_rows[1])
        self.assertEqual(( (0,1)          , (2,3)          ), feature_rows[2])
        self.assertEqual(( (0,1,2,3,4,5,6), (3,1,1,9,1,1,1)), feature_rows[3])

        self.assertEqual((0,1,0), label_col[0])
        self.assertEqual((0,0,1), label_col[1])
        self.assertEqual((1,0,0), label_col[2])
        self.assertEqual((1,0,0), label_col[3])

    def test_arff_not_classification(self):

        CobaConfig.Api_Keys['openml'] = None
        CobaConfig.Cacher = MemoryCacher()

        #data description query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/data/42693', b'{"data_set_description":{"id":"42693","name":"testdata","version":"2","description":"this is test data","format":"ARFF","upload_date":"2020-10-01T20:47:23","licence":"CC0","url":"https:\\/\\/www.openml.org\\/data\\/v1\\/download\\/22044555\\/testdata.arff","file_id":"22044555","visibility":"public","status":"active","processing_date":"2020-10-01 20:48:03","md5_checksum":"6656a444676c309dd8143aa58aa796ad"}}')
        #data types query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/data/features/42693', b'{"data_features":{"feature":[{"index":"0","name":"pH","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"1","name":"temperature","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"2","name":"conductivity","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"3","name":"coli","data_type":"nominal","nominal_value":[1,2],"is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"4","name":"play","data_type":"numeric","nominal_value":["no","yes"],"is_target":"true","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"}]}}')
        #data content query
        CobaConfig.Cacher.put('http://www.openml.org/data/v1/download/22044555', b'@relation weather\r\n\r\n@attribute pH real\r\n@attribute temperature real\r\n@attribute conductivity real\r\n@attribute coli real\r\n@attribute play {yes, no}\r\n\r\n@data\r\n8.1,27,1410,2,no\r\n8.2,29,1180,2,no\r\n8.2,28,1410,2,yes\r\n8.3,27,1020,2,yes\r\n7.6,23,4700,2,yes\r\n\r\n')

        #trials query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/task/list/data_id/42693', b'{"tasks":{"task":[\n    { "task_id":338754,\n    "task_type_id":5,\n    "task_type":"Clustering",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n,  { "task_id":359909,\n    "task_type_id":5,\n    "task_type":"Clustering",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n  ]}\n}\n')

        with self.assertRaises(Exception) as e:
            feature_rows, label_col = OpenmlSource(42693).read()

        self.assertTrue("does not appear" in str(e.exception))

    def test_arff_not_default_classification(self):

        CobaConfig.Api_Keys['openml'] = None
        CobaConfig.Cacher = MemoryCacher()

        #data description query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/data/42693', b'{"data_set_description":{"id":"42693","name":"testdata","version":"2","description":"this is test data","format":"ARFF","upload_date":"2020-10-01T20:47:23","licence":"CC0","url":"https:\\/\\/www.openml.org\\/data\\/v1\\/download\\/22044555\\/testdata.arff","file_id":"22044555","visibility":"public","status":"active","processing_date":"2020-10-01 20:48:03","md5_checksum":"6656a444676c309dd8143aa58aa796ad"}}')
        #data types query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/data/features/42693', b'{"data_features":{"feature":[{"index":"0","name":"pH","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"1","name":"temperature","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"2","name":"conductivity","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"3","name":"coli","data_type":"nominal","nominal_value":[1,2],"is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"4","name":"play","data_type":"numeric","is_target":"true","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"}]}}')
        #data content query
        CobaConfig.Cacher.put('http://www.openml.org/data/v1/download/22044555', b'@relation weather\r\n\r\n@attribute pH real\r\n@attribute temperature real\r\n@attribute conductivity real\r\n@attribute coli {1, 2}}\r\n@attribute play real\r\n\r\n@data\r\n8.1,27,1410,2,1\r\n8.2,29,1180,2,2\r\n8.2,28,1410,2,3\r\n8.3,27,1020,1,4\r\n7.6,23,4700,1,5\r\n\r\n')
        #trials query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/task/list/data_id/42693', b'{"tasks":{"task":[\n    { "task_id":338754,\n    "task_type_id":1,\n    "task_type":"Classification",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ,              {"name":"target_feature", "value":"coli"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n,  { "task_id":359909,\n    "task_type_id":5,\n    "task_type":"Clustering",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n  ]}\n}\n')

        feature_rows, label_col = OpenmlSource(42693).read()

        self.assertEqual(len(feature_rows), 5)
        self.assertEqual(len(label_col), 5)

        self.assertEqual((8.1, 27, 1410, 1), feature_rows[0])
        self.assertEqual((8.2, 29, 1180, 2), feature_rows[1])
        self.assertEqual((8.2, 28, 1410, 3), feature_rows[2])
        self.assertEqual((8.3, 27, 1020, 4), feature_rows[3])
        self.assertEqual((7.6, 23, 4700, 5), feature_rows[4])

        self.assertEqual((0,1), label_col[0])
        self.assertEqual((0,1), label_col[1])
        self.assertEqual((0,1), label_col[2])
        self.assertEqual((1,0), label_col[3])
        self.assertEqual((1,0), label_col[4])

class ClassificationSimulation_Tests(unittest.TestCase):

    def assert_simulation_for_data(self, simulation, features, answers) -> None:

        self.assertEqual(len(simulation.interactions), len(features))

        answers = simulation.one_hot_encoder.encode(answers)

        #first we make sure that all the labels are included 
        #in the first interactions actions without any concern for order
        self.assertCountEqual(simulation.interactions[0].actions, set(answers))

        #then we set our expected actions to the first interaction
        #to make sure that every interaction has the exact same actions
        #with the exact same order
        expected_actions = simulation.interactions[0].actions

        for f,l,i in zip(features, answers, simulation.interactions):

            expected_context = f
            expected_rewards = [ int(a == l) for a in i.actions]

            actual_context = i.context
            actual_actions = i.actions
            
            actual_rewards  = simulation.reward.observe(_choices(i))

            self.assertEqual(actual_context, expected_context)            
            self.assertSequenceEqual(actual_actions, expected_actions)
            self.assertSequenceEqual(actual_rewards, expected_rewards)

    def test_constructor_with_good_features_and_labels1(self) -> None:
        features   = [1,2,3,4]
        labels     = [1,1,0,0]
        simulation = ClassificationSimulation(features, labels)

        self.assert_simulation_for_data(simulation, features, labels)
    
    def test_constructor_with_good_features_and_labels2(self) -> None:
        features   = ["a","b"]
        labels     = ["good","bad"]
        simulation = ClassificationSimulation(features, labels)

        self.assert_simulation_for_data(simulation, features, labels)

    def test_constructor_with_good_features_and_labels3(self) -> None:
        features   = [(1,2),(3,4)]
        labels     = ["good","bad"]
        simulation = ClassificationSimulation(features, labels)

        self.assert_simulation_for_data(simulation, features, labels)

    def test_constructor_with_too_few_features(self) -> None:
        with self.assertRaises(AssertionError): 
            ClassificationSimulation([1], [1,1])

    def test_constructor_with_too_few_labels(self) -> None:
        with self.assertRaises(AssertionError): 
            ClassificationSimulation([1,1], [1])

    def test_simple_openml_source(self) -> None:
        #this test requires interet acess to download the data

        CobaConfig.Cacher = NoneCacher()

        simulation = ClassificationSimulation(*OpenmlSource(1116).read())
        #simulation = ClassificationSimulation.from_source(OpenmlSource(273))

        self.assertEqual(len(simulation.interactions), 6598)

        for rnd in simulation.interactions:

            hash(rnd.context)    #make sure these are hashable
            hash(rnd.actions[0]) #make sure these are hashable
            hash(rnd.actions[1]) #make sure these are hashable

            self.assertEqual(len(cast(Tuple,rnd.context)), 268)
            self.assertIn((1,0), rnd.actions)
            self.assertIn((0,1), rnd.actions)
            self.assertEqual(len(rnd.actions),2)
            
            actual_rewards  = simulation.reward.observe(_choices(rnd))

            self.assertIn(1, actual_rewards)
            self.assertIn(0, actual_rewards)

    @unittest.skip("much of what makes this openml set slow is now tested locally in `test_large_from_table`")
    def test_large_from_openml(self) -> None:
        #this test requires interet acess to download the data

        CobaConfig.Cacher = MemoryCacher()
        OpenmlSource(154).read() #this will cause it to read and cache in memory so we don't measure read time

        time = min(timeit.repeat(lambda:ClassificationSimulation(*OpenmlSource(154).read()), repeat=1, number=1))

        print(time)

        #with caching took approximately 17 seconds to encode
        self.assertLess(time, 30)

class MemorySimulation_Tests(unittest.TestCase):

    def test_interactions(self):
        interactions = [Interaction(0, 1, [1,2,3]), Interaction(1, 2, [4,5,6])]
        reward       = MemoryReward([ (0,1,0), (0,2,1), (0,3,2), (1,4,2), (1,5,3), (1,6,4) ])

        simulation = MemorySimulation(interactions, reward)

        self.assertEqual(interactions[0], simulation.interactions[0])
        self.assertEqual(interactions[1], simulation.interactions[1])
        self.assertEqual(reward         , simulation.reward)

class LambdaSimulation_Tests(unittest.TestCase):

    def test_interactions(self):
        
        def C(i:int) -> int:
            return [1,2][i]

        def A(i:int,c:int) -> List[int]:
            return [[1,2,3],[4,5,6]][i]
        
        def R(i:int,c:int,a:int) -> int:
            return a-c

        simulation = LambdaSimulation(2,C,A,R).read() #type: ignore

        self.assertEqual(1      , simulation.interactions[0].context)
        self.assertEqual([1,2,3], simulation.interactions[0].actions)
        self.assertEqual([0,1,2], simulation.reward.observe([(0,1,1),(0,1,2),(0,1,3)]))

        self.assertEqual(2      , simulation.interactions[1].context)
        self.assertEqual([4,5,6], simulation.interactions[1].actions)
        self.assertEqual([2,3,4], simulation.reward.observe([(1,1,4),(1,1,5),(1,1,6)]))

    def test_interactions_len(self):
        def C(i:int) -> int:
            return [1,2][i]

        def A(i:int,c:int) -> List[int]:
            return [[1,2,3],[4,5,6]][i]

        def R(i:int, c:int,a:int) -> int:
            return a-c

        simulation = LambdaSimulation(2,C,A,R).read() #type: ignore
        self.assertEqual(len(simulation.interactions), 2)

class OpenmlSimulation_Tests(unittest.TestCase):

    def test_simple(self):
        OpenmlSimulation(150)
        self.assertTrue(True)

    def test_repr(self):
        self.assertEqual('{"OpenmlSimulation":150}', str(OpenmlSimulation(150)))

class Shuffle_Tests(unittest.TestCase):
    
    def test_shuffle(self):
        interactions = list(map(Interaction, range(3), repeat(1), repeat([1,2])))
        reward       = MemoryReward([ (0,1,3), (0,2,3), (1,1,4), (1,2,4), (2,1,5), (2,2,5) ])
        
        simulation = MemorySimulation(interactions,reward)
        shuffled_simulation = Shuffle(40).filter(simulation)

        self.assertEqual(3, len(simulation.interactions))
        self.assertEqual(interactions[0], simulation.interactions[0])
        self.assertEqual(interactions[1], simulation.interactions[1])
        self.assertEqual(interactions[2], simulation.interactions[2])
        self.assertEqual(reward         , simulation.reward)

        self.assertEqual(3, len(shuffled_simulation.interactions))
        self.assertEqual(interactions[1], shuffled_simulation.interactions[0])
        self.assertEqual(interactions[2], shuffled_simulation.interactions[1])
        self.assertEqual(interactions[0], shuffled_simulation.interactions[2])
        self.assertEqual(reward         , shuffled_simulation.reward)

class Take_Tests(unittest.TestCase):
    
    def test_take1(self):

        interactions = list(map(Interaction, range(3), repeat(1), repeat([1,2])))
        reward       = MemoryReward([ (0,1,3), (0,2,3), (1,1,4), (1,2,4), (2,1,5), (2,2,5) ])

        simulation = MemorySimulation(interactions,reward)
        take_simulation = Take(1).filter(simulation)

        self.assertEqual(1, len(take_simulation.interactions))
        self.assertEqual(interactions[0], take_simulation.interactions[0])
        self.assertEqual(reward, take_simulation.reward)

        self.assertEqual(3, len(simulation.interactions))
        self.assertEqual(interactions[0], simulation.interactions[0])
        self.assertEqual(interactions[1], simulation.interactions[1])
        self.assertEqual(interactions[2], simulation.interactions[2])
        self.assertEqual(reward         , simulation.reward)

    def test_take2(self):
        interactions = list(map(Interaction, range(3), repeat(1), repeat([1,2])))
        reward       = MemoryReward([ (0,1,3), (0,2,3), (1,1,4), (1,2,4), (2,1,5), (2,2,5) ])

        simulation = MemorySimulation(interactions,reward)
        take_simulation = Take(2).filter(simulation)

        self.assertEqual(2              , len(take_simulation.interactions))
        self.assertEqual(interactions[0], take_simulation.interactions[0])
        self.assertEqual(interactions[1], take_simulation.interactions[1])
        self.assertEqual(reward         , take_simulation.reward)

        self.assertEqual(3              , len(simulation.interactions))
        self.assertEqual(interactions[0], simulation.interactions[0])
        self.assertEqual(interactions[1], simulation.interactions[1])
        self.assertEqual(interactions[2], simulation.interactions[2])
        self.assertEqual(reward         , simulation.reward)

    def test_take3(self):
        interactions = list(map(Interaction, range(3), repeat(1), repeat([1,2])))
        reward       = MemoryReward([ (0,1,3), (0,2,3), (1,1,4), (1,2,4), (2,1,5), (2,2,5) ])
        
        simulation = MemorySimulation(interactions,reward)
        take_simulation = Take(3).filter(simulation)

        self.assertEqual(3              , len(take_simulation.interactions))
        self.assertEqual(interactions[0], take_simulation.interactions[0])
        self.assertEqual(interactions[1], take_simulation.interactions[1])
        self.assertEqual(interactions[2], take_simulation.interactions[2])
        self.assertEqual(reward         , take_simulation.reward)

        self.assertEqual(3              , len(simulation.interactions))
        self.assertEqual(interactions[0], simulation.interactions[0])
        self.assertEqual(interactions[1], simulation.interactions[1])
        self.assertEqual(interactions[2], simulation.interactions[2])
        self.assertEqual(reward         , simulation.reward)

    def test_take4(self):
        interactions = list(map(Interaction, range(3), repeat(1), repeat([1,2])))
        reward       = MemoryReward([ (0,1,3), (0,2,3), (1,1,4), (1,2,4), (2,1,5), (2,2,5) ])
        
        simulation = MemorySimulation(interactions,reward)
        take_simulation = Take(4).filter(simulation)

        self.assertEqual(3, len(simulation.interactions))
        self.assertEqual(0, len(take_simulation.interactions))

    def test_repr(self):
        self.assertEqual('{"Take":2}', str(Take(2)))
        self.assertEqual('{"Take":null}', str(Take(None)))

class Batch_Tests(unittest.TestCase):

    def test_size_batch1(self):
        interactions = list(map(Interaction, range(3), repeat(1), repeat([1,2])))
        reward       = MemoryReward([ (0,1,3), (0,2,3), (1,1,4), (1,2,4), (2,1,5), (2,2,5) ])

        simulation = MemorySimulation(interactions,reward)
        batch_simulation = Batch(size=1).filter(simulation)

        self.assertEqual(3, len(batch_simulation.interaction_batches))

        self.assertEqual(1, len(batch_simulation.interaction_batches[0]))
        self.assertEqual(0, batch_simulation.interaction_batches[0][0].key)

        self.assertEqual(1, len(batch_simulation.interaction_batches[1]))
        self.assertEqual(1, batch_simulation.interaction_batches[1][0].key)

        self.assertEqual(1, len(batch_simulation.interaction_batches[2]))
        self.assertEqual(2, batch_simulation.interaction_batches[2][0].key)

    def test_size_batch2(self):
        interactions = list(map(Interaction, range(3), repeat(1), repeat([1,2])))
        reward       = MemoryReward([ (0,1,3), (0,2,3), (1,1,4), (1,2,4), (2,1,5), (2,2,5) ])

        simulation = MemorySimulation(interactions,reward)
        batch_simulation = Batch(size=2).filter(simulation)

        self.assertEqual(1, len(batch_simulation.interaction_batches))

        self.assertEqual(2, len(batch_simulation.interaction_batches[0]))
        self.assertEqual(0, batch_simulation.interaction_batches[0][0].key)
        self.assertEqual(1, batch_simulation.interaction_batches[0][1].key)

    def test_size_batch3(self):
        interactions = list(map(Interaction, range(3), repeat(1), repeat([1,2])))
        reward       = MemoryReward([ (0,1,3), (0,2,3), (1,1,4), (1,2,4), (2,1,5), (2,2,5) ])

        simulation = MemorySimulation(interactions,reward)
        batch_simulation = Batch(size=3).filter(simulation)

        self.assertEqual(1, len(batch_simulation.interaction_batches))

        self.assertEqual(3, len(batch_simulation.interaction_batches[0]))
        self.assertEqual(0, batch_simulation.interaction_batches[0][0].key)
        self.assertEqual(1, batch_simulation.interaction_batches[0][1].key)
        self.assertEqual(2, batch_simulation.interaction_batches[0][2].key)

    def test_count_batch1(self):
        interactions = list(map(Interaction, range(3), repeat(1), repeat([1,2])))
        reward       = MemoryReward([ (0,1,3), (0,2,3), (1,1,4), (1,2,4), (2,1,5), (2,2,5) ])

        simulation = MemorySimulation(interactions,reward)
        batch_simulation = Batch(count=1).filter(simulation)

        self.assertEqual(1, len(batch_simulation.interaction_batches))

        self.assertEqual(3, len(batch_simulation.interaction_batches[0]))
        self.assertEqual(0, batch_simulation.interaction_batches[0][0].key)
        self.assertEqual(1, batch_simulation.interaction_batches[0][1].key)
        self.assertEqual(2, batch_simulation.interaction_batches[0][2].key)

    def test_count_batch2(self):
        interactions = list(map(Interaction, range(3), repeat(1), repeat([1,2])))
        reward       = MemoryReward([ (0,1,3), (0,2,3), (1,1,4), (1,2,4), (2,1,5), (2,2,5) ])

        simulation = MemorySimulation(interactions,reward)
        batch_simulation = Batch(count=2).filter(simulation)

        self.assertEqual(2, len(batch_simulation.interaction_batches))

        self.assertEqual(2, len(batch_simulation.interaction_batches[0]))
        self.assertEqual(0, batch_simulation.interaction_batches[0][0].key)
        self.assertEqual(1, batch_simulation.interaction_batches[0][1].key)
        
        self.assertEqual(1, len(batch_simulation.interaction_batches[1]))
        self.assertEqual(2, batch_simulation.interaction_batches[1][0].key)

    def test_count_batch3(self):
        interactions = list(map(Interaction, range(3), repeat(1), repeat([1,2])))
        reward       = MemoryReward([ (0,1,3), (0,2,3), (1,1,4), (1,2,4), (2,1,5), (2,2,5) ])

        simulation = MemorySimulation(interactions,reward)
        batch_simulation = Batch(count=3).filter(simulation)

        self.assertEqual(3, len(batch_simulation.interaction_batches))

        self.assertEqual(1, len(batch_simulation.interaction_batches[0]))
        self.assertEqual(0, batch_simulation.interaction_batches[0][0].key)

        self.assertEqual(1, len(batch_simulation.interaction_batches[1]))
        self.assertEqual(1, batch_simulation.interaction_batches[1][0].key)

        self.assertEqual(1, len(batch_simulation.interaction_batches[2]))
        self.assertEqual(2, batch_simulation.interaction_batches[2][0].key)

    def test_repr(self):
        self.assertEqual('{"Batch":{"count":2}}', str(Batch(count=2)))
        self.assertEqual('{"Batch":{"size":2}}', str(Batch(size=2)))
        self.assertEqual('{"Batch":{"sizes":[1,2,3]}}', str(Batch(sizes=[1,2,3])))

class Interaction_Tests(unittest.TestCase):

    def test_constructor_no_context(self) -> None:
        Interaction(0, None, [1, 2])

    def test_constructor_context(self) -> None:
        Interaction(0, (1,2,3,4), [1, 2])

    def test_context_correct_1(self) -> None:
        self.assertEqual(None, Interaction(0, None, [1, 2]).context)

    def test_actions_correct_1(self) -> None:
        self.assertSequenceEqual([1, 2], Interaction(0, None, [1, 2]).actions)

    def test_actions_correct_2(self) -> None:
        self.assertSequenceEqual(["A", "B"], Interaction(0, None, ["A", "B"]).actions)

    def test_actions_correct_3(self) -> None:
        self.assertSequenceEqual([(1,2), (3,4)], Interaction(0, None, [(1,2), (3,4)]).actions)

class PCA_Tests(unittest.TestCase):
    def test_PCA(self):        
        interactions = [
            Interaction(0, (1,2), [1]),
            Interaction(1, (1,9), [1]),
            Interaction(2, (7,3), [1])
        ]
        rewards = MemoryReward([ (0,1,1), (1,1,1), (2,1,1) ])

        mem_sim = MemorySimulation(interactions, rewards)
        pca_sim = PCA().filter(mem_sim)

        self.assertEqual((1,2), mem_sim.interactions[0].context)
        self.assertEqual((1,9), mem_sim.interactions[1].context)
        self.assertEqual((7,3), mem_sim.interactions[2].context)

        self.assertNotEqual((1,2), pca_sim.interactions[0].context)
        self.assertNotEqual((1,9), pca_sim.interactions[1].context)
        self.assertNotEqual((7,3), pca_sim.interactions[2].context)

    def test_repr(self):
        self.assertEqual('"PCA"', str(PCA()))

class Sort_tests(unittest.TestCase):

    def test_sort1(self) -> None:

        interactions = [
            Interaction(0, (7,2), [1]),
            Interaction(1, (1,9), [1]),
            Interaction(2, (8,3), [1])
        ]
        rewards = MemoryReward([ (0,1,1), (1,1,1), (2,1,1) ])

        mem_sim = MemorySimulation(interactions, rewards)
        srt_sim = Sort([0]).filter(mem_sim)

        self.assertEqual((7,2), mem_sim.interactions[0].context)
        self.assertEqual((1,9), mem_sim.interactions[1].context)
        self.assertEqual((8,3), mem_sim.interactions[2].context)

        self.assertEqual((1,9), srt_sim.interactions[0].context)
        self.assertEqual((7,2), srt_sim.interactions[1].context)
        self.assertEqual((8,3), srt_sim.interactions[2].context)

    def test_sort2(self) -> None:

        interactions = [
            Interaction(0, (1,2), [1]),
            Interaction(1, (1,9), [1]),
            Interaction(2, (1,3), [1])
        ]
        reward = MemoryReward([ (0,1,1), (1,1,1), (2,1,1) ])

        mem_sim = MemorySimulation(interactions, reward)
        srt_sim = Sort([0,1]).filter(mem_sim)

        self.assertEqual((1,2), mem_sim.interactions[0].context)
        self.assertEqual((1,9), mem_sim.interactions[1].context)
        self.assertEqual((1,3), mem_sim.interactions[2].context)

        self.assertEqual((1,2), srt_sim.interactions[0].context)
        self.assertEqual((1,3), srt_sim.interactions[1].context)
        self.assertEqual((1,9), srt_sim.interactions[2].context)

    def test_sort3(self) -> None:
        interactions = [
            Interaction(0, (1,2), [1]),
            Interaction(1, (1,9), [1]),
            Interaction(2, (1,3), [1])
        ]
        reward = MemoryReward([ (0,1,1), (1,1,1), (2,1,1) ])

        mem_sim = MemorySimulation(interactions, reward)
        srt_sim = Sort(*[0,1]).filter(mem_sim)

        self.assertEqual((1,2), mem_sim.interactions[0].context)
        self.assertEqual((1,9), mem_sim.interactions[1].context)
        self.assertEqual((1,3), mem_sim.interactions[2].context)

        self.assertEqual((1,2), srt_sim.interactions[0].context)
        self.assertEqual((1,3), srt_sim.interactions[1].context)
        self.assertEqual((1,9), srt_sim.interactions[2].context)
    
    def test_repr(self):
        self.assertEqual('{"Sort":[0]}', str(Sort([0])))
        self.assertEqual('{"Sort":[1,2]}', str(Sort([1,2])))

if __name__ == '__main__':
    unittest.main()