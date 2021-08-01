import unittest

from typing import cast, Tuple

from coba.config import CobaConfig, NoneLogger, MemoryCacher, NoneCacher
from coba.simulations import OpenmlSimulation, OpenmlSource

CobaConfig.Logger = NoneLogger()

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

        self.assertEqual((8.1, 27, 1410, (1,)), feature_rows[0])
        self.assertEqual((8.2, 29, 1180, (1,)), feature_rows[1])
        self.assertEqual((8.2, 28, 1410, (1,)), feature_rows[2])
        self.assertEqual((8.3, 27, 1020, (0,)), feature_rows[3])
        self.assertEqual((7.6, 23, 4700, (0,)), feature_rows[4])

        self.assertEqual('no', label_col[0])
        self.assertEqual('no', label_col[1])
        self.assertEqual('yes', label_col[2])
        self.assertEqual('yes', label_col[3])
        self.assertEqual('yes', label_col[4])

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

        self.assertEqual((8.1, 27, 1410, (1,)), feature_rows[0])
        self.assertEqual((8.2, 29, 1180, (1,)), feature_rows[1])
        self.assertEqual((8.2, 28, 1410, (1,)), feature_rows[2])
        self.assertEqual((8.3, 27, 1020, (0,)), feature_rows[3])
        self.assertEqual((7.6, 23, 4700, (0,)), feature_rows[4])

        self.assertEqual('no', label_col[0])
        self.assertEqual('no', label_col[1])
        self.assertEqual('yes', label_col[2])
        self.assertEqual('yes', label_col[3])
        self.assertEqual('yes', label_col[4])

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
    
    def test_csv_not_classification_no_tasks(self):

        CobaConfig.Api_Keys['openml'] = None
        CobaConfig.Cacher = MemoryCacher()

        #data description query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/data/42693', b'{"data_set_description":{"id":"42693","name":"testdata","version":"2","description":"this is test data","format":"ARFF","upload_date":"2020-10-01T20:47:23","licence":"CC0","url":"https:\\/\\/www.openml.org\\/data\\/v1\\/download\\/22044555\\/testdata.arff","file_id":"22044555","visibility":"public","status":"active","processing_date":"2020-10-01 20:48:03","md5_checksum":"6656a444676c309dd8143aa58aa796ad"}}')
        #data types query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/data/features/42693', b'{"data_features":{"feature":[{"index":"0","name":"pH","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"1","name":"temperature","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"2","name":"conductivity","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"3","name":"coli","data_type":"nominal","nominal_value":[1,2],"is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"4","name":"play","data_type":"numeric","nominal_value":["no","yes"],"is_target":"true","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"}]}}')
        #data content query
        CobaConfig.Cacher.put('http://www.openml.org/data/v1/get_csv/22044555', b'"pH","temperature","conductivity","coli","play"\n8.1,27,1410,2,1\r\n8.2,29,1180,2,2\r\n8.2,28,1410,2,3\r\n8.3,27,1020,1,4\r\n7.6,23,4700,1,5\r\n\r\n')
        #trials query
        CobaConfig.Cacher.put('https://www.openml.org/api/v1/json/task/list/data_id/42693', b'{}\n')

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

        self.assertEqual('2', label_col[0])
        self.assertEqual('2', label_col[1])
        self.assertEqual('2', label_col[2])
        self.assertEqual('1', label_col[3])
        self.assertEqual('1', label_col[4])

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

        self.assertEqual((8.1, 27.0, 1410.0, (1,)), feature_rows[0])
        self.assertEqual((8.2, 29.0, 1180.0, (1,)), feature_rows[1])
        self.assertEqual((8.2, 28.0, 1410.0, (1,)), feature_rows[2])
        self.assertEqual((8.3, 27.0, 1020.0, (0,)), feature_rows[3])
        self.assertEqual((7.6, 23.0, 4700.0, (0,)), feature_rows[4])

        self.assertEqual('no' , label_col[0])
        self.assertEqual('no' , label_col[1])
        self.assertEqual('yes', label_col[2])
        self.assertEqual('yes', label_col[3])
        self.assertEqual('yes', label_col[4])

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
        self.assertEqual(len(label_col   ), 4)

        self.assertEqual(( (0,1)          , (2,3)          ), feature_rows[0])
        self.assertEqual(( (2,3,4,6,8)    , (1,1,1,1,1)    ), feature_rows[1])
        self.assertEqual(( (0,1,2,3,4,5,6), (3,1,1,9,1,1,1)), feature_rows[2])
        self.assertEqual(( (0,3,6,7,8,9)  , (1,1,1,1,1,2)  ), feature_rows[3])

        self.assertEqual('class_A', label_col[0])
        self.assertEqual('class_B', label_col[1])
        self.assertEqual('class_C', label_col[2])
        self.assertEqual('class_D', label_col[3])

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

        self.assertEqual(( (0,1)          , (2,3)          ), feature_rows[0])
        self.assertEqual(( (2,3,4,6,8)    , (1,1,1,1,1)    ), feature_rows[1])
        self.assertEqual(( (0,1,2,3,4,5,6), (3,1,1,9,1,1,1)), feature_rows[2])
        self.assertEqual(( (0,3,6,7,8,9)  , (1,1,1,1,1,2)  ), feature_rows[3])


        self.assertEqual('0'      , label_col[0])
        self.assertEqual('class_B', label_col[1])
        self.assertEqual('0'      , label_col[2])
        self.assertEqual('class_D', label_col[3])

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

        self.assertEqual('2', label_col[0])
        self.assertEqual('2', label_col[1])
        self.assertEqual('2', label_col[2])
        self.assertEqual('1', label_col[3])
        self.assertEqual('1', label_col[4])

class OpenmlSimulation_Tests(unittest.TestCase):

    def test_simple_openml_source(self) -> None:
        #this test requires interet acess to download the data

        CobaConfig.Cacher = NoneCacher()

        interactions = list(OpenmlSimulation(1116).read())
        
        self.assertEqual(len(interactions), 6598)

        for rnd in interactions:

            hash(rnd.context)    #make sure these are hashable
            hash(rnd.actions[0]) #make sure these are hashable
            hash(rnd.actions[1]) #make sure these are hashable

            self.assertEqual(len(cast(Tuple,rnd.context)), 268)
            self.assertIn('0', rnd.actions)
            self.assertIn('1', rnd.actions)
            self.assertEqual(len(rnd.actions),2)
            self.assertIn(1, rnd.feedbacks)
            self.assertIn(0, rnd.feedbacks)

    def test_repr(self):
        self.assertEqual('{"OpenmlSimulation":150}', str(OpenmlSimulation(150)))

if __name__ == '__main__':
    unittest.main()