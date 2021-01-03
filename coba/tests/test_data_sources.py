import unittest

from coba.execution import ExecutionContext, MemoryCache
from coba.simulations import OpenmlClassificationSource

class OpenmlSource_Tests(unittest.TestCase):
    
    def test_default_classification(self):

        ExecutionContext.Config.openml_api_key = None
        ExecutionContext.FileCache = MemoryCache()

        #data description query
        ExecutionContext.FileCache.put('78c13f08e4efec8a7989618d0e009bcd.json', b'{"data_set_description":{"id":"42693","name":"testdata","version":"2","description":"this is test data","format":"ARFF","upload_date":"2020-10-01T20:47:23","licence":"CC0","url":"https:\\/\\/www.openml.org\\/data\\/v1\\/download\\/22044555\\/testdata.arff","file_id":"22044555","visibility":"public","status":"active","processing_date":"2020-10-01 20:48:03","md5_checksum":"6656a444676c309dd8143aa58aa796ad"}}')
        #data types query
        ExecutionContext.FileCache.put('8267b721252d39cfbded0eb5c3ed9b9d.json', b'{"data_features":{"feature":[{"index":"0","name":"pH","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"1","name":"temperature","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"2","name":"conductivity","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"3","name":"coli","data_type":"nominal","nominal_value":[1,2],"is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"4","name":"play","data_type":"nominal","nominal_value":["no","yes"],"is_target":"true","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"}]}}')
        #data content query
        ExecutionContext.FileCache.put('bc4715912b0aa900573293dd05d1f780.csv', b'"pH","temperature","conductivity","coli","play"\n8.1,27,1410,2,no\r\n8.2,29,1180,2,no\r\n8.2,28,1410,2,yes\r\n8.3,27,1020,1,yes\r\n7.6,23,4700,1,yes\r\n\r\n')
        #trials query
        ExecutionContext.FileCache.put('2552595de3c454d50c937f8425b846d5.json', b'{"tasks":{"task":[\n    { "task_id":338754,\n    "task_type_id":5,\n    "task_type":"Clustering",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n,  { "task_id":359909,\n    "task_type_id":5,\n    "task_type":"Clustering",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n  ]}\n}\n')

        feature_rows, label_rows = OpenmlClassificationSource(42693).read()

        self.assertEqual(len(feature_rows), 5)
        self.assertEqual(len(label_rows), 5)

        self.assertEqual((8.1, 27, 1410, 0), feature_rows[0])
        self.assertEqual((8.2, 29, 1180, 0), feature_rows[1])
        self.assertEqual((8.2, 28, 1410, 0), feature_rows[2])
        self.assertEqual((8.3, 27, 1020, 1), feature_rows[3])
        self.assertEqual((7.6, 23, 4700, 1), feature_rows[4])

        self.assertEqual((1,0), label_rows[0])
        self.assertEqual((1,0), label_rows[1])
        self.assertEqual((0,1), label_rows[2])
        self.assertEqual((0,1), label_rows[3])
        self.assertEqual((0,1), label_rows[4])

    def test_not_classification(self):

        ExecutionContext.Config.openml_api_key = None
        ExecutionContext.FileCache = MemoryCache()

        #data description query
        ExecutionContext.FileCache.put('78c13f08e4efec8a7989618d0e009bcd.json', b'{"data_set_description":{"id":"42693","name":"testdata","version":"2","description":"this is test data","format":"ARFF","upload_date":"2020-10-01T20:47:23","licence":"CC0","url":"https:\\/\\/www.openml.org\\/data\\/v1\\/download\\/22044555\\/testdata.arff","file_id":"22044555","visibility":"public","status":"active","processing_date":"2020-10-01 20:48:03","md5_checksum":"6656a444676c309dd8143aa58aa796ad"}}')
        #data types query
        ExecutionContext.FileCache.put('8267b721252d39cfbded0eb5c3ed9b9d.json', b'{"data_features":{"feature":[{"index":"0","name":"pH","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"1","name":"temperature","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"2","name":"conductivity","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"3","name":"coli","data_type":"nominal","nominal_value":[1,2],"is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"4","name":"play","data_type":"numeric","nominal_value":["no","yes"],"is_target":"true","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"}]}}')
        #data content query
        ExecutionContext.FileCache.put('bc4715912b0aa900573293dd05d1f780.csv', b'"pH","temperature","conductivity","coli","play"\n8.1,27,1410,2,1\r\n8.2,29,1180,2,2\r\n8.2,28,1410,2,3\r\n8.3,27,1020,1,4\r\n7.6,23,4700,1,5\r\n\r\n')
        #trials query
        ExecutionContext.FileCache.put('2552595de3c454d50c937f8425b846d5.json', b'{"tasks":{"task":[\n    { "task_id":338754,\n    "task_type_id":5,\n    "task_type":"Clustering",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n,  { "task_id":359909,\n    "task_type_id":5,\n    "task_type":"Clustering",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n  ]}\n}\n')

        with self.assertRaises(Exception) as e:
            feature_rows, label_rows = OpenmlClassificationSource(42693).read()

        self.assertTrue("does not appear" in str(e.exception))

    def test_not_default_classification(self):

        ExecutionContext.Config.openml_api_key = None
        ExecutionContext.FileCache = MemoryCache()

        #data description query
        ExecutionContext.FileCache.put('78c13f08e4efec8a7989618d0e009bcd.json', b'{"data_set_description":{"id":"42693","name":"testdata","version":"2","description":"this is test data","format":"ARFF","upload_date":"2020-10-01T20:47:23","licence":"CC0","url":"https:\\/\\/www.openml.org\\/data\\/v1\\/download\\/22044555\\/testdata.arff","file_id":"22044555","visibility":"public","status":"active","processing_date":"2020-10-01 20:48:03","md5_checksum":"6656a444676c309dd8143aa58aa796ad"}}')
        #data types query
        ExecutionContext.FileCache.put('8267b721252d39cfbded0eb5c3ed9b9d.json', b'{"data_features":{"feature":[{"index":"0","name":"pH","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"1","name":"temperature","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"2","name":"conductivity","data_type":"numeric","is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"3","name":"coli","data_type":"nominal","nominal_value":[1,2],"is_target":"false","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"},{"index":"4","name":"play","data_type":"numeric","nominal_value":["no","yes"],"is_target":"true","is_ignore":"false","is_row_identifier":"false","number_of_missing_values":"0"}]}}')
        #data content query
        ExecutionContext.FileCache.put('bc4715912b0aa900573293dd05d1f780.csv', b'"pH","temperature","conductivity","coli","play"\n8.1,27,1410,2,1\r\n8.2,29,1180,2,2\r\n8.2,28,1410,2,3\r\n8.3,27,1020,1,4\r\n7.6,23,4700,1,5\r\n\r\n')
        #trials query
        ExecutionContext.FileCache.put('2552595de3c454d50c937f8425b846d5.json', b'{"tasks":{"task":[\n    { "task_id":338754,\n    "task_type_id":1,\n    "task_type":"Classification",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ,              {"name":"target_feature", "value":"coli"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n,  { "task_id":359909,\n    "task_type_id":5,\n    "task_type":"Clustering",\n    "did":42693,\n    "name":"testdata",\n    "status":"active",\n    "format":"ARFF"\n        ,"input": [\n                    {"name":"estimation_procedure", "value":"17"}\n            ,              {"name":"source_data", "value":"42693"}\n            ]\n            ,"quality": [\n                    {"name":"NumberOfFeatures", "value":"5.0"}\n            ,              {"name":"NumberOfInstances", "value":"5.0"}\n            ,              {"name":"NumberOfInstancesWithMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfMissingValues", "value":"0.0"}\n            ,              {"name":"NumberOfNumericFeatures", "value":"4.0"}\n            ,              {"name":"NumberOfSymbolicFeatures", "value":"1.0"}\n            ]\n          }\n  ]}\n}\n')

        feature_rows, label_rows = OpenmlClassificationSource(42693).read()

        self.assertEqual(len(feature_rows), 5)
        self.assertEqual(len(label_rows), 5)

        self.assertEqual((8.1, 27, 1410, 1), feature_rows[0])
        self.assertEqual((8.2, 29, 1180, 2), feature_rows[1])
        self.assertEqual((8.2, 28, 1410, 3), feature_rows[2])
        self.assertEqual((8.3, 27, 1020, 4), feature_rows[3])
        self.assertEqual((7.6, 23, 4700, 5), feature_rows[4])

        self.assertEqual((0,1), label_rows[0])
        self.assertEqual((0,1), label_rows[1])
        self.assertEqual((0,1), label_rows[2])
        self.assertEqual((1,0), label_rows[3])
        self.assertEqual((1,0), label_rows[4])

if __name__ == '__main__':
    unittest.main()
