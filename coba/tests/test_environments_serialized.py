import unittest
from pathlib import Path

from coba.environments.serialized import EnvironmentFromObjects, EnvironmentToObjects, ZipMemberToObjects, ObjectsToZipMember

class EnvironmentToAndFromBytes_Tests(unittest.TestCase):

    def test_simple(self):

        class SimpleEnvironment:
            @property
            def params(self):
                return {'a':1}
            def read(self):
                yield {'b':1}
                yield {'b':2}

        input_env  = SimpleEnvironment()
        output_env = EnvironmentFromObjects(EnvironmentToObjects(input_env))

        self.assertEqual(input_env.params, output_env.params)
        self.assertEqual(list(input_env.read()), list(output_env.read()))

class ObjectsToAndFromZipMember_Tests(unittest.TestCase):

    def setUp(self) -> None:
        if Path("coba/tests/.temp/test.zip").exists(): Path("coba/tests/.temp/test.zip").unlink()

    def tearDown(self) -> None:
        if Path("coba/tests/.temp/test.zip").exists(): Path("coba/tests/.temp/test.zip").unlink()

    def test_simple(self):
        ObjectsToZipMember("coba/tests/.temp/test.zip", "a").write([1,2,3])
        self.assertEqual(list(ZipMemberToObjects("coba/tests/.temp/test.zip", "a").read()),[1,2,3])

if __name__ == '__main__':
    unittest.main()
    