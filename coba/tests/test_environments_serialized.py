import unittest

from coba.environments.serialized import EnvironmentToBytes, EnvironmentFromBytes

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
        output_env = EnvironmentFromBytes(EnvironmentToBytes(input_env))

        self.assertEqual(input_env.params, output_env.params)
        self.assertEqual(list(input_env.read()), list(output_env.read()))

if __name__ == '__main__':
    unittest.main()
    