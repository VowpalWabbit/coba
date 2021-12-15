
import unittest

from coba.environments import EnvironmentPipe, SimulatedEnvironment, SimulatedInteraction, Take, Identity

class TestEnvironment(SimulatedEnvironment):

    def __init__(self, id) -> None:
        self._id = id

    @property
    def params(self):
        return {'id':self._id}

    def read(self):
        return [
            SimulatedInteraction(1, [None,None], rewards=[1,2]),
            SimulatedInteraction(2, [None,None], rewards=[2,3]),
            SimulatedInteraction(3, [None,None], rewards=[3,4]),
        ]

    def __str__(self) -> str:
        return str(self.params)

class NoParamIdent:
    def filter(self,item):
        return item

    def __str__(self) -> str:
        return 'NoParamIdent'

class EnvironmentPipe_Tests(unittest.TestCase):

    def test_environment_no_filters(self):
        ep = EnvironmentPipe(TestEnvironment("A"))

        self.assertEqual(3, len(ep.read()))
        self.assertEqual({"id":"A"}, ep.params)
        self.assertEqual(str({"id":"A"}), str(ep))

    def test_environment_one_filter(self):
        ep = EnvironmentPipe(TestEnvironment("A"), Take(1))

        self.assertEqual(1, len(list(ep.read())))
        self.assertEqual({'id':'A', 'take':1}, ep.params)
        self.assertEqual("{'id': 'A'},{'take': 1}", str(ep))

    def test_environment_ident_filter_removed(self):
        ep = EnvironmentPipe(TestEnvironment("A"), Identity(), Take(1))

        self.assertEqual(1, len(list(ep.read())))
        self.assertEqual({'id':'A', 'take':1}, ep.params)
        self.assertEqual("{'id': 'A'},{'take': 1}", str(ep))

    def test_environment_no_param_ident(self):
        ep = EnvironmentPipe(TestEnvironment("A"), NoParamIdent())

        self.assertEqual(3, len(list(ep.read())))
        self.assertEqual({'id':'A'}, ep.params)
        self.assertEqual("{'id': 'A'},NoParamIdent", str(ep))


if __name__ == '__main__':
    unittest.main()