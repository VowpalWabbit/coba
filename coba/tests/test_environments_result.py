import unittest.mock
import unittest

from coba.json         import loads
from coba.exceptions   import CobaException
from coba.pipes        import IdentitySource
from coba.environments import ResultEnvironment
from coba.primitives   import L1Reward

class ResultEnvironment_Tests(unittest.TestCase):

    def test_actions_rewards(self):
        source = {
            "actions":[[1,2],[1,2]],
            "rewards":[[0,1],[1,0]]
        }
        actual = list(ResultEnvironment(IdentitySource(source),None,None,None).read())
        expected = [
            {'actions':[1,2],"rewards":[0,1]},
            {'actions':[1,2],"rewards":[1,0]}
        ]
        self.assertEqual(actual,expected)

    def test_action_reward(self):
        source = {
            "action":[1,2],
            "reward":[1,0]
        }
        actual = list(ResultEnvironment(IdentitySource(source),None,None,None).read())
        expected = [
            {'action':1,"reward":1},
            {'action':2,"reward":0}
        ]
        self.assertEqual(actual,expected)

    def test_registration_json(self):
        source = loads('''{"actions":[[1,2],[3,4]],"rewards":[{"L1":1},{"L1":2}]}''')
        actual = list(ResultEnvironment(IdentitySource(source),None,None,None).read())
        expected = [
            {'actions':[1,2],"rewards":L1Reward(1)},
            {'actions':[3,4],"rewards":L1Reward(2)},
        ]
        self.assertEqual(actual,expected)

    def test_no_action_or_actions(self):
        source = {"reward":[1,0]}

        with self.assertRaises(CobaException) as e:
            list(ResultEnvironment(IdentitySource(source),None,None,None).read())

        self.assertIn("ResultEnvironment",str(e.exception))

if __name__ == '__main__':
    unittest.main()
