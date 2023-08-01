import importlib.util
import math
import unittest
import unittest.mock
import warnings

from coba.exceptions import CobaException
from coba.contexts import CobaContext
from coba.environments import Batch, OpeRewards, SimpleEnvironment
from coba.environments import SimulatedInteraction, LoggedInteraction, GroundedInteraction
from coba.learners import Learner, VowpalSoftmaxLearner
from coba.primitives import SequenceReward, Batch as BatchType

from coba.evaluators import OnPolicyEvaluator, OffPolicyEvaluator, ExplorationEvaluator

#for testing purposes
class BatchFixedLearner(Learner):

    @property
    def params(self):
        return {"family": "Recording"}

    def predict(self, context, actions):
        return [[1,0,0],[0,0,1]]

    def learn(self, context, action, reward, probability, **kwargs):
        pass

class FixedActionProbLearner(Learner):
    def __init__(self, actions, probs):
        self._actions = actions
        self._probs   = probs

    @property
    def params(self):
        return {"family": "Recording"}

    def predict(self, context, actions):
        has_p = bool(self._probs)
        a = self._actions.pop(0)
        p = self._probs.pop(0) if has_p else None
        return (a, p) if has_p else (a)

    def learn(self, context, action, reward, probability, **kwargs):
        pass

class RecordingLearner(Learner):
    def __init__(self, with_info:bool = True, with_log:bool = True):

        self._i               = 0
        self.predict_calls   = []
        self.predict_returns = []
        self.learn_calls     = []
        self._with_info       = with_info
        self._with_log        = with_log

    @property
    def params(self):
        return {"family": "Recording"}

    def predict(self, context, actions):

        self._i += 1

        if self._with_log:
            CobaContext.learning_info.update(predict=self._i)

        action_index = len(self.predict_calls) % len(actions)
        self.predict_calls.append((context, actions))

        probs = [ int(i == action_index) for i in range(len(actions)) ]
        self.predict_returns.append((probs, {'i':self._i}) if self._with_info else probs)
        return self.predict_returns[-1]

    def learn(self, context, action, reward, probability, **kwargs):

        if self._with_log:
            CobaContext.learning_info.update(learn=self._i)

        self.learn_calls.append((context, action, reward, probability, kwargs))

class DummyIglLearner(Learner):

    def __init__(self, predictions):
        self._predictions = predictions
        self._n_predicts = 0
        self._predict_calls = []
        self._learn_calls = []

    def predict(self, *args):
        self._n_predicts += 1
        CobaContext.learning_info['n_predict'] = self._n_predicts
        self._predict_calls.append(args)
        return self._predictions.pop(0)

    def learn(self, *args, **kwargs):
        self._learn_calls.append((*args, kwargs))
#for testing purposes

class OnPolicyEvaluator_Tests(unittest.TestCase):

    def test_no_actions(self):
        task         = OnPolicyEvaluator("reward")
        learner      = RecordingLearner(with_info=False, with_log=False)
        interactions = [
            {'context':[1,2,3], "rewards":[1,2,3]}
        ]

        with self.assertRaises(CobaException):
            task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

    def test_none_actions(self):
        task         = OnPolicyEvaluator("reward")
        learner      = RecordingLearner(with_info=False, with_log=False)
        interactions = [
            {'context':[1,2,3], "rewards":[1,2,3], "actions":None}
        ]

        with self.assertRaises(CobaException):
            task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

    def test_no_rewards(self):
        task         = OnPolicyEvaluator("reward")
        learner      = RecordingLearner(with_info=False, with_log=False)
        interactions = [
            {'context':[1,2,3], "actions":[1,2,3]}
        ]

        with self.assertRaises(CobaException):
            task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

    def test_none_rewards(self):
        task         = OnPolicyEvaluator("reward")
        learner      = RecordingLearner(with_info=False, with_log=False)
        interactions = [
            {'context':[1,2,3], "actions":[1,2,3], "rewards":None}
        ]

        with self.assertRaises(CobaException):
            task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

    def test_one_metric(self):
        task         = OnPolicyEvaluator("reward")
        learner      = RecordingLearner(with_info=False, with_log=False)
        interactions = [
            SimulatedInteraction(None,[1,2,3],[7,8,9]),
            SimulatedInteraction(None,[4,5,6],[4,5,6]),
            SimulatedInteraction(None,[7,8,9],[1,2,3]),
        ]

        task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        expected_predict_calls   = [(None,[1,2,3]),(None,[4,5,6]),(None,[7,8,9])]
        expected_predict_returns = [[1,0,0],[0,1,0],[0,0,1]]
        expected_learn_calls     = [(None,1,7,1,{}),(None,5,5,1,{}),(None,9,3,1,{})]
        expected_task_results    = [
            {"reward":7},
            {"reward":5},
            {"reward":3}
        ]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_all_reward_metrics(self):
        task         = OnPolicyEvaluator(["reward","rank","regret"])
        learner      = RecordingLearner(with_info=False, with_log=False)
        interactions = [
            SimulatedInteraction(None,[1,2,3],[7,8,9]),
            SimulatedInteraction(None,[4,5,6],[4,5,6]),
            SimulatedInteraction(None,[7,8,9],[1,2,3]),
        ]

        task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        expected_predict_calls   = [(None,[1,2,3]),(None,[4,5,6]),(None,[7,8,9])]
        expected_predict_returns = [[1,0,0],[0,1,0],[0,0,1]]
        expected_learn_calls     = [(None,1,7,1,{}),(None,5,5,1,{}),(None,9,3,1,{})]
        expected_task_results    = [
            {"reward":7,'rank':0.,'regret':2},
            {"reward":5,'rank':.5,'regret':1},
            {"reward":3,'rank':1.,'regret':0}
        ]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_none_no_info_no_logs_no_kwargs(self):
        task         = OnPolicyEvaluator()
        learner      = RecordingLearner(with_info=False, with_log=False)
        interactions = [
            SimulatedInteraction(None,[1,2,3],[7,8,9]),
            SimulatedInteraction(None,[4,5,6],[4,5,6]),
            SimulatedInteraction(None,[7,8,9],[1,2,3]),
        ]

        task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        expected_predict_calls   = [(None,[1,2,3]),(None,[4,5,6]),(None,[7,8,9])]
        expected_predict_returns = [[1,0,0],[0,1,0],[0,0,1]]
        expected_learn_calls     = [(None,1,7,1,{}),(None,5,5,1,{}),(None,9,3,1,{})]
        expected_task_results    = [
            {"reward":7},
            {"reward":5},
            {"reward":3}
        ]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_sparse_no_info_no_logs_no_kwargs(self):
        task         = OnPolicyEvaluator()
        learner      = RecordingLearner(with_info=False, with_log=False)
        interactions = [
            SimulatedInteraction({'c':1},[{'a':1},{'a':2},{'a':2}],[7,8,9]),
            SimulatedInteraction({'c':2},[{'a':4},{'a':5},{'a':2}],[4,5,9]),
        ]

        task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        expected_predict_calls   = [({'c':1},[{'a':1},{'a':2},{'a':2}]),({'c':2},[{'a':4},{'a':5},{'a':2}])]
        expected_predict_returns = [[1,0,0],[0,1,0]]
        expected_learn_calls     = [({'c':1},{'a':1},7,1,{}),({'c':2},{'a':5},5,1,{})]
        expected_task_results    = [
            {"reward":7},
            {"reward":5},
        ]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_dense_no_info_no_logs_no_kwargs(self):
        task         = OnPolicyEvaluator()
        learner      = RecordingLearner(with_info=False, with_log=False)
        interactions = [
            SimulatedInteraction(1,[1,2,3],[7,8,9]),
            SimulatedInteraction(2,[4,5,6],[4,5,6]),
            SimulatedInteraction(3,[7,8,9],[1,2,3]),
        ]

        task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        expected_predict_calls   = [(1,[1,2,3]),(2,[4,5,6]),(3,[7,8,9])]
        expected_predict_returns = [[1,0,0],[0,1,0],[0,0,1]]
        expected_learn_calls     = [(1,1,7,1,{}),(2,5,5,1,{}),(3,9,3,1,{})]
        expected_task_results    = [
            {"reward":7},
            {"reward":5},
            {"reward":3}
        ]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_continuous_no_info_no_logs_no_kwargs(self):
        task         = OnPolicyEvaluator(["reward","rank"])
        learner      = FixedActionProbLearner([0,1,2],None)
        interactions = [
            SimulatedInteraction(1,[],SequenceReward([0,1,2],[7,8,9])),
            SimulatedInteraction(2,[],SequenceReward([0,1,2],[4,5,6])),
            SimulatedInteraction(3,[],SequenceReward([0,1,2],[1,2,3])),
        ]

        with self.assertWarns(UserWarning) as w:
            task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        self.assertEqual("The rank metric can only be calculated for discrete environments", str(w.warning))

        expected_task_results = [{"reward":7},{"reward":5},{"reward":3}]

        self.assertEqual(expected_task_results, task_results)

        # test warning about rewards metric
        task         = OnPolicyEvaluator(["reward","rewards"])
        learner      = FixedActionProbLearner([0,1,2],None)
        with self.assertWarns(UserWarning) as w:
            list(task.evaluate(SimpleEnvironment(interactions),learner))
        self.assertEqual("The rewards metric can only be calculated for discrete environments", str(w.warning))

        # test warnings for both rank and rewards
        task         = OnPolicyEvaluator(["reward","rewards","rank"])
        learner      = FixedActionProbLearner([0,1,2],None)
        with self.assertWarns(UserWarning) as w:
            list(task.evaluate(SimpleEnvironment(interactions),learner))

        self.assertEqual(2, len(w.warnings))
        self.assertTrue(all([str(warning.message).endswith("can only be calculated for discrete environments")
                             for warning in w.warnings]))

        # test for neither
        task         = OnPolicyEvaluator(["reward","actions","context"])
        learner      = FixedActionProbLearner([0,1,2],None)
        with self.assertWarns(UserWarning) as w:
            # Adding a dummy warning log to test the absence of any actual warning
            # Should use assertNoLogs when switching to Python 3.10+"
            warnings.warn("Dummy warning")
            list(task.evaluate(SimpleEnvironment(interactions),learner))
        self.assertEqual(1, len(w.warnings))
        self.assertEqual("Dummy warning", str(w.warning))

    def test_info_logs_kwargs(self):
        task         = OnPolicyEvaluator()
        learner      = RecordingLearner(with_info=True, with_log=True)
        interactions = [
            SimulatedInteraction(1,[1,2,3],[7,8,9],I=1),
            SimulatedInteraction(2,[4,5,6],[4,5,6],I=2),
            SimulatedInteraction(3,[7,8,9],[1,2,3],I=3),
        ]

        task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        expected_predict_calls   = [(1,[1,2,3]),(2,[4,5,6]),(3,[7,8,9])]
        expected_predict_returns = [([1,0,0],{'i':1}),([0,1,0],{'i':2}),([0,0,1],{'i':3})]
        expected_learn_calls     = [(1,1,7,1,{'i':1}),(2,5,5,1,{'i':2}),(3,9,3,1,{'i':3})]
        expected_task_results    = [
            {"reward":7,'learn':1,'predict':1,'I':1},
            {"reward":5,'learn':2,'predict':2,'I':2},
            {"reward":3,'learn':3,'predict':3,'I':3}
        ]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_info_logs_kwargs_partial(self):
        task         = OnPolicyEvaluator()
        learner      = RecordingLearner(with_info=True, with_log=True)
        interactions = [
            SimulatedInteraction(1,[1,2,3],[7,8,9]),
            SimulatedInteraction(2,[4,5,6],[4,5,6],letter='d'),
            SimulatedInteraction(3,[7,8,9],[1,2,3],letter='g'),
        ]

        task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        expected_predict_calls   = [(1,[1,2,3]),(2,[4,5,6]),(3,[7,8,9])]
        expected_predict_returns = [([1,0,0],{'i':1}),([0,1,0],{'i':2}),([0,0,1],{'i':3})]
        expected_learn_calls     = [(1,1,7,1,{'i':1}),(2,5,5,1,{'i':2}),(3,9,3,1,{'i':3})]
        expected_task_results    = [
            {"reward":7,'learn':1,'predict':1,            },
            {"reward":5,'learn':2,'predict':2,'letter':'d'},
            {"reward":3,'learn':3,'predict':3,'letter':'g'}
        ]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_two_grounded_interactions(self):
        task    = OnPolicyEvaluator()
        learner = DummyIglLearner([[1,0,0],[0,0,1]])
        interactions = [
            GroundedInteraction(0,[1,2,3],[1,0,0],[4,5,6],userid=0,isnormal=False),
            GroundedInteraction(1,[1,2,3],[0,1,0],[7,8,9],userid=1,isnormal=True),
        ]

        expected_predict_calls = [
            (0,[1,2,3]),
            (1,[1,2,3])
        ]

        expected_learn_calls = [
            (0,1,4,1,{}),
            (1,3,9,1,{})
        ]

        expected_results = [
            dict(reward=1,feedback=4,userid=0,isnormal=False,n_predict=1),
            dict(reward=0,feedback=9,userid=1,isnormal=True ,n_predict=2)
        ]

        actual_results = list(task.evaluate(SimpleEnvironment(interactions),learner))

        self.assertEqual(expected_results, actual_results)
        self.assertEqual(expected_predict_calls, learner._predict_calls)
        self.assertEqual(expected_learn_calls, learner._learn_calls)

    def test_time(self):
        task         = OnPolicyEvaluator(['time'])
        learner      = RecordingLearner()
        interactions = [SimulatedInteraction(1,[0,1,2],SequenceReward([0,1,2],[7,8,9]))]

        task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        self.assertAlmostEqual(0, task_results[0]["predict_time"], places=1)
        self.assertAlmostEqual(0, task_results[0]["learn_time"]  , places=1)

    def test_context_logging(self):
        task                 = OnPolicyEvaluator(['reward','probability','context'])
        task_without_context = OnPolicyEvaluator(['reward','probability'])
        learner              = RecordingLearner()
        interactions         = [
            SimulatedInteraction(None ,[1,2,3],[7,8,9]),
            SimulatedInteraction(1    ,[4,5,6],[4,5,6]),
            SimulatedInteraction([1,2],[7,8,9],[1,2,3]),
        ]
        # without recording context
        task_results = list(task_without_context.evaluate(SimpleEnvironment(interactions),learner))
        self.assertNotIn('context', task_results[0])

        # recording context
        task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))
        result_contexts = [result['context'] for result in task_results]
        self.assertListEqual(result_contexts, [None, 1, [1,2]])

    def test_rewards_logging(self):
        task         = OnPolicyEvaluator(['reward','rewards'])
        learner      = FixedActionProbLearner(["action_1","action_2"],[None,None])
        interactions = [
            SimulatedInteraction(1, ["action_1", "action_2", "action_3"], [1,0,0]),
            SimulatedInteraction(2, ["action_1", "action_2", "action_3"], [0,2,0]),
        ]

        results = list(task.evaluate(SimpleEnvironment(interactions),learner))
        self.assertListEqual([[1,0,0],[0,2,0]], [result['rewards'] for result in results])
        self.assertListEqual([1,2], [result['reward'] for result in results])

    def test_rewards_logging_batched(self):
        task         = OnPolicyEvaluator(['reward','rewards'])
        learner      = BatchFixedLearner()
        interactions = [
            SimulatedInteraction(1, ["action_1", "action_2", "action_3"], [1,0,0]),
            SimulatedInteraction(2, ["action_1", "action_2", "action_3"], [0,0,2]),
        ]

        results = list(task.evaluate(SimpleEnvironment(Batch(2).filter(interactions)), learner))

        self.assertEqual(2, len(results))
        self.assertEqual(1,results[0]['reward'])
        self.assertEqual(2,results[1]['reward'])
        self.assertEqual([1, 0, 0],results[0]['rewards'])
        self.assertEqual([0, 0, 2],results[1]['rewards'])

    def test_batched(self):

        class SimpleLearner:
            def __init__(self) -> None:
                self.predict_call = []
            def predict(self,*args):
                self.predict_call.append(args)
                return [[1,0,0],[0,1,0],[0,0,1]][:len(args[0])]
            def learn(self,*args):
                self.learn_call = args

        task         = OnPolicyEvaluator()
        learner      = SimpleLearner()
        interactions = [
            SimulatedInteraction(1,[1,2,3],[7,8,9]),
            SimulatedInteraction(2,[4,5,6],[4,5,6]),
            SimulatedInteraction(3,[7,8,9],[1,2,3]),
        ]

        task_results = list(task.evaluate(SimpleEnvironment(Batch(3).filter(interactions)), learner))

        expected_predict_call = ([1,2,3],[[1,2,3],[4,5,6],[7,8,9]])
        expected_learn_call   = ([1,2,3],[1,5,9],[7,5,3],[1,1,1])
        expected_task_results  = [ {"reward":7},{"reward":5},{"reward":3} ]

        self.assertEqual(expected_predict_call, learner.predict_call[0])
        self.assertEqual(expected_learn_call, learner.learn_call)
        self.assertEqual(expected_task_results, task_results)

class OffPolicyEvaluator_Tests(unittest.TestCase):

    def test_no_rewards(self):
        task    = OffPolicyEvaluator()
        learner = RecordingLearner(with_info=False,with_log=False)
        interactions = [
            LoggedInteraction(1, 2, 3),
        ]

        with self.assertRaises(CobaException):
            task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

    def test_none_rewards(self):
        task    = OffPolicyEvaluator()
        learner = RecordingLearner(with_info=False,with_log=False)
        interactions = [
            LoggedInteraction(1, 2, 3,rewards=None,actions=[1,2,3]),
        ]

        with self.assertRaises(CobaException):
            task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

    def test_no_actions(self):
        task    = OffPolicyEvaluator()
        learner = RecordingLearner(with_info=False,with_log=False)
        interactions = [
            LoggedInteraction(1, 2, 3),
        ]

        with self.assertRaises(CobaException):
            task_results = list(task.evaluate(SimpleEnvironment(OpeRewards("IPS").filter(interactions)),learner))

    def test_no_actions_no_rewards_no_eval(self):
        task    = OffPolicyEvaluator(predict=False)
        learner = RecordingLearner(with_info=False,with_log=False)
        interactions = [
            LoggedInteraction(1, 2, 3),
            LoggedInteraction(2, 3, 4),
            LoggedInteraction(3, 4, 5)
        ]

        task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        expected_predict_calls   = []
        expected_predict_returns = []
        expected_learn_calls     = [(1,2,3,None,{}),(2,3,4,None,{}),(3,4,5,None,{})]
        expected_task_results    = []

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_actions_no_probability_no_info_no_logs(self):
        task    = OffPolicyEvaluator()
        learner = RecordingLearner(with_info=False,with_log=False)
        interactions = [
            LoggedInteraction(1, 2, 3, actions=[2,5,8]),
            LoggedInteraction(2, 3, 4, actions=[3,6,9]),
            LoggedInteraction(3, 4, 5, actions=[4,7,0])
        ]

        task_results = list(task.evaluate(SimpleEnvironment(OpeRewards("IPS").filter(interactions)),learner))

        expected_predict_calls   = [(1,[2,5,8]),(2,[3,6,9]),(3,[4,7,0])]
        expected_predict_returns = [[1,0,0],[0,1,0],[0,0,1]]
        expected_learn_calls     = [(1,2,3,None,{}),(2,3,4,None,{}),(3,4,5,None,{})]
        expected_task_results    = [{'reward': 3.0}, {'reward': 0.0}, {'reward': 0.0}]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_actions_probability_no_info_no_logs(self):
        task    = OffPolicyEvaluator()
        learner = RecordingLearner(with_info=False,with_log=False)
        interactions = [
            LoggedInteraction(1, 2, 3, probability=.2, actions=[2,5,8]),
            LoggedInteraction(2, 3, 4, probability=.3, actions=[3,6,9]),
            LoggedInteraction(3, 4, 5, probability=.4, actions=[4,7,0])
        ]

        task_results = list(task.evaluate(SimpleEnvironment(OpeRewards("IPS").filter(interactions)),learner))

        expected_predict_calls   = [(1,[2,5,8]),(2,[3,6,9]),(3,[4,7,0])]
        expected_predict_returns = [[1,0,0],[0,1,0],[0,0,1]]
        expected_learn_calls     = [(1,2,3,.2,{}),(2,3,4,.3,{}),(3,4,5,.4,{})]
        expected_task_results    = [{'reward':3/.2},{'reward':0},{'reward':0}]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_actions_probability_info_logs_kwargs(self):
        task    = OffPolicyEvaluator()
        learner = RecordingLearner(with_info=True,with_log=True)
        interactions = [
            LoggedInteraction(1, 2, 3, probability=.2, actions=[2,5,8], L='a'),
            LoggedInteraction(2, 3, 4, probability=.3, actions=[3,6,9], L='b'),
            LoggedInteraction(3, 4, 5, probability=.4, actions=[4,7,0], L='c')
        ]

        task_results = list(task.evaluate(SimpleEnvironment(OpeRewards("IPS").filter(interactions)),learner))

        expected_predict_calls   = [(1,[2,5,8]),(2,[3,6,9]),(3,[4,7,0])]
        expected_predict_returns = [([1,0,0],{'i':1}),([0,1,0],{'i':2}),([0,0,1],{'i':3})]
        expected_learn_calls     = [(1,2,3,.2,{}),(2,3,4,.3,{}),(3,4,5,.4,{})]
        expected_task_results    = [
            {'reward':3/.2, 'learn':1, 'predict':1, 'L':'a'},
            {'reward':0   , 'learn':2, 'predict':2, 'L':'b'},
            {'reward':0   , 'learn':3, 'predict':3, 'L':'c'}
        ]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_time(self):
        task         = OffPolicyEvaluator(['time'])
        learner      = RecordingLearner()
        interactions = [LoggedInteraction(1, 2, 3, actions=[2,5,8], probability=.2)]

        task_results = list(task.evaluate(SimpleEnvironment(OpeRewards("IPS").filter(interactions)),learner))

        self.assertAlmostEqual(0, task_results[0]["predict_time"], places=1)
        self.assertAlmostEqual(0, task_results[0]["learn_time"]  , places=1)

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW is not installed")
    def test_ope_loss(self):
        task         = OffPolicyEvaluator(['ope_loss'])
        interactions = [
            LoggedInteraction(1, 0, 0, actions=[0, 1], probability=1.0),
            LoggedInteraction(2, 0, 1, actions=[0, 1], probability=1.0),
            LoggedInteraction(3, 0, 0, actions=[0, 1], probability=1.0),
        ]

        #VW learner
        task_results = list(task.evaluate(SimpleEnvironment(OpeRewards("IPS").filter(interactions)),VowpalSoftmaxLearner()))
        ope_losses = [result['ope_loss'] for result in task_results]
        self.assertListEqual(ope_losses, [0.0, -1.0, -1.0])

        # Non-VW learner
        task_results = list(task.evaluate(SimpleEnvironment(OpeRewards("IPS").filter(interactions)),RecordingLearner()))
        self.assertTrue(all([math.isnan(result['ope_loss']) for result in task_results]))

    def test_batched_no_request(self):
        task                 = OffPolicyEvaluator()
        learner              = BatchFixedLearner()
        interactions         = [
            LoggedInteraction(1, "action_1", 1, probability=1.0, actions=["action_1", "action_2", "action_3"]),
            LoggedInteraction(2, "action_2", 2, probability=1.0, actions=["action_1", "action_2", "action_3"])
        ]

        #with self.assertRaises(CobaException):
        task_results = list(task.evaluate(SimpleEnvironment(Batch(2).filter(OpeRewards("IPS").filter(interactions))), learner))

        self.assertEqual(2, len(task_results))
        self.assertEqual(1,task_results[0]['reward'])
        self.assertEqual(0,task_results[1]['reward'])

    def test_batched_request_discrete(self):

        class TestLearner:
            def request(self, context, actions, request):
                return [[1,0,0],[0,0,1]]

        task                 = OffPolicyEvaluator(learn=False)
        learner              = TestLearner()
        interactions         = [
            LoggedInteraction(1, "action_1", 1, probability=1.0, actions=["action_1", "action_2", "action_3"]),
            LoggedInteraction(2, "action_2", 2, probability=1.0, actions=["action_1", "action_2", "action_3"])
        ]

        #with self.assertRaises(CobaException):
        task_results = list(task.evaluate(SimpleEnvironment(Batch(2).filter(OpeRewards("IPS").filter(interactions))),learner))

        self.assertEqual(2, len(task_results))
        self.assertEqual(1,task_results[0]['reward'])
        self.assertEqual(0,task_results[1]['reward'])

    def test_batched_request_continuous(self):
        class TestLearner:
            def request(self,context,actions,request):
                if isinstance(context,BatchType): raise Exception()
                return .5

        task    = OffPolicyEvaluator(learn=False)
        learner = TestLearner()
        interactions = [
            LoggedInteraction(1, 2, 3,actions=[]),
            LoggedInteraction(2, 3, 4,actions=[]),
        ]

        task_results = list(task.evaluate(SimpleEnvironment(Batch(2).filter(OpeRewards("IPS").filter(interactions))),learner))

        self.assertEqual(1.5,task_results[0]['reward'])
        self.assertEqual(2.0,task_results[1]['reward'])

    def test_with_request_continuous(self):
        class MyLearner:
            def request(self,context,actions,request):
                return .5

        task    = OffPolicyEvaluator(learn=False)
        learner = MyLearner()
        interactions = [
            LoggedInteraction(1, 2, 3,actions=[]),
            LoggedInteraction(2, 3, 4,actions=[]),
        ]

        task_results = list(task.evaluate(SimpleEnvironment(OpeRewards("IPS").filter(interactions)),learner))

        self.assertEqual(1.5,task_results[0]['reward'])
        self.assertEqual(2.0,task_results[1]['reward'])

class ExploreEvaluator_Tests(unittest.TestCase):

    def test_partial_interactions(self):
        task         = ExplorationEvaluator()
        learner      = RecordingLearner(with_info=False,with_log=False)
        interactions = [LoggedInteraction(1, 2, 3)]

        with self.assertRaises(CobaException) as r:
            task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        self.assertIn("`['context', 'action', 'reward', 'actions', 'probability']`",str(r.exception))

    def test_continuous_interactions(self):
        task         = ExplorationEvaluator()
        learner      = RecordingLearner(with_info=False,with_log=False)
        interactions = [LoggedInteraction(1, 2, 3, probability=1, actions=[])]

        with self.assertRaises(CobaException) as r:
            task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        self.assertIn("ExplorationEvaluator does not currently support continuous actions",str(r.exception))

    def test_batched_interactions(self):
        task         = ExplorationEvaluator()
        learner      = RecordingLearner(with_info=False,with_log=False)
        interactions = list(Batch(2).filter([LoggedInteraction(1, 2, 3, probability=1, actions=[1,2])]*2))

        with self.assertRaises(CobaException) as r:
            task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        self.assertIn("ExplorationEvaluator does not currently support batching",str(r.exception))

    def test_no_request_learner(self):
        task         = ExplorationEvaluator()
        learner      = RecordingLearner(with_info=False,with_log=False)
        interactions = [LoggedInteraction(1, 2, 3, probability=1, actions=[1,2])]

        with self.assertRaises(CobaException) as r:
            task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        self.assertIn("ExplorationEvaluator requires Learners to implement a `request` method",str(r.exception))

    def test_probability_one(self):
        class FixedRequestLearner:
            def request(self,*args):
                return [1,0,0]
            def learn(self,*args):
                pass
        task         = ExplorationEvaluator()
        learner      = FixedRequestLearner()
        interactions = [LoggedInteraction(1, 2, 3, probability=1, actions=[1,2,3])]

        task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

    def test_ope(self):

        request_returns = [
            [.25,.25,.5],
            [.05,.25,.7],
            [.25,.25,.5],
            [.05,.25,.7],
        ]

        class FixedRequestLearner:
            def request(self,*args):
                return request_returns.pop(0)
            def learn(self,*args):
                pass

        task    = ExplorationEvaluator(qpct=1,record=['reward'],cinit=1,seed=2)
        learner = FixedRequestLearner()

        interactions = [ LoggedInteraction(1, 2, 5, actions=[2,5,8], probability=.25) ] * 3
        task_results = list(task.evaluate(SimpleEnvironment(OpeRewards("IPS").filter(interactions)), learner))

        self.assertEqual(task_results,[{"reward":3}])

    def test_ope_false(self):

        request_calls = []
        learn_calls   = []

        class FixedRequestLearner:
            def request(self,*args):
                request_calls.append(args)
                return [.25,.25,.5]
            def learn(self,*args):
                learn_calls.append(args)
                pass

        task    = ExplorationEvaluator(ope=False,qpct=1,cinit=1)
        learner = FixedRequestLearner()

        interactions = [ LoggedInteraction(1, 2, 3, actions=[2,5,8], probability=.25) ] * 6
        task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        expected_request_call    = [(1,[2,5,8],[2,5,8])] * 7
        expected_learn_calls     = [(1,2,3,.25)] * 6
        expected_task_results    = [{'reward': 3.0}] * 6

        self.assertEqual(expected_request_call, request_calls)
        self.assertEqual(expected_learn_calls, learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_ope_no_rewards(self):

        class TestLearner:
            def request(self,*args):
                pass
            def learn(self,*args):
                pass

        interactions = [ LoggedInteraction(1, 2, 3, actions=[2,5,8], probability=.25) ] * 6

        with self.assertRaises(CobaException) as e:
            list(ExplorationEvaluator(ope=True).evaluate(SimpleEnvironment(interactions),TestLearner()))

        self.assertIn('interactions do not have an ope rewards', str(e.exception))

    def test_record_time(self):

        class FixedRequestLearner:
            def request(self,*args):
                return [.25,.25,.5]
            def learn(self,*args):
                pass

        task    = ExplorationEvaluator(ope=False,cinit=1,qpct=1,record=['reward','time'])
        learner = FixedRequestLearner()

        interactions = [ LoggedInteraction(1, 2, 3, actions=[2,5,8], probability=.25) ]
        task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        self.assertIn('predict_time', task_results[0])
        self.assertIn('learn_time', task_results[0])

if __name__ == '__main__':
    unittest.main()
