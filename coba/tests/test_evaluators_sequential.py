import math
import unittest
import unittest.mock
import warnings

from coba.utilities    import PackageChecker
from coba.exceptions   import CobaException
from coba.context      import CobaContext
from coba.environments import Batch, OpeRewards
from coba.learners     import VowpalSoftmaxLearner, EpsilonBanditLearner
from coba.primitives   import is_batch, Learner, SimulatedInteraction, LoggedInteraction, GroundedInteraction
from coba.rewards      import L1Reward, DiscreteReward
from coba.safety       import SafeLearner

from coba.evaluators.sequential import RejectionCB, SequentialCB, SequentialIGL, get_ope_loss

class SimpleEnvironment:
    def __init__(self, interactions=(), params={}) -> None:
        self.interactions = interactions
        self.params = params
    def read(self):
        return self.interactions

#for testing purposes
class BatchFixedLearner(Learner):

    @property
    def params(self):
        return {"family": "Recording"}

    def predict(self, context, actions):
        return [[1,0,0],[0,0,1]]

    def learn(self, context, action, reward, probability, **kwargs):
        pass

class FixedPredLearner(Learner):
    def __init__(self, preds):
        self._preds = preds

    @property
    def params(self):
        return {"family": "Recording"}

    def predict(self, context, actions):
        return self._preds.pop(0)

    def learn(self, context, action, reward, probability, **kwargs):
        pass

class RecordingLearner(Learner):
    def __init__(self, with_kwargs:bool = True, with_info:bool = True):

        self._i              = 0
        self.predict_calls   = []
        self.predict_returns = []
        self.learn_calls     = []
        self._with_kwargs    = with_kwargs
        self._with_info      = with_info

    @property
    def params(self):
        return {"family": "Recording"}

    def predict(self, context, actions):

        self._i += 1

        if self._with_info:
            CobaContext.learning_info.update(predict=self._i)

        action_index = len(self.predict_calls) % len(actions)
        self.predict_calls.append((context, actions))

        probs = [ int(i == action_index) for i in range(len(actions)) ]
        self.predict_returns.append((probs, {'i':self._i}) if self._with_kwargs else probs)
        return self.predict_returns[-1]

    def learn(self, context, action, reward, probability, **kwargs):

        if self._with_info:
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

class SequentialCB_Tests(unittest.TestCase):

    def test_params(self):
        self.assertEqual(SequentialCB().params,{'learn':'on','eval':'on','seed':None})

    def test_on_no_actions(self):
        task         = SequentialCB()
        learner      = RecordingLearner(with_kwargs=False, with_info=False)
        interactions = [
            {'context':[1,2,3], "rewards":[1,2,3]}
        ]

        with self.assertRaises(CobaException):
            list(task.evaluate(SimpleEnvironment(interactions), learner))

    def test_on_no_rewards(self):
        task         = SequentialCB()
        learner      = RecordingLearner(with_kwargs=False, with_info=False)
        interactions = [
            {'context':[1,2,3], "actions":[1,2,3]}
        ]

        with self.assertRaises(CobaException):
            list(task.evaluate(SimpleEnvironment(interactions), learner))

    def test_off_dr_no_actions(self):
        task    = SequentialCB(learn='off',eval='dr')
        learner = RecordingLearner(with_kwargs=False, with_info=False)
        interactions = [
            LoggedInteraction(1, 2, 3),
        ]

        with self.assertRaises(CobaException):
            list(task.evaluate(SimpleEnvironment(interactions),learner))

    def test_off_dm_no_actions(self):
        task    = SequentialCB(learn='off',eval='dm')
        learner = RecordingLearner(with_kwargs=False, with_info=False)
        interactions = [
            LoggedInteraction(1, 2, 3),
        ]

        with self.assertRaises(CobaException):
            list(task.evaluate(SimpleEnvironment(interactions),learner))

    @unittest.skip("Old functionality that we're removing to reduce upkeep.")
    def test_all_reward_metrics(self):
        task         = SequentialCB(["reward","rank","regret"])
        learner      = RecordingLearner(with_kwargs=False, with_info=False)
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

    def test_ips_None(self):
        task    = SequentialCB(learn='ips',eval=None)
        learner = RecordingLearner(with_kwargs=False, with_info=False)
        interactions = [
            LoggedInteraction(1, 2, 3, probability=.2, actions=[2,5,8]),
            LoggedInteraction(2, 3, 4, probability=.3, actions=[3,6,9]),
            LoggedInteraction(3, 4, 5, probability=.4, actions=[4,7,0])
        ]

        actual_task_results      = list(task.evaluate(SimpleEnvironment(interactions),learner))

        expected_rewards         = [ i['rewards'](action) for i,action in zip(OpeRewards("IPS").filter(interactions),[2,6,0]) ]
        expected_predict_calls   = [(1,[2,5,8]),(2,[3,6,9]),(3,[4,7,0])]
        expected_predict_returns = [[1,0,0],[0,1,0],[0,0,1]]
        expected_learn_calls     = [(1,2,expected_rewards[0],1,{}),(2,6,expected_rewards[1],1,{}),(3,0,expected_rewards[2],1,{})]
        expected_task_results    = []

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, actual_task_results)

    @unittest.skipUnless(PackageChecker.vowpalwabbit(strict=False), "VW is not installed")
    def test_dm_None(self):
        task    = SequentialCB(learn='dm',eval=None)
        learner = RecordingLearner(with_kwargs=False, with_info=False)
        interactions = [
            LoggedInteraction(1, 2, 3, probability=.2, actions=[2,5,8]),
            LoggedInteraction(2, 3, 4, probability=.3, actions=[3,6,9]),
            LoggedInteraction(3, 4, 5, probability=.4, actions=[4,7,0])
        ]

        actual_task_results      = list(task.evaluate(SimpleEnvironment(interactions),learner))

        expected_rewards         = [ i['rewards'](action) for i,action in zip(OpeRewards("DM",features=[1,'a','xa']).filter(interactions),[2,6,0]) ]
        expected_predict_calls   = [(1,[2,5,8]),(2,[3,6,9]),(3,[4,7,0])]
        expected_predict_returns = [[1,0,0],[0,1,0],[0,0,1]]
        expected_learn_calls     = [(1,2,expected_rewards[0],1,{}),(2,6,expected_rewards[1],1,{}),(3,0,expected_rewards[2],1,{})]
        expected_task_results    = []

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, actual_task_results)

    @unittest.skipUnless(PackageChecker.vowpalwabbit(strict=False), "VW is not installed")
    def test_dr_None(self):
        task    = SequentialCB(learn='dr',eval=None)
        learner = RecordingLearner(with_kwargs=False, with_info=False)
        interactions = [
            LoggedInteraction(1, 2, 3, probability=.2, actions=[2,5,8]),
            LoggedInteraction(2, 3, 4, probability=.3, actions=[3,6,9]),
            LoggedInteraction(3, 4, 5, probability=.4, actions=[4,7,0])
        ]

        actual_task_results      = list(task.evaluate(SimpleEnvironment(interactions),learner))

        expected_rewards         = [ i['rewards'](action) for i,action in zip(OpeRewards("DR",features=[1,'a','xa']).filter(interactions),[2,6,0]) ]
        expected_predict_calls   = [(1,[2,5,8]),(2,[3,6,9]),(3,[4,7,0])]
        expected_predict_returns = [[1,0,0],[0,1,0],[0,0,1]]
        expected_learn_calls     = [(1,2,expected_rewards[0],1,{}),(2,6,expected_rewards[1],1,{}),(3,0,expected_rewards[2],1,{})]
        expected_task_results    = []

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, actual_task_results)

    def test_None_ips(self):
        task    = SequentialCB(learn=None,eval='ips')
        learner = RecordingLearner(with_kwargs=False, with_info=False)
        interactions = [
            LoggedInteraction(1, 2, 3, probability=.2, actions=[2,5,8]),
            LoggedInteraction(2, 3, 4, probability=.3, actions=[3,6,9]),
            LoggedInteraction(3, 4, 5, probability=.4, actions=[4,7,0])
        ]

        actual_task_results      = list(task.evaluate(SimpleEnvironment(interactions),learner))

        expected_rewards         = [ i['rewards'](action) for i,action in zip(OpeRewards("IPS").filter(interactions),[2,6,0]) ]
        expected_predict_calls   = [(1,[2,5,8]),(2,[3,6,9]),(3,[4,7,0])]
        expected_predict_returns = [[1,0,0],[0,1,0],[0,0,1]]
        expected_learn_calls     = []
        expected_task_results    = [{'reward':expected_rewards[0]},{'reward':expected_rewards[1]},{'reward':expected_rewards[2]}]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, actual_task_results)

    @unittest.skipUnless(PackageChecker.vowpalwabbit(strict=False), "VW is not installed")
    def test_None_dm(self):
        task    = SequentialCB(learn=None,eval='dm')
        learner = RecordingLearner(with_kwargs=False, with_info=False)
        interactions = [
            LoggedInteraction(1, 2, 3, probability=.2, actions=[2,5,8]),
            LoggedInteraction(2, 3, 4, probability=.3, actions=[3,6,9]),
            LoggedInteraction(3, 4, 5, probability=.4, actions=[4,7,0])
        ]

        actual_task_results      = list(task.evaluate(SimpleEnvironment(interactions),learner))

        expected_rewards         = [ i['rewards'](action) for i,action in zip(OpeRewards("DM",features=[1,'a','xa']).filter(interactions),[2,6,0]) ]
        expected_predict_calls   = [(1,[2,5,8]),(2,[3,6,9]),(3,[4,7,0])]
        expected_predict_returns = [[1,0,0],[0,1,0],[0,0,1]]
        expected_learn_calls     = []
        expected_task_results    = [{'reward':expected_rewards[0]},{'reward':expected_rewards[1]},{'reward':expected_rewards[2]}]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, actual_task_results)

    @unittest.skipUnless(PackageChecker.vowpalwabbit(strict=False), "VW is not installed")
    def test_None_dr(self):
        task    = SequentialCB(learn=None,eval='dr')
        learner = RecordingLearner(with_kwargs=False, with_info=False)
        interactions = [
            LoggedInteraction(1, 2, 3, probability=.2, actions=[2,5,8]),
            LoggedInteraction(2, 3, 4, probability=.3, actions=[3,6,9]),
            LoggedInteraction(3, 4, 5, probability=.4, actions=[4,7,0])
        ]

        actual_task_results      = list(task.evaluate(SimpleEnvironment(interactions),learner))

        expected_rewards         = [ i['rewards'](action) for i,action in zip(OpeRewards("DR",features=[1,'a','xa']).filter(interactions),[2,6,0]) ]
        expected_predict_calls   = [(1,[2,5,8]),(2,[3,6,9]),(3,[4,7,0])]
        expected_predict_returns = [[1,0,0],[0,1,0],[0,0,1]]
        expected_learn_calls     = []
        expected_task_results    = [{'reward':expected_rewards[0]},{'reward':expected_rewards[1]},{'reward':expected_rewards[2]}]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, actual_task_results)

    @unittest.skipUnless(PackageChecker.vowpalwabbit(strict=False), "VW is not installed")
    def test_dr_dr(self):
        task    = SequentialCB(learn='dr',eval='dr')
        learner = RecordingLearner(with_kwargs=False, with_info=False)
        interactions = [
            LoggedInteraction(1, 2, 3, probability=.2, actions=[2,5,8]),
            LoggedInteraction(2, 3, 4, probability=.3, actions=[3,6,9]),
            LoggedInteraction(3, 4, 5, probability=.4, actions=[4,7,0])
        ]

        actual_task_results = list(task.evaluate(SimpleEnvironment(interactions),learner))

        dr_rewards  = [ i['rewards'](action) for i,action in zip(OpeRewards("DR",features=[1,'a','xa']).filter(interactions),[2,6,0]) ]

        expected_predict_calls   = [(1,[2,5,8]),(2,[3,6,9]),(3,[4,7,0])]
        expected_predict_returns = [[1,0,0],[0,1,0],[0,0,1]]
        expected_learn_calls     = [(1,2,dr_rewards[0],1,{}),(2,6,dr_rewards[1],1,{}),(3,0,dr_rewards[2],1,{})]
        expected_task_results    = [{'reward':dr_rewards[0]},{'reward':dr_rewards[1]},{'reward':dr_rewards[2]}]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, actual_task_results)

    @unittest.skipUnless(PackageChecker.vowpalwabbit(strict=False), "VW is not installed")
    def test_ips_dr(self):
        task    = SequentialCB(learn='ips',eval='dr')
        learner = RecordingLearner(with_kwargs=False, with_info=False)
        interactions = [
            LoggedInteraction(1, 2, 3, probability=.2, actions=[2,5,8]),
            LoggedInteraction(2, 3, 4, probability=.3, actions=[3,6,9]),
            LoggedInteraction(3, 4, 5, probability=.4, actions=[4,7,0])
        ]

        actual_task_results      = list(task.evaluate(SimpleEnvironment(interactions),learner))

        ips_rewards = [ i['rewards'](action) for i,action in zip(OpeRewards("IPS",features=[1,'a','xa']).filter(interactions),[2,6,0]) ]
        dr_rewards  = [ i['rewards'](action) for i,action in zip(OpeRewards("DR",features=[1,'a','xa']).filter(interactions),[2,6,0]) ]

        expected_predict_calls   = [(1,[2,5,8]),(2,[3,6,9]),(3,[4,7,0])]
        expected_predict_returns = [[1,0,0],[0,1,0],[0,0,1]]
        expected_learn_calls     = [(1,2,ips_rewards[0],1,{}),(2,6,ips_rewards[1],1,{}),(3,0,ips_rewards[2],1,{})]
        expected_task_results    = [{'reward':dr_rewards[0]},{'reward':dr_rewards[1]},{'reward':dr_rewards[2]}]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, actual_task_results)

    def test_on_none_context(self):
        task         = SequentialCB()
        learner      = RecordingLearner(with_kwargs=False, with_info=False)
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

    def test_on_sparse_context_sparse_actions(self):
        task         = SequentialCB()
        learner      = RecordingLearner(with_kwargs=False, with_info=False)
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

    def test_on_dense_context_dense_actions(self):
        task         = SequentialCB()
        learner      = RecordingLearner(with_kwargs=False, with_info=False)
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

    def test_on_continuous_actions(self):
        task         = SequentialCB()
        learner      = FixedPredLearner([0,1,2])
        interactions = [
            SimulatedInteraction(1,[],DiscreteReward([0,1,2],[7,8,9])),
            SimulatedInteraction(2,[],DiscreteReward([0,1,2],[4,5,6])),
            SimulatedInteraction(3,[],DiscreteReward([0,1,2],[1,2,3])),
        ]

        actual_task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))
        expected_task_results = [{"reward":7},{"reward":5},{"reward":3}]

        self.assertEqual(expected_task_results, actual_task_results)

        # test warning about rewards metric
        task         = SequentialCB(["reward","rewards"])
        learner      = FixedPredLearner([0,1,2])

        # test for neither
        task         = SequentialCB(["reward","actions","context"])
        learner      = FixedPredLearner([0,1,2])
        with self.assertWarns(UserWarning) as w:
            # Adding a dummy warning log to test the absence of any actual warning
            # Should use assertNoLogs when switching to Python 3.10+"
            warnings.warn("Dummy warning")
            list(task.evaluate(SimpleEnvironment(interactions),learner))
        self.assertEqual(1, len(w.warnings))
        self.assertEqual("Dummy warning", str(w.warning))

    def test_on_info_kwargs(self):
        task         = SequentialCB()
        learner      = RecordingLearner(with_kwargs=True, with_info=True)
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

    def test_on_info_kwargs_partial(self):
        task         = SequentialCB()
        learner      = RecordingLearner(with_kwargs=True, with_info=True)
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

    def test_on_record_context(self):
        task                 = SequentialCB(['reward','probability','context'])
        task_without_context = SequentialCB(['reward','probability'])
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

    def test_on_record_discrete_rewards(self):
        task         = SequentialCB(['reward','rewards'])
        learner      = FixedPredLearner([("action_1",None),("action_2",None)])
        interactions = [
            SimulatedInteraction(1, ["action_1", "action_2", "action_3"], [1,0,0]),
            SimulatedInteraction(2, ["action_1", "action_2", "action_3"], [0,2,0]),
        ]

        results = list(task.evaluate(SimpleEnvironment(interactions),learner))
        self.assertListEqual([[1,0,0],[0,2,0]], [result['rewards'] for result in results])
        self.assertListEqual([1,2], [result['reward'] for result in results])

    def test_on_record_continuous_rewards(self):
        task         = SequentialCB(['reward','rewards'])
        learner      = FixedPredLearner([1,3])
        interactions = [
            SimulatedInteraction(1, [], L1Reward(1)),
            SimulatedInteraction(2, [], L1Reward(3)),
        ]

        results = list(task.evaluate(SimpleEnvironment(interactions),learner))
        self.assertListEqual([L1Reward(1),L1Reward(3)], [result['rewards'] for result in results])
        self.assertListEqual([0,0], [result['reward'] for result in results])

    def test_None_ips_score_continuous(self):
        class MyLearner:
            def score(self,context,actions,action):
                return .5
            def predict(self, context, actions):
                return {'action_prob':(2, 0.5)}

        task    = SequentialCB(learn=None,eval='ips')
        learner = MyLearner()
        interactions = [
            LoggedInteraction(1, 2, 3,actions=[]),
            LoggedInteraction(2, 3, 4,actions=[]),
        ]

        task_results = list(task.evaluate(SimpleEnvironment(interactions),learner))

        self.assertEqual(1.5,task_results[0]['reward'])
        self.assertEqual(2.0,task_results[1]['reward'])

    def test_off_None_no_actions_no_rewards(self):
        task    = SequentialCB(learn='off',eval=None)
        learner = RecordingLearner(with_kwargs=False, with_info=False)
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

    def test_off_ips_actions_no_prob(self):
        task    = SequentialCB(learn='off',eval='ips')
        learner = RecordingLearner(with_kwargs=False,with_info=False)
        interactions = [
            LoggedInteraction(1, 2, 3, actions=[2,5,8]),
            LoggedInteraction(2, 3, 4, actions=[3,6,9]),
            LoggedInteraction(3, 4, 5, actions=[4,7,0])
        ]

        task_results = list(task.evaluate(SimpleEnvironment(interactions),learner))

        expected_predict_calls   = [(1,[2,5,8]),(2,[3,6,9]),(3,[4,7,0])]
        expected_predict_returns = [[1,0,0],[0,1,0],[0,0,1]]
        expected_learn_calls     = [(1,2,3,None,{}),(2,3,4,None,{}),(3,4,5,None,{})]
        expected_task_results    = [{'reward': 3.0}, {'reward': 0.0}, {'reward': 0.0}]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_off_ips_actions_prob(self):
        task    = SequentialCB(learn='off',eval='ips')
        learner = RecordingLearner(with_kwargs=False,with_info=False)
        interactions = [
            LoggedInteraction(1, 2, 3, probability=.2, actions=[2,5,8]),
            LoggedInteraction(2, 3, 4, probability=.3, actions=[3,6,9]),
            LoggedInteraction(3, 4, 5, probability=.4, actions=[4,7,0])
        ]

        task_results = list(task.evaluate(SimpleEnvironment(interactions),learner))

        expected_predict_calls   = [(1,[2,5,8]),(2,[3,6,9]),(3,[4,7,0])]
        expected_predict_returns = [[1,0,0],[0,1,0],[0,0,1]]
        expected_learn_calls     = [(1,2,3,.2,{}),(2,3,4,.3,{}),(3,4,5,.4,{})]
        expected_task_results    = [{'reward':3/.2},{'reward':0},{'reward':0}]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_off_ips_actions_prob_kwargs_info_extra(self):
        task    = SequentialCB(learn='off',eval='ips')
        learner = RecordingLearner(with_kwargs=True,with_info=True)
        interactions = [
            LoggedInteraction(1, 2, 3, probability=.2, actions=[2,5,8], L='a'),
            LoggedInteraction(2, 3, 4, probability=.3, actions=[3,6,9], L='b'),
            LoggedInteraction(3, 4, 5, probability=.4, actions=[4,7,0], L='c')
        ]

        task_results = list(task.evaluate(SimpleEnvironment(interactions),learner))

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

    @unittest.skipUnless(PackageChecker.vowpalwabbit(strict=False), "VW is not installed")
    def test_off_dr_actions_prob_kwargs_info_extra(self):
        task    = SequentialCB(learn='off',eval='dr')
        learner = RecordingLearner(with_kwargs=True,with_info=True)
        interactions = [
            LoggedInteraction(1, 2, 3, probability=.2, actions=[2,5,8], L='a'),
            LoggedInteraction(2, 3, 4, probability=.3, actions=[3,6,9], L='b'),
            LoggedInteraction(3, 4, 5, probability=.4, actions=[4,7,0], L='c')
        ]

        actual_task_results      = list(task.evaluate(SimpleEnvironment(interactions),learner))

        expected_rewards         = [ i['rewards'](action) for i,action in zip(OpeRewards("DR",features=[1,'a','xa']).filter(interactions),[2,6,0]) ]
        expected_predict_calls   = [(1,[2,5,8]),(2,[3,6,9]),(3,[4,7,0])]
        expected_predict_returns = [([1,0,0],{'i':1}),([0,1,0],{'i':2}),([0,0,1],{'i':3})]
        expected_learn_calls     = [(1,2,3,.2,{}),(2,3,4,.3,{}),(3,4,5,.4,{})]
        expected_task_results    = [
            {'reward':expected_rewards[0], 'learn':1, 'predict':1, 'L':'a'},
            {'reward':expected_rewards[1], 'learn':2, 'predict':2, 'L':'b'},
            {'reward':expected_rewards[2], 'learn':3, 'predict':3, 'L':'c'}
        ]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, actual_task_results)

    def test_on_record_time(self):
        task         = SequentialCB(['time'])
        learner      = RecordingLearner()
        interactions = [SimulatedInteraction(1,[0,1,2],DiscreteReward([0,1,2],[7,8,9]))]

        task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        self.assertAlmostEqual(0, task_results[0]["predict_time"], places=1)
        self.assertAlmostEqual(0, task_results[0]["learn_time"]  , places=1)

    def test_off_ips_record_time(self):
        task         = SequentialCB(learn='off',eval='ips',record=['time'])
        learner      = RecordingLearner()
        interactions = [LoggedInteraction(1, 2, 3, actions=[2,5,8], probability=.2)]

        task_results = list(task.evaluate(SimpleEnvironment(interactions),learner))

        self.assertAlmostEqual(0, task_results[0]["predict_time"], places=1)
        self.assertAlmostEqual(0, task_results[0]["learn_time"]  , places=1)

    @unittest.skipUnless(PackageChecker.vowpalwabbit(strict=False), "VW is not installed")
    def test_off_ips_record_ope_loss(self):
        task         = SequentialCB(learn='off',eval='ips',record=['ope_loss'])
        interactions = [
            LoggedInteraction(1, 0, 0, actions=[0, 1], probability=1.0),
            LoggedInteraction(2, 0, 1, actions=[0, 1], probability=1.0),
            LoggedInteraction(3, 0, 0, actions=[0, 1], probability=1.0),
        ]

        #VW learner
        task_results = list(task.evaluate(SimpleEnvironment(interactions),VowpalSoftmaxLearner()))
        ope_losses = [result['ope_loss'] for result in task_results]
        self.assertListEqual(ope_losses, [0.0, -1.0, -1.0])

        # Non-VW learner
        task_results = list(task.evaluate(SimpleEnvironment(interactions),RecordingLearner()))
        self.assertTrue(all([math.isnan(result['ope_loss']) for result in task_results]))

    def test_on_batched(self):

        class SimpleLearner:
            def __init__(self) -> None:
                self.predict_call = []
            def predict(self,*args):
                self.predict_call.append(args)
                return [[1,0,0],[0,1,0],[0,0,1]][:len(args[0])]
            def learn(self,*args):
                self.learn_call = args

        task         = SequentialCB()
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

    def test_on_batched_record_discrete_rewards(self):
        task         = SequentialCB(['reward','rewards'])
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

    def test_on_batched_record_continuous_rewards(self):
        task         = SequentialCB(['reward','rewards'])
        learner      = FixedPredLearner([{'action_prob':[(1,None),(3,None)]}])
        interactions = [
            SimulatedInteraction(1, [], L1Reward(1)),
            SimulatedInteraction(2, [], L1Reward(1)),
        ]

        results = list(task.evaluate(SimpleEnvironment(Batch(2).filter(interactions)), learner))

        self.assertEqual(2 , len(results))
        self.assertEqual(0 , results[0]['reward'])
        self.assertEqual(-2, results[1]['reward'])
        self.assertEqual(L1Reward(1),results[0]['rewards'])
        self.assertEqual(L1Reward(1),results[1]['rewards'])

    def test_off_ips_batched_no_score(self):
        task                 = SequentialCB(learn='off',eval='ips')
        learner              = BatchFixedLearner()
        interactions         = [
            LoggedInteraction(1, "action_1", 1, probability=1.0, actions=["action_1", "action_2", "action_3"]),
            LoggedInteraction(2, "action_2", 2, probability=1.0, actions=["action_1", "action_2", "action_3"])
        ]

        #with self.assertRaises(CobaException):
        task_results = list(task.evaluate(SimpleEnvironment(Batch(2).filter(interactions)), learner))

        self.assertEqual(2, len(task_results))
        self.assertEqual(1,task_results[0]['reward'])
        self.assertEqual(0,task_results[1]['reward'])

    def test_None_ips_batched_score_discrete(self):
        class TestLearner:
            probs = [0,1]
            def score(self, context, actions, action):
                if action == 'action_1':
                    return TestLearner.probs.pop()
                if action == 'action_2':
                    return TestLearner.probs.pop()
                raise Exception()

        task                 = SequentialCB(learn=None,eval='ips')
        learner              = TestLearner()
        interactions         = [
            LoggedInteraction(1, "action_1", 1, probability=1.0, actions=["action_1", "action_2", "action_3"]),
            LoggedInteraction(2, "action_2", 2, probability=1.0, actions=["action_1", "action_2", "action_3"])
        ]

        task_results = list(task.evaluate(SimpleEnvironment(Batch(2).filter(interactions)),learner))

        self.assertEqual(2, len(task_results))
        self.assertEqual(1,task_results[0]['reward'])
        self.assertEqual(0,task_results[1]['reward'])

    def test_None_ips_batched_score_continuous(self):
        class TestLearner:
            def score(self,context,actions,action):
                if is_batch(context):
                    raise Exception()
                return 0.5

            def predict(self, context, actions):
                return [(2, 0.5, None), (3, 0.5, None)]

        task    = SequentialCB(learn=None,eval='ips')
        learner = TestLearner()
        interactions = [
            LoggedInteraction(1, 2, 3,actions=[]),
            LoggedInteraction(2, 3, 4,actions=[]),
        ]

        task_results = list(task.evaluate(SimpleEnvironment(Batch(2).filter(interactions)),learner))

        self.assertEqual(1.5,task_results[0]['reward'])
        self.assertEqual(2.0,task_results[1]['reward'])

class SequentialIGL_Tests(unittest.TestCase):
    def test_params(self):
        self.assertEqual(SequentialIGL().params,{'seed':None})

    def test_empty_interaction(self):
        task    = SequentialIGL()
        learner = DummyIglLearner([[1,0,0],[0,0,1]])
        interactions = []
        actual_results = list(task.evaluate(SimpleEnvironment(interactions),learner))
        self.assertEqual(actual_results, [])

    def test_two_grounded_interactions_missing_context(self):
        task    = SequentialIGL()
        learner = DummyIglLearner([[1,0,0],[0,0,1]])
        interactions = [
            GroundedInteraction(None,[1,2,3],[1,0,0],[4,5,6],userid=0,isnormal=False),
            GroundedInteraction(None,[1,2,3],[0,1,0],[7,8,9],userid=1,isnormal=True),
        ]

        for interaction in interactions:
            del interaction['context']

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

    def test_two_grounded_interactions_none_context(self):
        task    = SequentialIGL()
        learner = DummyIglLearner([[1,0,0],[0,0,1]])
        interactions = [
            GroundedInteraction(None,[1,2,3],[1,0,0],[4,5,6],userid=0,isnormal=False),
            GroundedInteraction(None,[1,2,3],[0,1,0],[7,8,9],userid=1,isnormal=True),
        ]

        expected_predict_calls = [
            ((0,None),[1,2,3]),
            ((1,None),[1,2,3])
        ]

        expected_learn_calls = [
            ((0,None),1,4,1,{}),
            ((1,None),3,9,1,{})
        ]

        expected_results = [
            dict(reward=1,feedback=4,userid=0,isnormal=False,n_predict=1),
            dict(reward=0,feedback=9,userid=1,isnormal=True ,n_predict=2)
        ]

        actual_results = list(task.evaluate(SimpleEnvironment(interactions),learner))

        self.assertEqual(expected_results, actual_results)
        self.assertEqual(expected_predict_calls, learner._predict_calls)
        self.assertEqual(expected_learn_calls, learner._learn_calls)

    def test_two_grounded_interactions_dense_context(self):
        task    = SequentialIGL()
        learner = DummyIglLearner([[1,0,0],[0,0,1]])
        interactions = [
            GroundedInteraction([2],[1,2,3],[1,0,0],[4,5,6],userid=0,isnormal=False),
            GroundedInteraction([3],[1,2,3],[0,1,0],[7,8,9],userid=1,isnormal=True),
        ]

        expected_predict_calls = [
            ((0,2),[1,2,3]),
            ((1,3),[1,2,3])
        ]

        expected_learn_calls = [
            ((0,2),1,4,1,{}),
            ((1,3),3,9,1,{})
        ]

        expected_results = [
            dict(reward=1,feedback=4,userid=0,isnormal=False,n_predict=1),
            dict(reward=0,feedback=9,userid=1,isnormal=True ,n_predict=2)
        ]

        actual_results = list(task.evaluate(SimpleEnvironment(interactions),learner))

        self.assertEqual(expected_results, actual_results)
        self.assertEqual(expected_predict_calls, learner._predict_calls)
        self.assertEqual(expected_learn_calls, learner._learn_calls)

    def test_two_grounded_interactions_sparse_context(self):
        task    = SequentialIGL()
        learner = DummyIglLearner([[1,0,0],[0,0,1]])
        interactions = [
            GroundedInteraction({'a':1},[1,2,3],[1,0,0],[4,5,6],userid=0,isnormal=False),
            GroundedInteraction({'b':2},[1,2,3],[0,1,0],[7,8,9],userid=1,isnormal=True),
        ]

        expected_predict_calls = [
            ({'userid':0,'a':1},[1,2,3]),
            ({'userid':1,'b':2},[1,2,3])
        ]

        expected_learn_calls = [
            ({'userid':0,'a':1},1,4,1,{}),
            ({'userid':1,'b':2},3,9,1,{})
        ]

        expected_results = [
            dict(reward=1,feedback=4,userid=0,isnormal=False,n_predict=1),
            dict(reward=0,feedback=9,userid=1,isnormal=True ,n_predict=2)
        ]

        actual_results = list(task.evaluate(SimpleEnvironment(interactions),learner))

        self.assertEqual(expected_results, actual_results)
        self.assertEqual(expected_predict_calls, learner._predict_calls)
        self.assertEqual(expected_learn_calls, learner._learn_calls)

class RejectionCB_Tests(unittest.TestCase):

    def test_params(self):
        self.assertEqual(RejectionCB().params,{'ope':None,'cpct':.005,'cmax':1,'cinit':None,'seed':None})

    def test_partial_interactions(self):
        task         = RejectionCB()
        learner      = RecordingLearner(with_kwargs=False,with_info=False)
        interactions = [LoggedInteraction(1, 2, 3)]

        with self.assertRaises(CobaException) as r:
            task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        self.assertIn("`['context', 'action', 'reward', 'actions', 'probability']`",str(r.exception))

    def test_continuous_interactions(self):
        task         = RejectionCB()
        learner      = RecordingLearner(with_kwargs=False,with_info=False)
        interactions = [LoggedInteraction(1, 2, 3, probability=1, actions=[])]

        with self.assertRaises(CobaException) as r:
            task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        self.assertIn("ExplorationEvaluator does not currently support continuous actions",str(r.exception))

    def test_batched_interactions(self):
        task         = RejectionCB()
        learner      = RecordingLearner(with_kwargs=False,with_info=False)
        interactions = list(Batch(2).filter([LoggedInteraction(1, 2, 3, probability=1, actions=[1,2])]*2))

        with self.assertRaises(CobaException) as r:
            task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        self.assertIn("ExplorationEvaluator does not currently support batching",str(r.exception))

    def test_no_score_learner(self):
        task         = RejectionCB()
        learner      = RecordingLearner(with_kwargs=False,with_info=False)
        interactions = [LoggedInteraction(1, 2, 3, probability=1, actions=[1,2])]

        with self.assertRaises(CobaException) as r:
            task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        self.assertIn("ExplorationEvaluator requires Learners to implement a `score` method",str(r.exception))

    def test_probability_one(self):
        class FixedScoreLearner:
            def score(self,context,actions,action):
                return [1,0,0][[1,2,3].index(action)]
            def learn(self,*args):
                pass
        task         = RejectionCB()
        learner      = FixedScoreLearner()
        interactions = [LoggedInteraction(1, 2, 3, probability=1, actions=[1,2,3])]

        task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

    def test_ope(self):

        score_returns = [.05, .25, .05]

        class FixedScoreLearner:
            def score(self,context,actions,action):
                if not isinstance(action,int): raise Exception()
                return score_returns.pop(0)
            def learn(self,*args):
                pass

        task    = RejectionCB(ope='ips',cpct=1,cinit=1,seed=2)
        learner = FixedScoreLearner()

        interactions = [ LoggedInteraction(1, 2, 5, actions=[2,5,8], probability=.25) ] * 3
        task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        self.assertEqual(task_results,[{"reward":3}])

    def test_ope_false(self):

        score_calls = []
        learn_calls = []

        class FixedScoreLearner:
            def score(self,*args):
                if not isinstance(args[-1],int):
                    raise Exception()
                score_calls.append(args)
                if args[-1] == 2: return .25
                if args[-1] == 5: return .25
                if args[-1] == 8: return .5
            def learn(self,*args):
                learn_calls.append(args)
                pass

        task    = RejectionCB(ope=None,cpct=1,cinit=1)
        learner = FixedScoreLearner()

        interactions = [ LoggedInteraction(1, 2, 3, actions=[2,5,8], probability=.25) ] * 6
        task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        expected_score_call   = [(1,[2,5,8],2)] * 6
        expected_learn_calls  = [(1,2,3,.25)] * 6
        expected_task_results = [{'reward': 3.0}] * 6

        self.assertEqual(expected_score_call, score_calls)
        self.assertEqual(expected_learn_calls, learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_record_time(self):

        class FixedScoreLearner:
            def score(self,context,actions,action):
                return [.25,.25,.5][[2,5,8].index(action)]
            def learn(self,*args):
                pass

        task    = RejectionCB(ope=False,cinit=1,cpct=1,record=['reward','time'])
        learner = FixedScoreLearner()

        interactions = [ LoggedInteraction(1, 2, 3, actions=[2,5,8], probability=.25) ]
        task_results = list(task.evaluate(SimpleEnvironment(interactions), learner))

        self.assertIn('predict_time', task_results[0])
        self.assertIn('learn_time', task_results[0])

class Helper_Tests(unittest.TestCase):
    @unittest.skipUnless(PackageChecker.vowpalwabbit(strict=False), "VW is not installed")
    def test_get_ope_loss(self):

        #VW learner
        learner = VowpalSoftmaxLearner()
        learner.learn(1, 1, 1, 1.0)
        self.assertEqual(get_ope_loss(SafeLearner(learner)), -1.0)

        # Non-VW learner
        self.assertTrue(math.isnan(get_ope_loss(SafeLearner(EpsilonBanditLearner()))))

if __name__ == '__main__':
    unittest.main()
