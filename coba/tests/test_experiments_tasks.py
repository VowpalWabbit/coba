import json
import unittest
import math
import unittest.mock
import importlib.util
import warnings

from coba.exceptions   import CobaException
from coba.contexts     import CobaContext
from coba.environments import Interaction, SimulatedInteraction, LoggedInteraction, GroundedInteraction, SupervisedSimulation
from coba.environments import Shuffle, Noise, Batch
from coba.primitives   import SequenceReward
from coba.learners     import Learner
from coba.pipes        import Pipes

from coba.experiments import ClassEnvironmentInfo, SimpleEnvironmentInfo, SimpleLearnerInfo, SimpleEvaluation

#for testing purposes
class FixedActionScoreLearner(Learner):
    def __init__(self, actions):
        self._actions = actions

    @property
    def params(self):
        return {"family": "Recording"}

    def predict(self, context, actions):
        return self._actions.pop(0), None

    def learn(self, context, actions, action, reward, probability, **kwargs):
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

    def learn(self, context, actions, action, reward, probability, **kwargs):

        if self._with_log:
            CobaContext.learning_info.update(learn=self._i)

        self.learn_calls.append((context, actions, action, reward, probability, kwargs))

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

class SimpleLearnerInfo_Tests(unittest.TestCase):
    def test_simple(self):
        self.assertEqual({"family":"Recording"}, SimpleLearnerInfo().process(RecordingLearner()))

class SimpleEnvironmentInfo_Tests(unittest.TestCase):

    def test_classification_statistics_dense(self):

        env  = SupervisedSimulation([[1,2],[3,4]]*10,["A","B"]*10)
        task = SimpleEnvironmentInfo()
        ints = list(env.read())

        self.assertEqual({**env.params}, task.process(env,ints))

    def test_environment_pipe_statistics_dense(self):

        env  = Pipes.join(SupervisedSimulation([[1,2],[3,4]]*10,["A","B"]*10), Shuffle(1))
        task = SimpleEnvironmentInfo()
        ints = list(env.read())

        self.assertEqual({**env.params}, task.process(env,ints))

class ClassEnvironmentInfo_Tests(unittest.TestCase):

    def test_classification_statistics_empty_interactions(self):        
        simulation = SupervisedSimulation([],[])
        row        = ClassEnvironmentInfo().process(simulation,simulation.read())
        self.assertEqual(row, {})

    def test_classification_statistics_dense_sans_sklearn(self):
        with unittest.mock.patch('importlib.import_module', side_effect=ImportError()):
            simulation = SupervisedSimulation([[1,2],[3,4]]*10,["A","B"]*10)
            row        = ClassEnvironmentInfo().process(simulation,simulation.read())

            self.assertEqual(2, row["class_count"])
            self.assertEqual(2, row["feature_count"])
            self.assertEqual(0, row["class_imbalance_ratio"])

    def test_classification_statistics_sparse_sans_sklearn(self):
        with unittest.mock.patch('importlib.import_module', side_effect=ImportError()):
            c1 = [{"1":1, "2":2}, "A"]
            c2 = [{"1":3, "2":4}, "B"]

            simulation = SupervisedSimulation(*zip(*[c1,c2]*10))
            row        = ClassEnvironmentInfo().process(simulation,simulation.read())

            self.assertEqual(2, row["class_count"])
            self.assertEqual(2, row["feature_count"])
            self.assertEqual(0, row["class_imbalance_ratio"])

    def test_classification_statistics_encodable_sans_sklearn(self):
        with unittest.mock.patch('importlib.import_module', side_effect=ImportError()):
            c1 = [{"1":1,"2":2}, "A" ]
            c2 = [{"1":3,"2":4}, "B" ]

            simulation = SupervisedSimulation(*zip(*[c1,c2]*10))
            row        = ClassEnvironmentInfo().process(simulation,simulation.read())

            json.dumps(row)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "sklearn is not installed so we must skip this test.")
    def test_classification_statistics_encodable_with_sklearn(self):
        import sklearn.exceptions
        warnings.filterwarnings("ignore", category=sklearn.exceptions.FitFailedWarning)

        simulation = Pipes.join(SupervisedSimulation([[1,2],[3,4]]*10,["A","B"]*10),Noise())
        row        = ClassEnvironmentInfo().process(simulation,simulation.read())

        json.dumps(row)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "sklearn is not installed so we must skip this test.")
    def test_classification_statistics_dense(self):
        import sklearn.exceptions
        warnings.filterwarnings("ignore", category=sklearn.exceptions.FitFailedWarning)

        simulation = Pipes.join(SupervisedSimulation([[1,2],[3,4]]*10,["A","B"]*10),Noise())
        row        = ClassEnvironmentInfo().process(simulation,simulation.read())

        self.assertEqual(2, row["class_count"])
        self.assertEqual(2, row["feature_count"])
        self.assertEqual(0, row["class_imbalance_ratio"])

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "sklearn is not installed so we must skip the sklearn test")
    def test_classification_statistics_sparse(self):
        import sklearn.exceptions
        warnings.filterwarnings("ignore", category=sklearn.exceptions.FitFailedWarning)

        simulation = Pipes.join(SupervisedSimulation([{"1":1,"2":2},{"3":3,"4":4}]*10,["A","B"]*10), Noise())
        row        = ClassEnvironmentInfo().process(simulation,simulation.read())

        self.assertEqual(2, row["class_count"])
        self.assertEqual(4, row["feature_count"])
        self.assertEqual(0, row["class_imbalance_ratio"])

    def test_entropy(self):
        a = [1,2,3,4,5,6]
        b = [1,1,1,1,1,1]
        c = [1,2,1,2,1,2]
        self.assertAlmostEqual(math.log2(len(a)), ClassEnvironmentInfo()._entropy(a))
        self.assertAlmostEqual(                0, ClassEnvironmentInfo()._entropy(b))
        self.assertAlmostEqual(                1, ClassEnvironmentInfo()._entropy(c))

    def test_entropy_normed(self):
        a = [1,2,3,4,5]
        b = [1,1,1,1,1]
        self.assertAlmostEqual(1, ClassEnvironmentInfo()._entropy_normed(a))
        self.assertAlmostEqual(0, ClassEnvironmentInfo()._entropy_normed(b))

    def test_mutual_info(self):
        #mutual info, I(), tells me how many bits of info two random variables convey about eachother
        #entropy, H(), tells me how many bits of info are needed in order to fully know a random variable
        #therefore, if knowing a random variable x tells me everything about y then I(x;y) == H(y)
        #this doesn't mean that y necessarily tells me everything about x (i.e., I(x;y) may not equal H(x))

        #In classification then I don't really care about how much y tells me about x... I do care how much
        #X tells me about y so I(x;y)/H(y) tells me what percentage of necessary bits x gives me about y.
        #This explains explains the equivalent num of attributes feature since \sum I(x;y) must be >= H(y)
        #for the dataset to be solvable.

        #I also care about how much information multiple features of x give me about eachother. That's
        #because if all x tell me about eachother than what might look like a lot of features actually
        #contains very little information. So, in that case I think I want 2*I(x1;x2)/(H(x1)+H(x2)). If
        #H(x1) >> H(x2) and I(x1;x2) == H(x2) then my above formulation will hide the fact that x1 tells
        #me everything about x2 but I think that's ok because it still tells me that taken as a pair there's
        #still a lot of information.

        #how much information does each X give me about y (1-H(y|X)/H(y)==I(y;x)/H(y) with 0 meaning no info)
        #how much information does x2 give about x1? 1-H(x1|x2)/H(x1)

        #x1 tells me z1 bits about y , x2 tells me z2 bits about y 
        #x1 tells me w1 bits about x2, x2 tells me w2 bits about x1

        #in this case knowing x or y tells me nothing about the other so mutual info is 0
        #P(b|a) = P(b)
        #P(a|b) = P(a)
        #H(b) = 0; H(b|a) = 0; H(a,b) = 2; I(b;a) = 0
        #H(a) = 2; H(a|b) = 2; H(a,b) = 2; I(a;b) = 0
        a = [1,2,3,4]
        b = [1,1,1,1]
        self.assertAlmostEqual(0, ClassEnvironmentInfo()._mutual_info(a,b))

        #H(b) = 1; H(b|a) = 0; H(a,b) = 1; I(b;a) = 1
        #H(a) = 1; H(a|b) = 0; H(a,b) = 1; I(a;b) = 1
        a = [1,2,1,2]
        b = [1,1,2,2]
        self.assertAlmostEqual(0, ClassEnvironmentInfo()._mutual_info(a,b))

        a = [1,2,1,2]
        b = [2,1,2,1]
        self.assertAlmostEqual(1, ClassEnvironmentInfo()._mutual_info(a,b))

        a = [1,2,3,4]
        b = [1,2,3,4]
        self.assertAlmostEqual(2, ClassEnvironmentInfo()._mutual_info(a,b))

    def test_dense(self):

        X = [[1,2,3],[4,5,6]]
        self.assertEqual(X, ClassEnvironmentInfo()._dense(X))

        X = [{'a':1}, {'b':2}, {'a':3, 'b':4}]
        self.assertEqual([[1,0],[0,2],[3,4]], ClassEnvironmentInfo()._dense(X))

    def test_bin(self):

        X = [[1,2],[2,3],[3,4],[4,5]]
        self.assertEqual([[0,0],[0,0],[1,1],[1,1]], ClassEnvironmentInfo()._bin(X,2))

        X = [[1,2],[2,3],[3,4],[4,5]]
        self.assertEqual([[0,0],[1,1],[1,1],[2,2]], ClassEnvironmentInfo()._bin(X,3))

        X = [[1,2],[2,3],[3,4],[4,5]]
        self.assertEqual([[0,0],[1,1],[2,2],[3,3]], ClassEnvironmentInfo()._bin(X,4))

    def test_imbalance_ratio_1(self):

        self.assertAlmostEqual(0, ClassEnvironmentInfo()._imbalance_ratio([1,1,2,2]))
        self.assertAlmostEqual(1, ClassEnvironmentInfo()._imbalance_ratio([1,1]))
        self.assertIsNone     (   ClassEnvironmentInfo()._imbalance_ratio([]))

    def test_volume_overlapping_region(self):

        X = [[1,1],[-5,-5],[-1,-1],[5,5]]
        Y = [1,1,2,2] 
        self.assertAlmostEqual(.04, ClassEnvironmentInfo()._volume_overlapping_region(X,Y))

    def test_max_individual_feature_efficiency(self):
        X = [[1,1],[-5,-5],[-1,-1],[5,5]]
        Y = [1,1,2,2]
        self.assertAlmostEqual(.5, ClassEnvironmentInfo()._max_individual_feature_efficiency(X,Y))

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed so we must skip this test.")
    def test_max_individual_feature_efficiency(self):
        X = [[1,1],[-5,-5],[-1,-1],[5,5]]
        Y = [1,1,2,2]
        self.assertAlmostEqual(.5, ClassEnvironmentInfo()._max_individual_feature_efficiency(X,Y))

    @unittest.skipUnless(not importlib.util.find_spec("numpy"), "numpy is installed so we must skip this test.")
    def test_max_individual_feature_efficiency_sans_numpy(self):
        X = [[1,1],[-5,-5],[-1,-1],[5,5]]
        Y = [1,1,2,2]
        self.assertIsNone(ClassEnvironmentInfo()._max_individual_feature_efficiency(X,Y))

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "sklearn is not installed so we must skip this test.")
    def test_max_directional_fisher_discriminant_ratio(self):
        X = [[1,1],[-5,-5],[-1,-1],[5,5]]
        Y = [1,1,2,2]
        self.assertAlmostEqual(.529, ClassEnvironmentInfo()._max_directional_fisher_discriminant_ratio(X,Y), places=3)

class SimpleEvaluation_Tests(unittest.TestCase):

    def test_simulated_interaction_one_metric(self):
        task         = SimpleEvaluation("reward")
        learner      = RecordingLearner(with_info=False, with_log=False)
        interactions = [
            SimulatedInteraction(None,[1,2,3],[7,8,9]),
            SimulatedInteraction(None,[4,5,6],[4,5,6]),
            SimulatedInteraction(None,[7,8,9],[1,2,3]),
        ]

        task_results = list(task.process(learner, interactions))

        expected_predict_calls   = [(None,[1,2,3]),(None,[4,5,6]),(None,[7,8,9])]
        expected_predict_returns = [[1,0,0],[0,1,0],[0,0,1]]
        expected_learn_calls     = [(None,[1,2,3],0,7,1,{}),(None,[4,5,6],1,5,1,{}),(None,[7,8,9],2,3,1,{})]
        expected_task_results    = [
            {"reward":7},
            {"reward":5},
            {"reward":3}
        ]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_simulated_interaction_all_reward_metrics(self):
        task         = SimpleEvaluation(["reward","rank","regret"])
        learner      = RecordingLearner(with_info=False, with_log=False)
        interactions = [
            SimulatedInteraction(None,[1,2,3],[7,8,9]),
            SimulatedInteraction(None,[4,5,6],[4,5,6]),
            SimulatedInteraction(None,[7,8,9],[1,2,3]),
        ]

        task_results = list(task.process(learner, interactions))

        expected_predict_calls   = [(None,[1,2,3]),(None,[4,5,6]),(None,[7,8,9])]
        expected_predict_returns = [[1,0,0],[0,1,0],[0,0,1]]
        expected_learn_calls     = [(None,[1,2,3],0,7,1,{}),(None,[4,5,6],1,5,1,{}),(None,[7,8,9],2,3,1,{})]
        expected_task_results    = [
            {"reward":7,'rank':0,'regret':2},
            {"reward":5,'rank':.5,'regret':1},
            {"reward":3,'rank':1,'regret':0}
        ]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_none_simulated_interaction_no_info_no_logs_no_kwargs(self):

        task         = SimpleEvaluation()
        learner      = RecordingLearner(with_info=False, with_log=False)
        interactions = [
            SimulatedInteraction(None,[1,2,3],[7,8,9]),
            SimulatedInteraction(None,[4,5,6],[4,5,6]),
            SimulatedInteraction(None,[7,8,9],[1,2,3]),
        ]

        task_results = list(task.process(learner, interactions))

        expected_predict_calls   = [(None,[1,2,3]),(None,[4,5,6]),(None,[7,8,9])]
        expected_predict_returns = [[1,0,0],[0,1,0],[0,0,1]]
        expected_learn_calls     = [(None,[1,2,3],0,7,1,{}),(None,[4,5,6],1,5,1,{}),(None,[7,8,9],2,3,1,{})]
        expected_task_results    = [
            {"reward":7},
            {"reward":5},
            {"reward":3}
        ]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_sparse_simulated_interaction_no_info_no_logs_no_kwargs(self):

        task         = SimpleEvaluation()
        learner      = RecordingLearner(with_info=False, with_log=False)
        interactions = [
            SimulatedInteraction({'c':1},[{'a':1},{'a':2},{'a':2}],[7,8,9]),
            SimulatedInteraction({'c':2},[{'a':4},{'a':5},{'a':2}],[4,5,9]),
        ]

        task_results = list(task.process(learner, interactions))

        expected_predict_calls   = [({'c':1},[{'a':1},{'a':2},{'a':2}]),({'c':2},[{'a':4},{'a':5},{'a':2}])]
        expected_predict_returns = [[1,0,0],[0,1,0]]
        expected_learn_calls     = [({'c':1},[{'a':1},{'a':2},{'a':2}],0,7,1,{}),({'c':2},[{'a':4},{'a':5},{'a':2}],1,5,1,{})]
        expected_task_results    = [
            {"reward":7},
            {"reward":5},
        ]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_dense_simulated_interaction_no_info_no_logs_no_kwargs(self):

        task         = SimpleEvaluation()
        learner      = RecordingLearner(with_info=False, with_log=False)
        interactions = [
            SimulatedInteraction(1,[1,2,3],[7,8,9]),
            SimulatedInteraction(2,[4,5,6],[4,5,6]),
            SimulatedInteraction(3,[7,8,9],[1,2,3]),
        ]

        task_results = list(task.process(learner, interactions))

        expected_predict_calls   = [(1,[1,2,3]),(2,[4,5,6]),(3,[7,8,9])]
        expected_predict_returns = [[1,0,0],[0,1,0],[0,0,1]]
        expected_learn_calls     = [(1,[1,2,3],0,7,1,{}),(2,[4,5,6],1,5,1,{}),(3,[7,8,9],2,3,1,{})]
        expected_task_results    = [
            {"reward":7},
            {"reward":5},
            {"reward":3}
        ]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_continuous_simulated_interaction_no_info_no_logs_no_kwargs(self):
        task         = SimpleEvaluation(["reward","rank"])
        learner      = FixedActionScoreLearner([0,1,2])
        interactions = [
            SimulatedInteraction(1,[],SequenceReward([7,8,9])),
            SimulatedInteraction(2,[],SequenceReward([4,5,6])),
            SimulatedInteraction(3,[],SequenceReward([1,2,3])),
        ]
        
        with self.assertWarns(UserWarning) as w:
            task_results = list(task.process(learner, interactions))

        self.assertEqual("The rank metric can only be calculated for discrete environments", str(w.warning))
        
        expected_task_results = [{"reward":7},{"reward":5},{"reward":3}]

        self.assertEqual(expected_task_results, task_results)

    def test_simulated_interaction_info_logs_kwargs(self):

        task         = SimpleEvaluation()
        learner      = RecordingLearner(with_info=True, with_log=True)
        interactions = [
            SimulatedInteraction(1,[1,2,3],[7,8,9],I=1),
            SimulatedInteraction(2,[4,5,6],[4,5,6],I=2),
            SimulatedInteraction(3,[7,8,9],[1,2,3],I=3),
        ]

        task_results = list(task.process(learner, interactions))

        expected_predict_calls   = [(1,[1,2,3]),(2,[4,5,6]),(3,[7,8,9])]
        expected_predict_returns = [([1,0,0],{'i':1}),([0,1,0],{'i':2}),([0,0,1],{'i':3})]
        expected_learn_calls     = [(1,[1,2,3],0,7,1,{'i':1}),(2,[4,5,6],1,5,1,{'i':2}),(3,[7,8,9],2,3,1,{'i':3})]
        expected_task_results    = [
            {"reward":7,'learn':1,'predict':1,'I':1},
            {"reward":5,'learn':2,'predict':2,'I':2},
            {"reward":3,'learn':3,'predict':3,'I':3}
        ]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_simulated_interaction_info_logs_kwargs_partial(self):

        task         = SimpleEvaluation()
        learner      = RecordingLearner(with_info=True, with_log=True)
        interactions = [
            SimulatedInteraction(1,[1,2,3],[7,8,9]),
            SimulatedInteraction(2,[4,5,6],[4,5,6],letter='d'),
            SimulatedInteraction(3,[7,8,9],[1,2,3],letter='g'),
        ]

        task_results = list(task.process(learner, interactions))

        expected_predict_calls   = [(1,[1,2,3]),(2,[4,5,6]),(3,[7,8,9])]
        expected_predict_returns = [([1,0,0],{'i':1}),([0,1,0],{'i':2}),([0,0,1],{'i':3})]
        expected_learn_calls     = [(1,[1,2,3],0,7,1,{'i':1}),(2,[4,5,6],1,5,1,{'i':2}),(3,[7,8,9],2,3,1,{'i':3})]
        expected_task_results    = [
            {"reward":7,'learn':1,'predict':1,            },
            {"reward":5,'learn':2,'predict':2,'letter':'d'},
            {"reward":3,'learn':3,'predict':3,'letter':'g'}
        ]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_logged_interaction_no_actions_no_probability_no_info_no_logs(self):
        task    = SimpleEvaluation()
        learner = RecordingLearner(with_info=False,with_log=False)
        interactions = [
            LoggedInteraction(1, 2, 3),
            LoggedInteraction(2, 3, 4),
            LoggedInteraction(3, 4, 5)
        ]

        task_results = list(task.process(learner, interactions))

        expected_predict_calls   = []
        expected_predict_returns = []
        expected_learn_calls     = [(1,None,2,3,None,{}),(2,None,3,4,None,{}),(3,None,4,5,None,{})]
        expected_task_results    = [{},{},{}]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_logged_interaction_actions_no_probability_no_info_no_logs(self):
        task    = SimpleEvaluation()
        learner = RecordingLearner(with_info=False,with_log=False)
        interactions = [
            LoggedInteraction(1, 2, 3, actions=[2,5,8]),
            LoggedInteraction(2, 3, 4, actions=[3,6,9]),
            LoggedInteraction(3, 4, 5, actions=[4,7,0])
        ]

        task_results = list(task.process(learner, interactions))

        expected_predict_calls   = []
        expected_predict_returns = []
        expected_learn_calls     = [(1,[2,5,8],2,3,None,{}),(2,[3,6,9],3,4,None,{}),(3,[4,7,0],4,5,None,{})]
        expected_task_results    = [{},{},{}]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_logged_interaction_actions_probability_no_info_no_logs(self):
        task    = SimpleEvaluation()
        learner = RecordingLearner(with_info=False,with_log=False)
        interactions = [
            LoggedInteraction(1, 2, 3, probability=.2, actions=[2,5,8]),
            LoggedInteraction(2, 3, 4, probability=.3, actions=[3,6,9]),
            LoggedInteraction(3, 4, 5, probability=.4, actions=[4,7,0])
        ]

        task_results = list(task.process(learner, interactions))

        expected_predict_calls   = [(1,[2,5,8]),(2,[3,6,9]),(3,[4,7,0])]
        expected_predict_returns = [[1,0,0],[0,1,0],[0,0,1]]
        expected_learn_calls     = [(1,[2,5,8],2,3,.2,{}),(2,[3,6,9],3,4,.3,{}),(3,[4,7,0],4,5,.4,{})]
        expected_task_results    = [{'reward':3/.2},{'reward':0},{'reward':0}]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_logged_interaction_actions_probability_info_logs_kwargs(self):
        task    = SimpleEvaluation()
        learner = RecordingLearner(with_info=True,with_log=True)
        interactions = [
            LoggedInteraction(1, 2, 3, probability=.2, actions=[2,5,8], L='a'),
            LoggedInteraction(2, 3, 4, probability=.3, actions=[3,6,9], L='b'),
            LoggedInteraction(3, 4, 5, probability=.4, actions=[4,7,0], L='c')
        ]

        task_results = list(task.process(learner, interactions))

        expected_predict_calls   = [(1,[2,5,8]),(2,[3,6,9]),(3,[4,7,0])]
        expected_predict_returns = [([1,0,0],{'i':1}),([0,1,0],{'i':2}),([0,0,1],{'i':3})]
        expected_learn_calls     = [(1,[2,5,8],2,3,.2,{}),(2,[3,6,9],3,4,.3,{}),(3,[4,7,0],4,5,.4,{})]
        expected_task_results    = [
            {'reward':3/.2, 'learn':1, 'predict':1, 'L':'a'},
            {'reward':0   , 'learn':2, 'predict':2, 'L':'b'},
            {'reward':0   , 'learn':3, 'predict':3, 'L':'c'}
        ]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_logged_and_simulated_interactions(self):
        task         = SimpleEvaluation(['reward','probability'])
        learner      = RecordingLearner(with_info=False, with_log=False)
        interactions = [
            LoggedInteraction(1, 2, 3),
            LoggedInteraction(2, 3, 4),
            LoggedInteraction(3, 4, 5),
            SimulatedInteraction(None,[1,2,3],[7,8,9]),
            SimulatedInteraction(None,[4,5,6],[4,5,6]),
            SimulatedInteraction(None,[7,8,9],[1,2,3]),
        ]

        task_results = list(task.process(learner, interactions))

        expected_predict_calls   = [(None,[1,2,3]),(None,[4,5,6]),(None,[7,8,9])]
        expected_predict_returns = [[1,0,0],[0,1,0],[0,0,1]]
        expected_learn_calls     = [
            (1,None,2,3,None,{}),
            (2,None,3,4,None,{}),
            (3,None,4,5,None,{}),
            (None,[1,2,3],0,7,1,{}),
            (None,[4,5,6],1,5,1,{}),
            (None,[7,8,9],2,3,1,{})
        ]
        expected_task_results = [
            {},
            {},
            {},
            {"reward":7,"probability":1},
            {"reward":5,"probability":1},
            {"reward":3,"probability":1}
        ]

        self.assertEqual(expected_predict_calls, learner.predict_calls)
        self.assertEqual(expected_predict_returns, learner.predict_returns)
        self.assertEqual(expected_learn_calls, learner.learn_calls)
        self.assertEqual(expected_task_results, task_results)

    def test_two_grounded_interactions(self):

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
            (0,[1,2,3],0,4,1,{}),
            (1,[1,2,3],2,9,1,{})
        ]

        expected_results = [
            dict(reward=1,feedback=4,userid=0,isnormal=False,n_predict=1),
            dict(reward=0,feedback=9,userid=1,isnormal=True ,n_predict=2)
        ]

        actual_results = list(SimpleEvaluation().process(learner,interactions))

        self.assertEqual(expected_results, actual_results)
        self.assertEqual(expected_predict_calls, learner._predict_calls)
        self.assertEqual(expected_learn_calls, learner._learn_calls)

    def test_time(self):
        task         = SimpleEvaluation(['time'])
        learner      = RecordingLearner()
        interactions = [LoggedInteraction(1, 2, 3, actions=[2,5,8], probability=.2)]

        task_results = list(task.process(learner, interactions))

        self.assertAlmostEqual(0, task_results[0]["predict_time"], places=1)
        self.assertAlmostEqual(0, task_results[0]["learn_time"]  , places=1)

    def test_batched_simulated_interaction(self):

        class SimpleLearner:
            def __init__(self) -> None:
                self.predict_call = []
            def predict(self,*args):
                self.predict_call.append(args)
                return [[1,0,0],[0,1,0],[0,0,1]][:len(args[0])]
            def learn(self,*args):
                self.learn_call = args

        task         = SimpleEvaluation()
        learner      = SimpleLearner()
        interactions = [
            SimulatedInteraction(1,[1,2,3],[7,8,9]),
            SimulatedInteraction(2,[4,5,6],[4,5,6]),
            SimulatedInteraction(3,[7,8,9],[1,2,3]),
        ]

        task_results = list(task.process(learner, Batch(3).filter(interactions)))

        expected_predict_call = ([1,2,3],[[1,2,3],[4,5,6],[7,8,9]])
        expected_learn_call   = ([1,2,3],[[1,2,3],[4,5,6],[7,8,9]],[0,1,2],[7,5,3],[1,1,1])
        expected_task_results  = [ {"reward":5 } ]

        self.assertEqual(expected_predict_call, learner.predict_call[0])
        self.assertEqual(expected_learn_call, learner.learn_call)
        self.assertEqual(expected_task_results, task_results)

if __name__ == '__main__':
    unittest.main()
