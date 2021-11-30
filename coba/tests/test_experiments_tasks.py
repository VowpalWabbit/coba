import unittest

import json
from typing import Iterable

from coba.environments import SimulatedInteraction, ClassificationSimulation
from coba.learners     import Learner, RandomLearner

from coba.experiments.tasks import OnPolicyEvaluationTask, EvaluationTask, ClassEnvironmentTask, SimpleEnvironmentTask

#for testing purposes
class ModuloLearner(Learner):
    def __init__(self, param:str="0"):
        self._param = param

    @property
    def params(self):
        return {"family": "Modulo", "p": self._param}

    def predict(self, key, context, actions):
        return [ int(i == actions.index(actions[context%len(actions)])) for i in range(len(actions)) ]

    def learn(self, key, context, action, reward, probability):
        pass

class ObserveTask(EvaluationTask):
    def __init__(self):
        self.observed = []

    def filter(self, item):
        self.observed = item
        return {}

class CountReadSimulation:
    def __init__(self) -> None:
        self._reads = 0

    def read(self) -> Iterable[SimulatedInteraction]:
        yield SimulatedInteraction(self._reads, [0,1], rewards=[0,1])
        self._reads += 1

class CountFilter:
    def __init__(self) -> None:
        self._count = 0

    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[SimulatedInteraction]:
        for interaction in interactions:
            yield SimulatedInteraction((interaction.context, self._count), interaction.actions, **interaction.kwargs)

        self._count += 1
#for testing purposes

class SimpleEnvironmentTask_Tests(unittest.TestCase):

    def test_classification_statistics_dense(self):

        simulation = ClassificationSimulation([[[1,2],"A"],[[3,4],"B"]]*10)
        task       = SimpleEnvironmentTask()

        self.assertEqual({'source':'ClassificationSimulation'}, task.process(simulation,simulation.read()))

class ClassEnvironmentTask_Tests(unittest.TestCase):

    def test_classification_statistics_dense(self):

        try:
            import sklearn
        except:
            raise unittest.SkipTest("sklearn not installed. Skip this Test")
        else:
            sklearn_installed = True

        simulation = ClassificationSimulation([[[1,2],"A"],[[3,4],"B"]]*10)
        row        = ClassEnvironmentTask().process(simulation,simulation.read())

        self.assertEqual(2, row["action_cardinality"])
        self.assertEqual(2, row["context_dimensions"])
        self.assertEqual(1, row["imbalance_ratio"])

        if sklearn_installed:
            self.assertEqual(1, row["bayes_rate_avg"])
            self.assertEqual(0, row["bayes_rate_iqr"])
            self.assertEqual(1, row["centroid_purity"])
            self.assertEqual(0, row["centroid_distance"])

    def test_classification_statistics_sparse1(self):
        
        try:
            import sklearn
        except:
            raise unittest.SkipTest("sklearn not installed. Skip this Test")
        else:
            sklearn_installed = True

        c1 = [{"1":1, "2":2}, "A"]
        c2 = [{"1":3, "2":4}, "B"]

        simulation = ClassificationSimulation([c1,c2]*10)
        row        = ClassEnvironmentTask().process(simulation,simulation.read())

        self.assertEqual(2, row["action_cardinality"])
        self.assertEqual(2, row["context_dimensions"])
        self.assertEqual(1, row["imbalance_ratio"])

        if sklearn_installed:
            self.assertEqual(1, row["bayes_rate_avg"])
            self.assertEqual(0, row["bayes_rate_iqr"])
            self.assertEqual(1, row["centroid_purity"])
            self.assertEqual(0, row["centroid_distance"])

    def test_classification_statistics_encodable(self):
        try:
            import sklearn
        except:
            raise unittest.SkipTest("sklearn not installed. Skip this Test")
        else:
            sklearn_installed = True

        c1 = [{"1":1, "2":2 }, "A" ]
        c2 = [{"1":3, "2":4 }, "B" ]

        simulation = ClassificationSimulation([c1,c2]*10)
        row        = ClassEnvironmentTask().process(simulation,simulation.read())

        json.dumps(row)

class OnPolicyEvaluationTask_Tests(unittest.TestCase):

    def test_simple(self):

        task = OnPolicyEvaluationTask()

        rows = list(task.process(RandomLearner(),[
            SimulatedInteraction(1,[1,2,3],rewards=[4,5,6]),
            SimulatedInteraction(1,[1,2,3],rewards=[4,5,6]),
            SimulatedInteraction(1,[1,2,3],rewards=[4,5,6]),
            SimulatedInteraction(1,[1,2,3],rewards=[4,5,6]),
        ]))

        self.assertEqual([{'rewards':reward} for reward in [4,6,5,4]], rows)

    def test_reveals_results(self):
        task = OnPolicyEvaluationTask()

        rows = list(task.process(RandomLearner(),[
            SimulatedInteraction(1,[1,2,3],reveals=[4,5,6],rewards=[1,2,3]),
            SimulatedInteraction(1,[1,2,3],reveals=[4,5,6],rewards=[4,5,6]),
            SimulatedInteraction(1,[1,2,3],reveals=[4,5,6],rewards=[7,8,9]),
            SimulatedInteraction(1,[1,2,3],reveals=[4,5,6],rewards=[0,1,2]),
        ]))

        self.assertEqual([{'reveals':rev, 'rewards':rwd} for rev,rwd in zip([4,6,5,4],[1,6,8,0])], rows)

    def test_partial_extras(self):
        task = OnPolicyEvaluationTask()

        actual = list(task.process(RandomLearner(),[
            SimulatedInteraction(1,[1,2,3],rewards=[1,2,3]),
            SimulatedInteraction(1,[1,2,3],rewards=[4,5,6], extra=[2,3,4]),
            SimulatedInteraction(1,[1,2,3],rewards=[7,8,9], extra=[2,3,4]),
            SimulatedInteraction(1,[1,2,3],rewards=[0,1,2], extra=[2,3,4]),
        ]))

        expected = [ {'rewards':1}, {'rewards':6, 'extra':4}, {'rewards':8, 'extra':3}, {'rewards':0, 'extra':2} ]

        self.assertEqual(expected, actual)

    def test_sparse_actions(self):
        task = OnPolicyEvaluationTask()

        rows = list(task.process(RandomLearner(),[
            SimulatedInteraction(1,[{'a':1},{'b':2},{'c':3}],reveals=[4,5,6],rewards=[1,2,3]),
            SimulatedInteraction(1,[{'a':1},{'b':2},{'c':3}],reveals=[4,5,6],rewards=[4,5,6]),
            SimulatedInteraction(1,[{'a':1},{'b':2},{'c':3}],reveals=[4,5,6],rewards=[7,8,9]),
            SimulatedInteraction(1,[{'a':1},{'b':2},{'c':3}],reveals=[4,5,6],rewards=[0,1,2]),
        ]))

        self.assertEqual([{'reveals':rev, 'rewards':rwd} for rev,rwd in zip([4,6,5,4],[1,6,8,0])], rows)

if __name__ == '__main__':
    unittest.main()
