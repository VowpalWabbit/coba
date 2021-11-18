import unittest

from typing import cast, Iterable
from coba.config.core import CobaConfig
from coba.config.loggers import NullLogger

from coba.environments import LambdaSimulation, SimulatedInteraction, Environments, DebugSimulation
from coba.pipes        import Pipe
from coba.learners     import Learner

from coba.experiments.results import Result
from coba.experiments.process import (
    WorkItem, CreateWorkItems, RemoveFinished, 
    ChunkByTask, ChunkBySource, ProcessWorkItems
)

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

class ObserveTask:
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

class CreateWorkItems_Tests(unittest.TestCase):

    def test_two_sim_two_learns(self):
        sim1 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        sim2 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")
        
        tasks = list(CreateWorkItems([sim1,sim2], [lrn1,lrn2], None, None, None).read())

        self.assertEqual(8, len(tasks))

        self.assertEqual(2, len([t for t in tasks if not t.environ and t.learner]) )
        self.assertEqual(2, len([t for t in tasks if t.environ and not t.learner]) )
        self.assertEqual(4, len([t for t in tasks if t.environ and t.learner]) )

class RemoveFinished_Tests(unittest.TestCase):

    def test_three_finished(self):

        restored = Result(lrn_rows=[dict(learner_id=1)], env_rows= [dict(environment_id=0)], int_rows=[dict(environment_id=0,learner_id=1)])

        tasks = [
            WorkItem( (0,), None, None),
            WorkItem( (1,), None, None),
            WorkItem( None, (0,), None),
            WorkItem( None, (1,), None),
            WorkItem( (0,), (0,), None),
            WorkItem( (0,), (1,), None),
            WorkItem( (1,), (0,), None),
            WorkItem( (1,), (1,), None),
        ]

        unfinished_tasks = list(RemoveFinished(restored).filter(tasks))

        self.assertEqual(5, len(unfinished_tasks))

class ChunkBySource_Tests(unittest.TestCase):

    def test_four_groups(self):
        sim1 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        sim2 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))

        tasks = [
            WorkItem( (0,), None, None),
            WorkItem( (1,), None, None),
            WorkItem( None, (0,sim1), None),
            WorkItem( None, (1,sim2), None),
            WorkItem( (0,), (0,sim1), None),
            WorkItem( (0,), (2,sim1), None),
            WorkItem( (1,), (0,sim1), None),
            WorkItem( (1,), (1,sim2), None),
        ]

        groups = list(ChunkBySource().filter(tasks))
        tasks  = list(groups[0])
        
        self.assertEqual(4, len(groups))

    def test_pipe_four_groups(self):
        sim1 = Environments(DebugSimulation()).shuffle([1,2])._environments
        sim2 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))

        tasks = [
            WorkItem( (0,), None, None),
            WorkItem( (1,), None, None),
            WorkItem( None, (0,sim1[0]), None),
            WorkItem( None, (1,sim2), None),
            WorkItem( (0,), (0,sim1[0]), None),
            WorkItem( (0,), (2,sim1[1]), None),
            WorkItem( (1,), (0,sim1[0]), None),
            WorkItem( (1,), (1,sim2), None),
        ]

        groups = list(ChunkBySource().filter(tasks))
        tasks  = list(groups[0])
        
        self.assertEqual(4, len(groups))

class ChunkByTask_Tests(unittest.TestCase):

    def test_eight_groups(self):
        sim1 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        sim2 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))

        tasks = [
            WorkItem( (0,), None, None),
            WorkItem( (1,), None, None),
            WorkItem( None, (0,(sim1,)), None),
            WorkItem( None, (1,(sim2,)), None),
            WorkItem( (0,), (0,(sim1,)), None),
            WorkItem( (0,), (2,(sim1,)), None),
            WorkItem( (1,), (0,(sim1,)), None),
            WorkItem( (1,), (1,(sim2,)), None),
        ]

        groups = list(ChunkByTask().filter(tasks))
        tasks  = list(groups[0])
        
        self.assertEqual(8, len(groups))

class ProcessTasks_Tests(unittest.TestCase):

    def setUp(self) -> None:
        CobaConfig.logger = NullLogger()

    def test_simple(self):

        sim1 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        lrn1 = ModuloLearner("1")
        task = ObserveTask()

        item = WorkItem( (1,lrn1), (1,sim1), task)

        transactions = list(ProcessWorkItems().filter([item]))

        self.assertEqual(len(task.observed[1]), 5)
        self.assertEqual(['I', (1,1), {"_packed":{}}], transactions[0])

    def test_two_eval_tasks_one_source_one_env(self):

        sim1 = CountReadSimulation()

        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")

        task1 = ObserveTask()
        task2 = ObserveTask()

        items = [ WorkItem((0,lrn1),(0,sim1), task1), WorkItem((1,lrn2),(0,sim1), task2) ]

        transactions = list(ProcessWorkItems().filter(items))

        self.assertEqual(len(task1.observed[1]), 1)
        self.assertEqual(len(task2.observed[1]), 1)

        self.assertEqual(task1.observed[1][0].context, 0)
        self.assertEqual(task2.observed[1][0].context, 0)

        self.assertEqual(['I', (0,0), {"_packed":{}}], transactions[0])
        self.assertEqual(['I', (0,1), {"_packed":{}}], transactions[1])

    def test_two_eval_tasks_two_source_two_env(self):

        sim1 = CountReadSimulation()
        sim2 = CountReadSimulation()

        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")

        task1 = ObserveTask()
        task2 = ObserveTask()

        items = [ WorkItem((0,lrn1),(0,sim1), task1), WorkItem((1,lrn2),(1,sim2), task2) ]

        transactions = list(ProcessWorkItems().filter(items))

        self.assertEqual(len(task1.observed[1]), 1)
        self.assertEqual(len(task2.observed[1]), 1)

        self.assertEqual(task1.observed[1][0].context, 0)
        self.assertEqual(task2.observed[1][0].context, 0)

        self.assertIn(['I', (0,0), {"_packed":{}}], transactions)
        self.assertIn(['I', (1,1), {"_packed":{}}], transactions)

    def test_two_eval_tasks_one_source_two_env(self):

        filter = CountFilter()
        src1   = CountReadSimulation()
        sim1   = Pipe.join(src1, [filter])
        sim2   = Pipe.join(src1, [filter])
        lrn1   = ModuloLearner("1")
        lrn2   = ModuloLearner("2")

        task1 = ObserveTask()
        task2 = ObserveTask()

        items = [ WorkItem((0,lrn1),(0,sim1), task1), WorkItem((1,lrn2),(1,sim2), task2) ]

        transactions = list(ProcessWorkItems().filter(items))

        self.assertEqual(len(task1.observed[1]), 1)
        self.assertEqual(len(task2.observed[1]), 1)

        self.assertEqual(task1.observed[1][0].context, (0,0))
        self.assertEqual(task2.observed[1][0].context, (0,1))

        self.assertEqual(['I', (0,0), []], transactions[0])
        self.assertEqual(['I', (1,1), []], transactions[1])

    def test_two_eval_tasks_one_source_two_env(self):

        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")

        task1 = ObserveTask()
        task2 = ObserveTask()

        items = [ WorkItem((0,lrn1), None, task1), WorkItem((1,lrn2), None, task2) ]

        transactions = list(ProcessWorkItems().filter(items))

        self.assertEqual(task1.observed, lrn1)
        self.assertEqual(task2.observed, lrn2)

        self.assertEqual(['L', 0, {}], transactions[0])
        self.assertEqual(['L', 1, {}], transactions[1])

if __name__ == '__main__':
    unittest.main()