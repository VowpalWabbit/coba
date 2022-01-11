import unittest

from typing import cast, Iterable

from coba.contexts     import CobaContext, NullLogger
from coba.environments import LambdaSimulation, SimulatedInteraction, Environments, LinearSyntheticSimulation
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

    def process(self, *items):
        self.observed = items
        return {}

class ExceptionTask:
    def __init__(self):
        self.observed = []

    def process(self, *items):
        raise Exception()

class CountReadSimulation:
    def __init__(self) -> None:
        self._reads = 0

    def read(self) -> Iterable[SimulatedInteraction]:
        yield SimulatedInteraction(self._reads, [0,1], rewards=[0,1])
        self._reads += 1

class ExceptionSimulation:
    def read(self) -> Iterable[SimulatedInteraction]:
        raise Exception()

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

        restored = Result(0, 0, lrn_rows={1:{}}, env_rows= {0:{}}, int_rows= {(0,1):{}})

        tasks = [
            WorkItem(None, 0, None, None, None),
            WorkItem(None, 1, None, None, None),
            WorkItem(0, None, None, None, None),
            WorkItem(1, None, None, None, None),
            WorkItem(0, 0, None, None, None),
            WorkItem(1, 0, None, None, None),
            WorkItem(0, 1, None, None, None),
            WorkItem(1, 1, None, None, None),
        ]

        unfinished_tasks = list(RemoveFinished(restored).filter(tasks))

        self.assertEqual(5, len(unfinished_tasks))

class ChunkBySource_Tests(unittest.TestCase):

    def test_four_groups(self):
        sim1 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        sim2 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))

        tasks = [
            WorkItem(None, 0, None, None, None),
            WorkItem(None, 1, None, None, None),
            WorkItem(1, None, sim2, None, None),
            WorkItem(0, None, sim1, None, None),
            WorkItem(0, 0, sim1, None, None),
            WorkItem(2, 0, sim1, None, None),
            WorkItem(0, 1, sim1, None, None),
            WorkItem(1, 1, sim2, None, None)
        ]

        groups = list(ChunkBySource().filter(tasks))

        self.assertEqual(len(groups), 4)
        self.assertEqual(groups[0], tasks[0:1])
        self.assertEqual(groups[1], tasks[1:2])
        self.assertEqual(groups[2], [tasks[3],tasks[4],tasks[6],tasks[5]])
        self.assertEqual(groups[3], [tasks[2],tasks[7]])

    def test_pipe_four_groups(self):
        sim1 = Environments(LinearSyntheticSimulation()).shuffle([1,2])._environments
        sim2 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))

        tasks = [
            WorkItem(None, 0, None, None, None),
            WorkItem(None, 1, None, None, None),
            WorkItem(1, None, sim2, None, None),
            WorkItem(0, None, sim1[0], None, None),            
            WorkItem(0, 0, sim1[0], None, None),
            WorkItem(2, 0, sim1[1], None, None),
            WorkItem(0, 1, sim1[0], None, None),
            WorkItem(1, 1, sim2, None, None)
        ]

        groups = list(ChunkBySource().filter(tasks))

        self.assertEqual(len(groups), 4)
        self.assertEqual(groups[0], tasks[0:1])
        self.assertEqual(groups[1], tasks[1:2])
        self.assertEqual(groups[2], [tasks[3],tasks[4],tasks[6],tasks[5]])
        self.assertEqual(groups[3], [tasks[2],tasks[7]])

class ChunkByTask_Tests(unittest.TestCase):

    def test_eight_groups(self):
        sim1 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        sim2 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))

        tasks = [
            WorkItem(None, 0, None, None, None),
            WorkItem(None, 1, None, None, None),
            WorkItem(1, None, sim2, None, None),
            WorkItem(0, None, sim1, None, None),
            WorkItem(1, 0, sim1, None, None),
            WorkItem(1, 1, sim1, None, None),
            WorkItem(0, 0, sim1, None, None),
            WorkItem(0, 1, sim2, None, None)
        ]

        groups = list(ChunkByTask().filter(tasks))

        self.assertEqual(len(groups), 8)
        self.assertEqual(groups[0], tasks[0:1])
        self.assertEqual(groups[1], tasks[1:2])
        self.assertEqual(groups[2], tasks[3:4])
        self.assertEqual(groups[3], tasks[6:7])
        self.assertEqual(groups[4], tasks[7:8])
        self.assertEqual(groups[5], tasks[2:3])
        self.assertEqual(groups[6], tasks[4:5])
        self.assertEqual(groups[7], tasks[5:6])

class ProcessTasks_Tests(unittest.TestCase):

    def setUp(self) -> None:
        CobaContext.logger = NullLogger()

    def test_simple(self):

        sim1 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        lrn1 = ModuloLearner("1")
        task = ObserveTask()

        item = WorkItem(1, 1, sim1, lrn1, task)

        transactions = list(ProcessWorkItems().filter([item]))

        self.assertEqual(len(task.observed[1]), 5)
        self.assertEqual(['T3', (1,1), []], transactions[0])

    def test_two_eval_tasks_one_source_one_env(self):

        sim1 = CountReadSimulation()

        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")

        task1 = ObserveTask()
        task2 = ObserveTask()

        items = [ WorkItem(0, 0, sim1, lrn1, task1), WorkItem(0, 1, sim1, lrn2, task2) ]

        transactions = list(ProcessWorkItems().filter(items))

        self.assertEqual(len(task1.observed[1]), 1)
        self.assertEqual(len(task2.observed[1]), 1)

        self.assertEqual(task1.observed[1][0].context, 0)
        self.assertEqual(task2.observed[1][0].context, 0)

        self.assertEqual(['T3', (0,0), []], transactions[0])
        self.assertEqual(['T3', (0,1), []], transactions[1])

    def test_two_eval_tasks_two_source_two_env(self):

        sim1 = CountReadSimulation()
        sim2 = CountReadSimulation()

        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")

        task1 = ObserveTask()
        task2 = ObserveTask()

        items = [ WorkItem(0, 0, sim1, lrn1, task1), WorkItem(1, 1, sim2, lrn2, task2) ]

        transactions = list(ProcessWorkItems().filter(items))

        self.assertEqual(len(task1.observed[1]), 1)
        self.assertEqual(len(task2.observed[1]), 1)

        self.assertEqual(task1.observed[1][0].context, 0)
        self.assertEqual(task2.observed[1][0].context, 0)

        self.assertIn(['T3', (0,0), []], transactions)
        self.assertIn(['T3', (1,1), []], transactions)

    def test_two_eval_tasks_one_source_two_env(self):

        filter = CountFilter()
        src1   = CountReadSimulation()
        sim1   = Pipe.join(src1, [filter])
        sim2   = Pipe.join(src1, [filter])
        lrn1   = ModuloLearner("1")
        lrn2   = ModuloLearner("2")

        task1 = ObserveTask()
        task2 = ObserveTask()

        items = [ WorkItem(0, 0, sim1, lrn1, task1), WorkItem(1, 1, sim2, lrn2, task2) ]

        transactions = list(ProcessWorkItems().filter(items))

        self.assertEqual(len(task1.observed[1]), 1)
        self.assertEqual(len(task2.observed[1]), 1)

        self.assertEqual(task1.observed[1][0].context, (0,0))
        self.assertEqual(task2.observed[1][0].context, (0,1))

        self.assertEqual(['T3', (0,0), []], transactions[0])
        self.assertEqual(['T3', (1,1), []], transactions[1])

    def test_two_learn_tasks(self):

        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")

        task1 = ObserveTask()
        task2 = ObserveTask()

        items = [ WorkItem(None, 0, None, lrn1, task1), WorkItem(None, 1, None, lrn2, task2) ]

        transactions = list(ProcessWorkItems().filter(items))

        self.assertNotEqual(task1.observed[0], (lrn1,))
        self.assertNotEqual(task2.observed[0], (lrn2,))

        self.assertEqual(task1.observed[0]._param, "1")
        self.assertEqual(task2.observed[0]._param, "2")

        self.assertEqual(['T1', 0, {}], transactions[0])
        self.assertEqual(['T1', 1, {}], transactions[1])

    def test_two_environment_tasks(self):

        filter = CountFilter()
        src1   = CountReadSimulation()
        sim1   = Pipe.join(src1, [filter])
        sim2   = Pipe.join(src1, [filter])

        task1 = ObserveTask()
        task2 = ObserveTask()

        items = [ WorkItem(0, None, sim1, None, task1), WorkItem(1, None, sim2, None, task2) ]

        transactions = list(ProcessWorkItems().filter(items))

        self.assertEqual(len(task1.observed[1]), 1)
        self.assertEqual(len(task2.observed[1]), 1)

        self.assertEqual(task1.observed[1][0].context, (0,0))
        self.assertEqual(task2.observed[1][0].context, (0,1))

        self.assertEqual(['T2', 0, {}], transactions[0])
        self.assertEqual(['T2', 1, {}], transactions[1])

    def test_two_environment_tasks_first_environment_is_empty(self):

        src1   = LinearSyntheticSimulation(n_interactions=0)
        src2   = LinearSyntheticSimulation(n_interactions=5)

        task1 = ObserveTask()
        task2 = ObserveTask()

        items = [ WorkItem(0, None, src1, None, task1), WorkItem(1, None, src2, None, task2) ]

        transactions = list(ProcessWorkItems().filter(items))

        self.assertEqual(len(task1.observed), 0)
        self.assertEqual(len(task2.observed[1]), 5)

    def test_empty_environment_tasks(self):

        src1  = LinearSyntheticSimulation(0)
        task1 = ObserveTask()

        items = [ WorkItem(0, None, src1, None, task1) ]

        transactions = list(ProcessWorkItems().filter(items))

        self.assertEqual(len(task1.observed), 0)
        self.assertEqual(len(transactions)  , 0)

    def test_exception_during_read_simulation(self):

        src1  = ExceptionSimulation()
        task1 = ObserveTask()

        items = [ WorkItem(0, None, src1, None, task1) ]

        transactions = list(ProcessWorkItems().filter(items))

        self.assertEqual(len(task1.observed), 0)
        self.assertEqual(len(transactions)  , 0)

    def test_exception_during_task_process(self):

        src1  = CountReadSimulation()

        task1 = ExceptionTask()
        task2 = ObserveTask()

        items = [ WorkItem(0, None, src1, None, task1), WorkItem(1, None, src1, None, task2) ]

        transactions = list(ProcessWorkItems().filter(items))

        self.assertEqual(len(task2.observed[1]), 1)

        self.assertEqual(task2.observed[1][0].context, 0)

        self.assertEqual(['T2', 1, {}], transactions[0])

if __name__ == '__main__':
    unittest.main()