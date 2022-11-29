import unittest

from typing import cast, Iterable

from coba.contexts     import CobaContext, BasicLogger
from coba.environments import LambdaSimulation, SimulatedInteraction, Environments, LinearSyntheticSimulation
from coba.pipes        import Pipes, ListSink
from coba.learners     import Learner

from coba.experiments.results import Result
from coba.experiments.process import (
    WorkItem, CreateWorkItems, RemoveFinished,
    ChunkByTask, ChunkBySource, ProcessWorkItems,
    MaxChunkSize
)

#for testing purposes
class ModuloLearner(Learner):
    def __init__(self, param:str="0"):
        self._param = param
        self.n_learns = 0

    @property
    def params(self):
        return {"family": "Modulo", "p": self._param}

    def predict(self, key, context, actions):
        return [ int(i == actions.index(actions[context%len(actions)])) for i in range(len(actions)) ]

    def learn(self, key, context, action, reward, probability):
        self.n_learns += 1

class ObserveTask:
    def __init__(self):
        self.observed = []

    def process(self, *items):
        self.observed = items

        if len(items) == 2 and isinstance(items[0], Learner): #eval task
            items[0].learn(None, None, None, None, None)
            return []
        else: #learner or environment task
            return {}

class ExceptionTask:
    def __init__(self):
        self.observed = []

    def process(self, *items):
        raise Exception()

class CountReadSimulation:
    def __init__(self) -> None:
        self.n_reads = 0

    def read(self) -> Iterable[SimulatedInteraction]:
        yield SimulatedInteraction(self.n_reads, [0,1], [0,1])
        self.n_reads += 1

    def __str__(self) -> str:
        return "CountRead"

class ExceptionSimulation:
    def read(self) -> Iterable[SimulatedInteraction]:
        raise Exception()

class CountFilter:
    def __init__(self) -> None:
        self.n_filter = 0

    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[SimulatedInteraction]:
        for interaction in interactions:
            yield SimulatedInteraction((interaction['context'], self.n_filter), interaction['actions'], interaction['rewards'], **interaction.extra)
        self.n_filter += 1
#for testing purposes

class CreateWorkItems_Tests(unittest.TestCase):

    def test_two_sim_two_learns(self):
        sim1 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        sim2 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")

        tasks = list(CreateWorkItems([sim1,sim2], [lrn1,lrn2], None, None, None).read())

        self.assertEqual(8, len(tasks))

        self.assertEqual(2, len([t for t in tasks if not t.env and t.lrn]) )
        self.assertEqual(2, len([t for t in tasks if t.env and not t.lrn]) )
        self.assertEqual(4, len([t for t in tasks if t.env and t.lrn]) )

class RemoveFinished_Tests(unittest.TestCase):

    def test_three_finished(self):

        restored = Result(lrn_rows={1:{}}, env_rows= {0:{}}, int_rows= {(0,1):{}})

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

    def test_restored_none(self):

        restored = None

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

        self.assertEqual(8, len(unfinished_tasks))

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
        sim1 = Environments(LinearSyntheticSimulation(500)).shuffle([1,2])._environments
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
            WorkItem(0, None, sim1, None, None),
            WorkItem(1, None, sim2, None, None),
            WorkItem(2, None, sim1, None, None),
            WorkItem(0, 0, sim1, None, None),
            WorkItem(0, 1, sim1, None, None),
            WorkItem(1, 0, sim2, None, None),
            WorkItem(1, 1, sim2, None, None),
            WorkItem(2, 0, sim1, None, None),
        ]

        groups = list(ChunkByTask().filter(tasks))

        self.assertEqual(len(groups), 10)
        self.assertEqual(groups[0], [tasks[0]])
        self.assertEqual(groups[1], [tasks[1]])
        self.assertEqual(groups[2], [tasks[2]])
        self.assertEqual(groups[3], [tasks[3]])
        self.assertEqual(groups[4], [tasks[4]])
        self.assertEqual(groups[5], [tasks[5]])
        self.assertEqual(groups[6], [tasks[6]])
        self.assertEqual(groups[7], [tasks[7]])
        self.assertEqual(groups[8], [tasks[8]])
        self.assertEqual(groups[9], [tasks[9]])

class MaxChunkSize_Tests(unittest.TestCase):
    
    def test_max_size_0(self):
        actual   = list(MaxChunkSize(0).filter([[1,2,3],[4,5],[6]]))
        expected = [[1,2,3],[4,5],[6]]
        self.assertEqual(expected, actual)

    def test_max_size_1(self):
        actual   = list(MaxChunkSize(1).filter([[1,2,3],[4,5],[6]]))
        expected = [[1],[2],[3],[4],[5],[6]]
        self.assertEqual(expected, actual)
    
    def test_max_size_2(self):
        actual   = list(MaxChunkSize(2).filter([[1,2,3],[4,5],[6]]))
        expected = [[1,2],[3],[4,5],[6]] 
        self.assertEqual(expected, actual)

class ProcessTasks_Tests(unittest.TestCase):

    def setUp(self) -> None:
        CobaContext.logger = BasicLogger(ListSink())

    def test_simple(self):

        sim1 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        lrn1 = ModuloLearner("1")
        task = ObserveTask()

        item = WorkItem(1, 1, sim1, lrn1, task)

        transactions = list(ProcessWorkItems().filter([item]))

        self.assertEqual(len(task.observed[1]), 5)
        self.assertEqual(['T3', (1,1), []], transactions[0])

    def test_one_source_environment_reused(self):

        sim1 = CountReadSimulation()

        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")

        task1 = ObserveTask()
        task2 = ObserveTask()
        task3 = ObserveTask()

        items = [ 
            WorkItem(0, None, sim1, None, task1), 
            WorkItem(0, 0   , sim1, lrn1, task2), 
            WorkItem(0, 1   , sim1, lrn2, task3) 
        ]

        transactions = list(ProcessWorkItems().filter(items))

        self.assertEqual(len(task1.observed[1]), 1)
        self.assertEqual(len(task2.observed[1]), 1)
        self.assertEqual(len(task3.observed[1]), 1)

        self.assertEqual(task1.observed[1][0]['context'], 0)
        self.assertEqual(task2.observed[1][0]['context'], 0)

        self.assertEqual(['T2',    0 , {}], transactions[0])
        self.assertEqual(['T3', (0,0), []], transactions[1])
        self.assertEqual(['T3', (0,1), []], transactions[2])

        self.assertEqual(sim1.n_reads, 1)

        self.assertGreater(lrn1.n_learns, 0)
        self.assertGreater(lrn2.n_learns, 0)

    def test_one_pipe_environment_reused(self):

        sim1 = CountReadSimulation()
        flt1 = CountFilter()
        env1 = Pipes.join(sim1, flt1)

        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")

        task1 = ObserveTask()
        task2 = ObserveTask()
        task3 = ObserveTask()

        items = [ 
            WorkItem(0, None, env1, None, task1), 
            WorkItem(0, 0   , env1, lrn1, task2), 
            WorkItem(0, 1   , env1, lrn2, task3) 
        ]

        transactions = list(ProcessWorkItems().filter(items))

        self.assertEqual(len(task1.observed[1]), 1)
        self.assertEqual(len(task2.observed[1]), 1)
        self.assertEqual(len(task3.observed[1]), 1)

        self.assertEqual(task1.observed[1][0]['context'], (0,0) )
        self.assertEqual(task2.observed[1][0]['context'], (0,0) )

        self.assertEqual(['T2',    0 , {}], transactions[0])
        self.assertEqual(['T3', (0,0), []], transactions[1])
        self.assertEqual(['T3', (0,1), []], transactions[2])

        self.assertEqual(sim1.n_reads , 1)
        self.assertEqual(flt1.n_filter, 1)

        self.assertEqual(lrn1.n_learns, 1)
        self.assertEqual(lrn2.n_learns, 1)

    def test_one_learner_evaluated_twice_is_deep_copied(self):

        #make sure both environments are read and learner is deepcopied

        sim1 = CountReadSimulation()
        sim2 = CountReadSimulation()

        lrn1 = ModuloLearner("1")

        task1 = ObserveTask()
        task2 = ObserveTask()

        items = [ WorkItem(0, 1, sim1, lrn1, task1), WorkItem(1, 1, sim2, lrn1, task2) ]

        transactions = list(ProcessWorkItems().filter(items))

        self.assertEqual(len(task1.observed[1]), 1)
        self.assertEqual(len(task2.observed[1]), 1)

        self.assertEqual(task1.observed[1][0]['context'], 0)
        self.assertEqual(task2.observed[1][0]['context'], 0)

        self.assertIn(['T3', (0,1), []], transactions)
        self.assertIn(['T3', (1,1), []], transactions)

        self.assertEqual(sim1.n_reads, 1)
        self.assertEqual(sim2.n_reads, 1)

        self.assertEqual(lrn1.n_learns, 0)

    def test_two_learners_evaluated_once_are_not_deep_copied(self):

        sim1 = CountReadSimulation()

        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")

        task1 = ObserveTask()
        task2 = ObserveTask()

        items = [ WorkItem(0, 0, sim1, lrn1, task1), WorkItem(0, 1, sim1, lrn2, task2) ]

        transactions = list(ProcessWorkItems().filter(items))

        self.assertEqual(len(task1.observed[1]), 1)
        self.assertEqual(len(task2.observed[1]), 1)

        self.assertEqual(task1.observed[1][0]['context'], 0)
        self.assertEqual(task2.observed[1][0]['context'], 0)

        self.assertEqual(['T3', (0,0), []], transactions[0])
        self.assertEqual(['T3', (0,1), []], transactions[1])

        self.assertEqual(sim1.n_reads, 1)

        self.assertEqual(lrn1.n_learns, 1)
        self.assertEqual(lrn2.n_learns, 1)

    def test_two_eval_tasks_two_source_two_env(self):

        #make both environments are read and learners aren't deepcopied

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

        self.assertEqual(task1.observed[1][0]['context'], 0)
        self.assertEqual(task2.observed[1][0]['context'], 0)

        self.assertIn(['T3', (0,0), []], transactions)
        self.assertIn(['T3', (1,1), []], transactions)

        self.assertGreater(lrn1.n_learns, 0)
        self.assertGreater(lrn2.n_learns, 0)

        self.assertEqual(sim1.n_reads, 1)
        self.assertEqual(sim2.n_reads, 1)

    def test_one_source_in_two_env_is_read_once(self):

        filt1 = CountFilter()
        filt2 = CountFilter()

        src1   = CountReadSimulation()
        sim1   = Pipes.join(src1, filt1)
        sim2   = Pipes.join(src1, filt2)

        lrn1   = ModuloLearner("1")
        lrn2   = ModuloLearner("2")

        task1 = ObserveTask()
        task2 = ObserveTask()

        items = [ WorkItem(0, 0, sim1, lrn1, task1), WorkItem(1, 1, sim2, lrn2, task2) ]

        transactions = list(ProcessWorkItems().filter(items))

        self.assertEqual(src1.n_reads  , 1)
        self.assertEqual(filt1.n_filter, 1)
        self.assertEqual(filt2.n_filter, 1)

        self.assertEqual(len(task1.observed[1]), 1)
        self.assertEqual(len(task2.observed[1]), 1)

        self.assertEqual(task1.observed[1][0]['context'], (0,0))
        self.assertEqual(task2.observed[1][0]['context'], (0,0))

        self.assertEqual(['T3', (0,0), []], transactions[0])
        self.assertEqual(['T3', (1,1), []], transactions[1])

    def test_one_source_in_two_env_and_materialized_is_not_read(self):

        filt1 = CountFilter()
        filt2 = CountFilter()

        src1   = CountReadSimulation()
        envs   = Environments([Pipes.join(src1, filt1), Pipes.join(src1, filt2)]).materialize()
        env1   = envs[0]
        env2   = envs[1]

        lrn1   = ModuloLearner("1")
        lrn2   = ModuloLearner("2")

        task1 = ObserveTask()
        task2 = ObserveTask()

        items = [ WorkItem(0, 0, env1, lrn1, task1), WorkItem(1, 1, env2, lrn2, task2) ]

        transactions = list(ProcessWorkItems().filter(items))

        self.assertEqual(CobaContext.logger.sink.items[1], 'Loading CountRead...')

        self.assertEqual(src1.n_reads  , 2)
        self.assertEqual(filt1.n_filter, 1)
        self.assertEqual(filt2.n_filter, 1)

        self.assertEqual(len(task1.observed[1]), 1)
        self.assertEqual(len(task2.observed[1]), 1)

        self.assertEqual(task1.observed[1][0]['context'], (0,0))
        self.assertEqual(task2.observed[1][0]['context'], (1,0)) #note that we load the source twice

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

        self.assertEqual(task1.observed[0].params['p'], "1")
        self.assertEqual(task2.observed[0].params['p'], "2")

        self.assertEqual(['T1', 0, {}], transactions[0])
        self.assertEqual(['T1', 1, {}], transactions[1])

        self.assertEqual(lrn1.n_learns, 0)
        self.assertEqual(lrn2.n_learns, 0)

    def test_two_environment_tasks(self):

        filter = CountFilter()
        src1   = CountReadSimulation()
        sim1   = Pipes.join(src1, filter)
        sim2   = Pipes.join(src1, filter)

        task1 = ObserveTask()
        task2 = ObserveTask()

        items = [ WorkItem(0, None, sim1, None, task1), WorkItem(1, None, sim2, None, task2) ]

        transactions = list(ProcessWorkItems().filter(items))

        self.assertEqual(len(task1.observed[1]), 1)
        self.assertEqual(len(task2.observed[1]), 1)

        self.assertEqual(task1.observed[1][0]['context'], (0,0))
        self.assertEqual(task2.observed[1][0]['context'], (0,1))

        self.assertEqual(['T2', 0, {}], transactions[0])
        self.assertEqual(['T2', 1, {}], transactions[1])

    def test_two_environment_tasks_first_environment_is_empty(self):

        src1  = LinearSyntheticSimulation(n_interactions=0)
        src2  = LinearSyntheticSimulation(n_interactions=5)

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

        self.assertEqual(task2.observed[1][0]['context'], 0)

        self.assertEqual(['T2', 1, {}], transactions[0])

if __name__ == '__main__':
    unittest.main()