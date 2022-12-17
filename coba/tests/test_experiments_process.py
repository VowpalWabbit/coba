import unittest

from itertools import product
from typing import cast, Iterable

from coba.contexts     import CobaContext, BasicLogger
from coba.environments import LambdaSimulation, SimulatedInteraction, Environments, LinearSyntheticSimulation
from coba.pipes        import Pipes, ListSink, Cache
from coba.learners     import Learner

from coba.experiments.results import Result
from coba.experiments.process import (
    WorkItem, CreateWorkItems, RemoveFinished,
    ChunkByChunk, ProcessWorkItems,
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
        if len(items)==2: 
            items = (items[0], list(items[1]))
        self.observed = list(items)

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

    def test_two_env_two_lrn(self):
        env1 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        env2 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")

        pairs = list(product([lrn1,lrn2],[env1,env2]))

        works = list(CreateWorkItems(pairs, None, None, None).read())

        self.assertEqual(8, len(works))

        self.assertTrue(all([w.lrn_id == 0 for w in works if w.lrn is lrn1]))
        self.assertTrue(all([w.lrn_id == 1 for w in works if w.lrn is lrn2]))
        self.assertTrue(all([w.env_id == 0 for w in works if w.env is env1]))
        self.assertTrue(all([w.env_id == 1 for w in works if w.env is env2]))

        self.assertEqual(1, len([t for t in works if not t.env and t.lrn is lrn1]))
        self.assertEqual(1, len([t for t in works if not t.env and t.lrn is lrn2]))

        self.assertEqual(1, len([t for t in works if t.env is env1 and not t.lrn]))
        self.assertEqual(1, len([t for t in works if t.env is env2 and not t.lrn]))

        for l,e in pairs:
            self.assertEqual(1, len([t for t in works if t.env is e and t.lrn is l]))

    def test_uneven_pairs(self):
        env1 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        env2 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")

        pairs = list(product([lrn1,lrn2],[env1,env2]))[0:3]

        works = list(CreateWorkItems(pairs, None, None, None).read())

        self.assertEqual(7, len(works))

        self.assertTrue(all([w.lrn_id == 0 for w in works if w.lrn is lrn1]))
        self.assertTrue(all([w.lrn_id == 1 for w in works if w.lrn is lrn2]))
        self.assertTrue(all([w.env_id == 0 for w in works if w.env is env1]))
        self.assertTrue(all([w.env_id == 1 for w in works if w.env is env2]))

        self.assertEqual(1, len([t for t in works if not t.env and t.lrn is lrn1]))
        self.assertEqual(1, len([t for t in works if not t.env and t.lrn is lrn2]))

        self.assertEqual(1, len([t for t in works if t.env is env1 and not t.lrn]))
        self.assertEqual(1, len([t for t in works if t.env is env2 and not t.lrn]))

        for l,e in pairs:
            self.assertEqual(1, len([t for t in works if t.env is e and t.lrn is l]))

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

class ChunkByChunk_Tests(unittest.TestCase):

    def test_no_chunks(self):
        src1 = Environments.from_linear_synthetic(10)
        src2 = Environments.from_linear_synthetic(10)

        envs = (src1+src2)

        tasks = [
            WorkItem(None, 0, None, None, None),
            WorkItem(None, 1, None, None, None),
            WorkItem(1, None, envs[0], None, None),
            WorkItem(0, None, envs[1], None, None),
            WorkItem(1, 1, envs[0], None, None),
            WorkItem(0, 0, envs[1], None, None),
            WorkItem(2, 0, envs[1], None, None),
            WorkItem(0, 1, envs[1], None, None),
        ]

        groups = list(ChunkByChunk().filter(tasks))

        self.assertEqual(len(groups), 8)
        self.assertEqual(groups[0], tasks[0:1])
        self.assertEqual(groups[1], tasks[1:2])
        self.assertEqual(groups[2], tasks[2:3])
        self.assertEqual(groups[3], tasks[3:4])
        self.assertEqual(groups[4], tasks[4:5])
        self.assertEqual(groups[5], tasks[5:6])
        self.assertEqual(groups[6], tasks[6:7])
        self.assertEqual(groups[7], tasks[7:8])

    def test_two_chunks(self):
        src1 = Environments.from_linear_synthetic(10)
        src2 = Environments.from_linear_synthetic(10)

        envs = (src1+src2).chunk()

        tasks = [
            WorkItem(None, 0, None, None, None),
            WorkItem(None, 1, None, None, None),
            WorkItem(1, None, envs[0], None, None),
            WorkItem(0, None, envs[1], None, None),
            WorkItem(1, 1, envs[0], None, None),
            WorkItem(0, 0, envs[1], None, None),
            WorkItem(2, 0, envs[1], None, None),
            WorkItem(0, 1, envs[1], None, None),
        ]

        groups = list(ChunkByChunk().filter(tasks))

        self.assertEqual(len(groups), 4)
        self.assertEqual(groups[0], tasks[0:1])
        self.assertEqual(groups[1], tasks[1:2])
        self.assertEqual(groups[2], [tasks[3],tasks[5],tasks[7],tasks[6]])
        self.assertCountEqual(groups[3], [tasks[2],tasks[4]])

    def test_two_chunks_two_shuffles(self):
        src1 = Environments.from_linear_synthetic(10)
        src2 = Environments.from_linear_synthetic(10)
        envs = (src1+src2).chunk().shuffle(n=2)

        tasks = [
            WorkItem(None, 0, None, None, None),
            WorkItem(None, 1, None, None, None),
            WorkItem(1, None, envs[0], None, None),
            WorkItem(0, None, envs[1], None, None),
            WorkItem(1, 1, envs[2], None, None),
            WorkItem(0, 0, envs[1], None, None),
            WorkItem(2, 0, envs[3], None, None),
            WorkItem(0, 1, envs[3], None, None),
        ]

        groups = list(ChunkByChunk().filter(tasks))

        self.assertEqual(len(groups), 4)
        self.assertEqual(groups[0], tasks[0:1])
        self.assertEqual(groups[1], tasks[1:2])
        self.assertCountEqual(groups[2], [tasks[3],tasks[5],tasks[7],tasks[6]])
        self.assertCountEqual(groups[3], [tasks[2],tasks[4]])

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

        sim1 = Pipes.join(CountReadSimulation(), Cache())

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

        self.assertEqual(sim1[0].n_reads, 1)

        self.assertGreater(lrn1.n_learns, 0)
        self.assertGreater(lrn2.n_learns, 0)

    def test_one_pipe_environment_reused(self):

        src1 = CountReadSimulation()
        flt1 = CountFilter()
        env1 = Pipes.join(src1, flt1, Cache())

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

        self.assertEqual(src1.n_reads , 1)
        self.assertEqual(flt1.n_filter, 1)

        self.assertEqual(lrn1.n_learns, 1)
        self.assertEqual(lrn2.n_learns, 1)

    def test_one_learner_evaluated_twice_is_deep_copied(self):

        #make sure the learner is deepcopied

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

        #make sure the environment is read twice and learners aren't deepcopied

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
        self.assertEqual(task2.observed[1][0]['context'], 1)

        self.assertEqual(['T3', (0,0), []], transactions[0])
        self.assertEqual(['T3', (0,1), []], transactions[1])

        self.assertEqual(sim1.n_reads, 2)

        self.assertEqual(lrn1.n_learns, 1)
        self.assertEqual(lrn2.n_learns, 1)

    def test_two_eval_tasks_two_source_two_env(self):

        #make sure both environments are read and learners aren't deepcopied

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
        cache  = Cache()
        sim1   = Pipes.join(src1, cache, filt1)
        sim2   = Pipes.join(src1, cache, filt2)

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
        cache  = Cache()
        sim1   = Pipes.join(src1, cache, filter)
        sim2   = Pipes.join(src1, cache, filter)

        task1 = ObserveTask()
        task2 = ObserveTask()

        items = [ WorkItem(0, None, sim1, None, task1), WorkItem(1, None, sim2, None, task2) ]

        transactions = list(ProcessWorkItems().filter(items))

        self.assertEqual(src1.n_reads  , 1)
        self.assertEqual(filter.n_filter,2)

        self.assertEqual(len(task1.observed[1]), 1)
        self.assertEqual(len(task2.observed[1]), 1)

        self.assertEqual(task1.observed[1][0]['context'], (0,0))
        self.assertEqual(task2.observed[1][0]['context'], (0,1))

        self.assertEqual(['T2', 0, {}], transactions[0])
        self.assertEqual(['T2', 1, {}], transactions[1])

    def test_two_environment_tasks_first_environment_is_empty(self):

        lrn1 = ModuloLearner("1")
        src1  = LinearSyntheticSimulation(n_interactions=0)
        src2  = LinearSyntheticSimulation(n_interactions=5)

        task1 = ObserveTask()
        task2 = ObserveTask()
        task3 = ObserveTask()


        items = [ WorkItem(0, None, src1, None, task1), WorkItem(0, 0, src1, lrn1, task2), WorkItem(1, None, src2, None, task3) ]

        transactions = list(ProcessWorkItems().filter(items))

        self.assertEqual(len(task1.observed), 2)
        self.assertEqual(len(task2.observed), 0)
        self.assertEqual(len(task3.observed[1]), 5)

    def test_empty_environment_eval_task(self):

        lrn1 = ModuloLearner("1")
        src1  = LinearSyntheticSimulation(0)
        task1 = ObserveTask()

        items = [ WorkItem(0, 0, src1, lrn1, task1) ]

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