import unittest

from typing import cast

from coba.simulations import LambdaSimulation
from coba.pipes import Source
from coba.learners import Learner

from coba.benchmarks.results import Result
from coba.benchmarks.tasks import BenchmarkTask, Tasks, Unfinished, GroupBySource

#for testing purposes
class ModuloLearner(Learner):
    def __init__(self, param:str="0"):
        self._param = param

    @property
    def family(self):
        return "Modulo"

    @property
    def params(self):
        return {"p":self._param}

    def predict(self, key, context, actions):
        return [ int(i == actions.index(actions[context%len(actions)])) for i in range(len(actions)) ]

    def learn(self, key, context, action, reward, probability):
        pass

class OneTimeSource(Source):

    def __init__(self, source: Source) -> None:
        self._source = source
        self._read_count = 0

    def read(self):
        if self._read_count > 0: raise Exception("Read more than once")

        self._read_count += 1

        return self._source.read()

class Tasks_Tests(unittest.TestCase):

    def test_one_sim_two_learns(self):
        sim1 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")

        tasks = list(Tasks([sim1,sim1], [lrn1,lrn2], seed=10).read())

        self.assertEqual(4, len(tasks))

        self.assertEqual(0, tasks[0].sim_id)
        self.assertEqual(0, tasks[1].sim_id)
        self.assertEqual(1, tasks[2].sim_id)
        self.assertEqual(1, tasks[3].sim_id)

        self.assertEqual(0, tasks[0].lrn_id)
        self.assertEqual(1, tasks[1].lrn_id)
        self.assertEqual(0, tasks[2].lrn_id)
        self.assertEqual(1, tasks[3].lrn_id)

        self.assertEqual(4, len(set([id(t.learner) for t in tasks ])))
        self.assertEqual(1, len(set([id(t.simulation.source) for t in tasks ])))

    def test_two_sims_two_learns(self):
        sim1 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        sim2 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")

        tasks = list(Tasks([sim1,sim2], [lrn1,lrn2], seed=10).read())

        self.assertEqual(4, len(tasks))

        self.assertEqual(0, tasks[0].sim_id)
        self.assertEqual(0, tasks[1].sim_id)
        self.assertEqual(1, tasks[2].sim_id)
        self.assertEqual(1, tasks[3].sim_id)

        self.assertEqual(0, tasks[0].lrn_id)
        self.assertEqual(1, tasks[1].lrn_id)
        self.assertEqual(0, tasks[2].lrn_id)
        self.assertEqual(1, tasks[3].lrn_id)

        self.assertEqual(4, len(set([id(t.learner) for t in tasks ])))
        self.assertEqual(2, len(set([id(t.simulation.source) for t in tasks ])))

class Unifinshed_Tests(unittest.TestCase):

    def test_one_finished(self):

        restored = Result()

        restored.simulations.add_row(simulation_id=0,batch_count=1)
        restored.batches.add_row(simulation_id=0,learner_id=1)

        sim1 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        lrn1 = ModuloLearner("1")

        tasks = [
            BenchmarkTask(0,0,0,sim1,lrn1,10),
            BenchmarkTask(0,0,1,sim1,lrn1,10),
            BenchmarkTask(0,1,0,sim1,lrn1,10),
            BenchmarkTask(0,1,1,sim1,lrn1,10),
        ]

        unfinished_tasks = list(Unfinished(restored).filter(tasks))

        self.assertEqual(3, len(unfinished_tasks))

        self.assertEqual(0, unfinished_tasks[0].sim_id)
        self.assertEqual(1, unfinished_tasks[1].sim_id)
        self.assertEqual(1, unfinished_tasks[2].sim_id)

        self.assertEqual(0, unfinished_tasks[0].lrn_id)
        self.assertEqual(0, unfinished_tasks[1].lrn_id)
        self.assertEqual(1, unfinished_tasks[2].lrn_id)

    def test_one_simulation_empty(self):

        restored = Result()
        restored.simulations.add_row(simulation_id=0, batch_count=0)

        sim1 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        lrn1 = ModuloLearner("1")

        tasks = [
            BenchmarkTask(0,0,0,sim1,lrn1,10),
            BenchmarkTask(0,0,1,sim1,lrn1,10),
            BenchmarkTask(0,1,0,sim1,lrn1,10),
            BenchmarkTask(0,1,1,sim1,lrn1,10),
        ]

        unfinished_tasks = list(Unfinished(restored).filter(tasks))

        self.assertEqual(2, len(unfinished_tasks))

        self.assertEqual(1, unfinished_tasks[0].sim_id)
        self.assertEqual(1, unfinished_tasks[1].sim_id)

        self.assertEqual(0, unfinished_tasks[0].lrn_id)
        self.assertEqual(1, unfinished_tasks[1].lrn_id)

class GroupBySource_Tests(unittest.TestCase):

    def test_one_group(self):
        sim1 = OneTimeSource(LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a)))
        lrn1 = ModuloLearner("1")

        tasks = [
            BenchmarkTask(0,0,0,sim1,lrn1,10),
            BenchmarkTask(0,0,1,sim1,lrn1,10),
            BenchmarkTask(0,1,0,sim1,lrn1,10),
            BenchmarkTask(0,1,1,sim1,lrn1,10),
        ]

        groups = list(GroupBySource().filter(tasks))
        tasks  = list(groups[0])
        
        self.assertEqual(1, len(groups))
        self.assertEqual(4, len(tasks))

    def test_two_groups(self):
        sim1 = OneTimeSource(LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a)))
        sim2 = OneTimeSource(LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a)))
        lrn1 = ModuloLearner("1")

        tasks = [
            BenchmarkTask(0,0,0,sim1,lrn1,10),
            BenchmarkTask(0,0,1,sim1,lrn1,10),
            BenchmarkTask(1,1,0,sim2,lrn1,10),
            BenchmarkTask(1,1,1,sim2,lrn1,10),
        ]

        groups = list(GroupBySource().filter(tasks))
        
        group_1_tasks = list(groups[0])
        group_2_tasks = list(groups[1])

        self.assertEqual(2, len(groups))
        self.assertEqual(2, len(group_1_tasks))
        self.assertEqual(2, len(group_2_tasks))

        self.assertEqual(0, group_1_tasks[0].sim_id)
        self.assertEqual(0, group_1_tasks[1].sim_id)
        self.assertEqual(1, group_2_tasks[0].sim_id)
        self.assertEqual(1, group_2_tasks[1].sim_id)

        self.assertEqual(0, group_1_tasks[0].lrn_id)
        self.assertEqual(1, group_1_tasks[1].lrn_id)
        self.assertEqual(0, group_2_tasks[0].lrn_id)
        self.assertEqual(1, group_2_tasks[1].lrn_id)

if __name__ == '__main__':
    unittest.main()