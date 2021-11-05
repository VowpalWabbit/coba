import unittest

import json
from typing import cast, Iterable, Any

from coba.environments import LambdaSimulation, SimulatedInteraction, ClassificationSimulation, SimSourceFilters, ValidationSimulation
from coba.pipes import Source, Pipe, IdentityFilter
from coba.learners import Learner, RandomLearner

from coba.experiments.results import Result
from coba.experiments.tasks import (
    Task, SimulationTask, SimulationEvaluationTask, CreateTasks, FilterFinished, ChunkBySource, ProcessTasks, Identifier
)

#for testing purposes
class ModuloLearner(Learner):
    def __init__(self, param:str="0"):
        self._param = param

    @property
    def params(self):
        return {"family": "Modulo", "p":self._param}

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

class ObserveTask(Task):
    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[Any]:
        self.observed = list(interactions)

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
            yield SimulatedInteraction((interaction.context, self._count), interaction.actions, reveals=interaction.reveals, **interaction.results)

        self._count += 1
#for testing purposes

class CreateTasks_Tests(unittest.TestCase):

    def test_one_sim_two_learns(self):
        sim1 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        sim2 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")

        tasks = list(CreateTasks([sim1,sim2], [lrn1,lrn2], seed=10).read())

        self.assertEqual(6, len(tasks))

        self.assertEqual(0, tasks[0].sim_id)
        self.assertEqual(1, tasks[1].sim_id)

        self.assertEqual(0, tasks[2].sim_id)
        self.assertEqual(0, tasks[3].sim_id)
        self.assertEqual(1, tasks[4].sim_id)
        self.assertEqual(1, tasks[5].sim_id)

        self.assertEqual(None, tasks[0].lrn_id)
        self.assertEqual(None, tasks[1].lrn_id)

        self.assertEqual(0, tasks[2].lrn_id)
        self.assertEqual(1, tasks[3].lrn_id)
        self.assertEqual(0, tasks[4].lrn_id)
        self.assertEqual(1, tasks[5].lrn_id)

        self.assertEqual(5, len(set([id(t.learner) for t in tasks ])))
        self.assertEqual(2, len(set([id(t.sim_source) for t in tasks ])))

    def test_one_src_two_sims_two_learns(self):
        src1 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        
        sim1 = Pipe.join(src1, [IdentityFilter()])
        sim2 = Pipe.join(src1, [IdentityFilter()])
        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")

        tasks = list(CreateTasks([sim1,sim2], [lrn1,lrn2], seed=10).read())

        self.assertEqual(6, len(tasks))

        self.assertEqual(0, tasks[0].sim_id)
        self.assertEqual(1, tasks[1].sim_id)
        
        self.assertEqual(0, tasks[2].sim_id)
        self.assertEqual(0, tasks[3].sim_id)
        self.assertEqual(1, tasks[4].sim_id)
        self.assertEqual(1, tasks[5].sim_id)

        self.assertEqual(None, tasks[0].lrn_id)
        self.assertEqual(None, tasks[1].lrn_id)

        self.assertEqual(0, tasks[2].lrn_id)
        self.assertEqual(1, tasks[3].lrn_id)
        self.assertEqual(0, tasks[4].lrn_id)
        self.assertEqual(1, tasks[5].lrn_id)

        self.assertEqual(5, len(set([id(t.learner) for t in tasks ])))
        self.assertEqual(1, len(set([id(t.sim_source) for t in tasks ])))

class Unifinshed_Tests(unittest.TestCase):

    def test_one_finished(self):

        restored = Result(sim_rows=[dict(simulation_id=0)],int_rows=[dict(simulation_id=0,learner_id=1)])

        sim1 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        lrn1 = ModuloLearner("1")

        tasks = [
            SimulationEvaluationTask(0,0,0,sim1,lrn1,10),
            SimulationEvaluationTask(0,0,1,sim1,lrn1,10),
            SimulationEvaluationTask(0,1,0,sim1,lrn1,10),
            SimulationEvaluationTask(0,1,1,sim1,lrn1,10),
        ]

        unfinished_tasks = list(FilterFinished(restored).filter(tasks))

        self.assertEqual(3, len(unfinished_tasks))

        self.assertEqual(0, unfinished_tasks[0].sim_id)
        self.assertEqual(1, unfinished_tasks[1].sim_id)
        self.assertEqual(1, unfinished_tasks[2].sim_id)

        self.assertEqual(0, unfinished_tasks[0].lrn_id)
        self.assertEqual(0, unfinished_tasks[1].lrn_id)
        self.assertEqual(1, unfinished_tasks[2].lrn_id)

class GroupBySource_Tests(unittest.TestCase):

    def test_one_group(self):
        sim1 = OneTimeSource(LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a)))
        lrn1 = ModuloLearner("1")

        tasks = [
            SimulationEvaluationTask(0,0,0,sim1,lrn1,10),
            SimulationEvaluationTask(0,0,1,sim1,lrn1,10),
            SimulationEvaluationTask(0,1,0,sim1,lrn1,10),
            SimulationEvaluationTask(0,1,1,sim1,lrn1,10),
        ]

        groups = list(ChunkBySource().filter(tasks))
        tasks  = list(groups[0])
        
        self.assertEqual(1, len(groups))
        self.assertEqual(4, len(tasks))

    def test_two_groups(self):
        sim1 = OneTimeSource(LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a)))
        sim2 = OneTimeSource(LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a)))
        lrn1 = ModuloLearner("1")

        tasks = [
            SimulationEvaluationTask(0,0,0,sim1,lrn1,10),
            SimulationEvaluationTask(0,0,1,sim1,lrn1,10),
            SimulationEvaluationTask(1,1,0,sim2,lrn1,10),
            SimulationEvaluationTask(1,1,1,sim2,lrn1,10),
        ]

        groups = list(ChunkBySource().filter(tasks))
        
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

class SimulationTask_Tests(unittest.TestCase):

    def test_classification_statistics_dense(self):
        
        try:
            import sklearn
        except:
            sklearn_installed = False
        else:
            sklearn_installed = True

        simulation   = ClassificationSimulation([(1,2),(3,4)]*10, ["A","B"]*10)
        task         = SimulationTask(0, 1, None, simulation, None)
        transactions = list(task.filter(simulation.read()))

        self.assertEqual(1  , len(transactions))
        self.assertEqual('S', transactions[0][0])
        self.assertEqual(1  , transactions[0][1])
        self.assertEqual(2  , transactions[0][2]["action_cardinality"])
        self.assertEqual(2  , transactions[0][2]["context_dimensions"])
        self.assertEqual(1  , transactions[0][2]["imbalance_ratio"])

        if sklearn_installed:
            self.assertEqual(1  , transactions[0][2]["bayes_rate_avg"])
            self.assertEqual(0  , transactions[0][2]["bayes_rate_iqr"])
            self.assertEqual(1  , transactions[0][2]["centroid_purity"])
            self.assertEqual(0  , transactions[0][2]["centroid_distance"])

    def test_classification_statistics_sparse1(self):
        
        try:
            import sklearn
        except:
            sklearn_installed = False
        else:
            sklearn_installed = True

        c1 = {"1":1, "2":2}
        c2 = {"1":3, "2":4}

        simulation   = ClassificationSimulation([c1,c2]*10, ["A","B"]*10)
        task         = SimulationTask(0, 1, None, simulation, None)
        transactions = list(task.filter(simulation.read()))

        self.assertEqual(1  , len(transactions))
        self.assertEqual('S', transactions[0][0])
        self.assertEqual(1  , transactions[0][1])
        self.assertEqual(2  , transactions[0][2]["action_cardinality"])
        self.assertEqual(2  , transactions[0][2]["context_dimensions"])
        self.assertEqual(1  , transactions[0][2]["imbalance_ratio"])

        if sklearn_installed:
            self.assertEqual(1  , transactions[0][2]["bayes_rate_avg"])
            self.assertEqual(0  , transactions[0][2]["bayes_rate_iqr"])
            self.assertEqual(1  , transactions[0][2]["centroid_purity"])
            self.assertEqual(0  , transactions[0][2]["centroid_distance"])

    def test_classification_statistics_sparse2(self):

        try:
            import sklearn
        except:
            sklearn_installed = False
        else:
            sklearn_installed = True

        simulation   = ClassificationSimulation([(1,'A')]*20, ["A","B"]*10)
        task         = SimulationTask(0, 1, None, simulation, None)
        transactions = list(task.filter(simulation.read()))

        self.assertEqual(1  , len(transactions))
        self.assertEqual('S', transactions[0][0])
        self.assertEqual(1  , transactions[0][1])
        self.assertEqual(2  , transactions[0][2]["action_cardinality"])
        self.assertEqual(2  , transactions[0][2]["context_dimensions"])
        self.assertEqual(1  , transactions[0][2]["imbalance_ratio"])

        if sklearn_installed:
            self.assertEqual(.5 , transactions[0][2]["bayes_rate_avg"])
            self.assertEqual(0  , transactions[0][2]["bayes_rate_iqr"])

    def test_classification_statistics_encodable(self):

        try:
            import sklearn
        except:
            sklearn_installed = False
        else:
            sklearn_installed = True

        simulation   = ClassificationSimulation([(1,'A')]*20, ["A","B"]*10)
        task         = SimulationTask(0, 1, None, simulation, None)
        transactions = list(task.filter(simulation.read()))

        self.assertEqual(1  , len(transactions))
        self.assertEqual('S', transactions[0][0])
        self.assertEqual(1  , transactions[0][1])
        self.assertEqual(2  , transactions[0][2]["action_cardinality"])
        self.assertEqual(2  , transactions[0][2]["context_dimensions"])
        self.assertEqual(1  , transactions[0][2]["imbalance_ratio"])

        if sklearn_installed:
            self.assertEqual(.5 , transactions[0][2]["bayes_rate_avg"])
            self.assertEqual(0  , transactions[0][2]["bayes_rate_iqr"])

        json.dumps(transactions)

class EvaluationTask_Tests(unittest.TestCase):

    def test_simple(self):

        task = SimulationEvaluationTask(0,0,0,None,RandomLearner(),0)

        transactions = list(task.filter([
            SimulatedInteraction(1,[1,2,3],rewards=[4,5,6]),
            SimulatedInteraction(1,[1,2,3],rewards=[4,5,6]),
            SimulatedInteraction(1,[1,2,3],rewards=[4,5,6]),
            SimulatedInteraction(1,[1,2,3],rewards=[4,5,6]),
        ]))

        self.assertEqual("I", transactions[0][0])
        self.assertEqual((0,0), transactions[0][1])
        self.assertEqual({"_packed": { 'reward':[4,6,5,4]}}, transactions[0][2])

    def test_reveals_results(self):

        task = SimulationEvaluationTask(0,0,0,None,RandomLearner(),0)

        transactions = list(task.filter([
            SimulatedInteraction(1,[1,2,3],reveals=[4,5,6],rewards=[1,2,3]),
            SimulatedInteraction(1,[1,2,3],reveals=[4,5,6],rewards=[4,5,6]),
            SimulatedInteraction(1,[1,2,3],reveals=[4,5,6],rewards=[7,8,9]),
            SimulatedInteraction(1,[1,2,3],reveals=[4,5,6],rewards=[0,1,2]),
        ]))

        self.assertEqual("I", transactions[0][0])
        self.assertEqual((0,0), transactions[0][1])
        self.assertEqual({"_packed": { 'reveal':[4,6,5,4], 'reward':[1,6,8,0]}}, transactions[0][2])

    def test_partial_extras(self):

        task = SimulationEvaluationTask(0,0,0,None,RandomLearner(),0)

        transactions = list(task.filter([
            SimulatedInteraction(1,[1,2,3],rewards=[1,2,3]),
            SimulatedInteraction(1,[1,2,3],rewards=[4,5,6], extra=[2,3,4]),
            SimulatedInteraction(1,[1,2,3],rewards=[7,8,9], extra=[2,3,4]),
            SimulatedInteraction(1,[1,2,3],rewards=[0,1,2], extra=[2,3,4]),
        ]))

        self.assertEqual("I", transactions[0][0])
        self.assertEqual((0,0), transactions[0][1])
        self.assertEqual({"_packed": {'reward':[1,6,8,0], 'extra':[None,4,3,2] }}, transactions[0][2])

    def test_sparse_actions(self):

        task = SimulationEvaluationTask(0,0,0,None,RandomLearner(),0)

        transactions = list(task.filter([
            SimulatedInteraction(1,[{'a':1},{'b':2},{'c':3}],reveals=[4,5,6],rewards=[1,2,3]),
            SimulatedInteraction(1,[{'a':1},{'b':2},{'c':3}],reveals=[4,5,6],rewards=[4,5,6]),
            SimulatedInteraction(1,[{'a':1},{'b':2},{'c':3}],reveals=[4,5,6],rewards=[7,8,9]),
            SimulatedInteraction(1,[{'a':1},{'b':2},{'c':3}],reveals=[4,5,6],rewards=[0,1,2]),
        ]))

        self.assertEqual("I", transactions[0][0])
        self.assertEqual((0,0), transactions[0][1])
        self.assertEqual({"_packed": { 'reveal':[4,6,5,4], 'reward':[1,6,8,0]}}, transactions[0][2])

class ProcessTasks_Tests(unittest.TestCase):

    def test_simple(self):

        sim1 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        lrn1 = ModuloLearner("1")
        task = ObserveTask(0, 0, 0, sim1, lrn1)

        list(ProcessTasks().filter([[task]]))

        self.assertEqual(len(task.observed), 5)

    def test_two_tasks_one_source_one_simulation(self):

        sim1 = CountReadSimulation()
        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")

        task1 = ObserveTask(0, 0, 0, sim1, lrn1)
        task2 = ObserveTask(0, 0, 1, sim1, lrn2)

        list(ProcessTasks().filter([[task1, task2]]))

        self.assertEqual(len(task1.observed), 1)
        self.assertEqual(len(task2.observed), 1)

        self.assertEqual(task1.observed[0].context, 0)
        self.assertEqual(task2.observed[0].context, 0)

    def test_two_tasks_two_sources_two_simulations(self):

        sim1 = CountReadSimulation()
        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")

        task1 = ObserveTask(0, 0, 0, sim1, lrn1)
        task2 = ObserveTask(1, 1, 1, sim1, lrn2)

        list(ProcessTasks().filter([[task1, task2]]))

        self.assertEqual(len(task1.observed), 1)
        self.assertEqual(len(task2.observed), 1)

        self.assertEqual(task1.observed[0].context, 0)
        self.assertEqual(task2.observed[0].context, 1)

    def test_two_tasks_one_source_two_simulations(self):

        filter = CountFilter()
        src1 = CountReadSimulation()
        sim1 = Pipe.join(src1, [filter])
        sim2 = Pipe.join(src1, [filter])
        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")

        task1 = ObserveTask(0, 0, 0, sim1, lrn1)
        task2 = ObserveTask(0, 1, 1, sim2, lrn2)

        list(ProcessTasks().filter([[task1, task2]]))

        self.assertEqual(len(task1.observed), 1)
        self.assertEqual(len(task2.observed), 1)

        self.assertEqual(task1.observed[0].context, (0,0))
        self.assertEqual(task2.observed[0].context, (0,1))

class Identify_Tests(unittest.TestCase):

    def test_simple(self):

        source  = ValidationSimulation()
        pipe1   = SimSourceFilters(source,[])
        pipe2   = Pipe.SourceFilters(source,[])
        pipe3   = SimSourceFilters(pipe1,[])
        pipe4   = Pipe.SourceFilters(pipe2,[])
        learner = RandomLearner()

        identifier = Identifier()

        src_id,sim_id,lrn_id = identifier.id(source,learner)

        self.assertEqual(0, src_id)
        self.assertEqual(0, sim_id)
        self.assertEqual(0, lrn_id)

        src_id,sim_id,lrn_id = identifier.id(source,learner)

        self.assertEqual(0, src_id)
        self.assertEqual(0, sim_id)
        self.assertEqual(0, lrn_id)

        src_id,sim_id,lrn_id = identifier.id(pipe1,learner)

        self.assertEqual(0, src_id)
        self.assertEqual(1, sim_id)
        self.assertEqual(0, lrn_id)

        src_id,sim_id,lrn_id = identifier.id(pipe2,learner)

        self.assertEqual(0, src_id)
        self.assertEqual(2, sim_id)
        self.assertEqual(0, lrn_id)

        src_id,sim_id,lrn_id = identifier.id(pipe3,learner)

        self.assertEqual(0, src_id)
        self.assertEqual(3, sim_id)
        self.assertEqual(0, lrn_id)

        src_id,sim_id,lrn_id = identifier.id(pipe4,learner)

        self.assertEqual(0, src_id)
        self.assertEqual(4, sim_id)
        self.assertEqual(0, lrn_id)


if __name__ == '__main__':
    unittest.main()