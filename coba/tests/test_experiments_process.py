import unittest

from itertools import product
from typing import cast, Iterable

from coba.context      import CobaContext, BasicLogger
from coba.environments import LambdaSimulation, Environments, LinearSyntheticSimulation, SupervisedSimulation
from coba.pipes        import Pipes, ListSink, Cache
from coba.evaluators   import SequentialCB
from coba.results      import Result
from coba.primitives   import Learner, SimulatedInteraction

from coba.experiments.process import Task, MakeTasks, ChunkTasks, ProcessTasks

#for testing purposes
class ModuloLearner(Learner):

    n_finish = 0

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

    def finish(self):
        ModuloLearner.n_finish += 1

class ObserveEvaluator:
    def __init__(self) -> None:
        self.observed = []
    def evaluate(self, *args):
        list(args[0].read())
        self.observed = args
        return []

class ExceptionEvaluator:
    def __init__(self):
        self.observed = []

    def evaluate(self, *items):
        raise Exception('ExceptionEvaluator')

class CountReadSimulation:
    def __init__(self) -> None:
        self.n_reads = 0

    def read(self) -> Iterable[SimulatedInteraction]:
        yield SimulatedInteraction(self.n_reads, [0,1], [0,1])
        self.n_reads += 1

    def __str__(self) -> str:
        return "CountRead"

class ExceptionSimulation:
    def __init__(self, params_ex = False, read_ex = False):
        self._params_ex = params_ex
        self._read_ex = read_ex
    @property
    def params(self):
        if self._params_ex:
            raise Exception('ExceptionSimulation.params')
        return {}

    def read(self) -> Iterable[SimulatedInteraction]:
        if self._read_ex:
            raise Exception('ExceptionSimulation.read')
        return []

class ParamObj:
    def __init__(self,**params):
        self.params = params

#for testing purposes

class MakeTasks_Tests(unittest.TestCase):

    def test_eq(self):
        env1 = ParamObj(a=1)
        env2 = ParamObj(a=2)
        self.assertEqual(Task((1,env1),None,None),Task((1,env1),None,None),)
        self.assertNotEqual(Task((1,env1),None,None),Task((1,env2),None,None),)
        self.assertNotEqual(Task((1,env1),None,None),Task((2,env2),None,None),)

    def test_two_env_two_lrn(self):
        env1 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        env2 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")
        evl1 = SequentialCB()

        triples = list(product([env1,env2],[lrn1,lrn2],[evl1]))

        tasks = list(MakeTasks(triples).read())

        self.assertEqual(9, len(tasks))

        self.assertTrue(all([t.lrn_id == 0 for t in tasks if t.lrn is lrn1]))
        self.assertTrue(all([t.lrn_id == 1 for t in tasks if t.lrn is lrn2]))
        self.assertTrue(all([t.env_id == 0 for t in tasks if t.env is env1]))
        self.assertTrue(all([t.env_id == 1 for t in tasks if t.env is env2]))
        self.assertTrue(all([t.val_id == 0 for t in tasks if t.val is evl1]))

        self.assertEqual(1, len([t for t in tasks if not t.env and not t.val and t.lrn is lrn1]))
        self.assertEqual(1, len([t for t in tasks if not t.env and not t.val and t.lrn is lrn2]))

        self.assertEqual(1, len([t for t in tasks if not t.lrn and not t.val and t.env is env1]))
        self.assertEqual(1, len([t for t in tasks if not t.lrn and not t.val and t.env is env2]))

        self.assertEqual(1, len([t for t in tasks if not t.lrn and not t.env and t.val is evl1]))

        for e,l,v in triples:
            self.assertEqual(1, len([t for t in tasks if t.env is e and t.lrn is l and t.val is v and t.copy==True]))

    def test_uneven_pairs(self):
        env1 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        env2 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")
        evl1 = SequentialCB()

        triples = list(product([env1,env2],[lrn1,lrn2],[evl1]))[0:3]
        works   = list(MakeTasks(triples).read())

        self.assertEqual(8, len(works))

        self.assertTrue(all([w.lrn_id == 0 for w in works if w.lrn is lrn1]))
        self.assertTrue(all([w.lrn_id == 1 for w in works if w.lrn is lrn2]))
        self.assertTrue(all([w.env_id == 0 for w in works if w.env is env1]))
        self.assertTrue(all([w.env_id == 1 for w in works if w.env is env2]))

        self.assertEqual(1, len([t for t in works if not t.env and t.lrn is lrn1]))
        self.assertEqual(1, len([t for t in works if not t.env and t.lrn is lrn2]))

        self.assertEqual(1, len([t for t in works if t.env is env1 and not t.lrn]))
        self.assertEqual(1, len([t for t in works if t.env is env2 and not t.lrn]))

        self.assertEqual(1, len([t for t in works if t.env is env1 and t.lrn is lrn1 and t.val is evl1 and t.copy == True]))
        self.assertEqual(1, len([t for t in works if t.env is env2 and t.lrn is lrn1 and t.val is evl1 and t.copy == True]))
        self.assertEqual(1, len([t for t in works if t.env is env1 and t.lrn is lrn2 and t.val is evl1 and t.copy == False]))

    def test_all_finished(self):

        restored = Result(
            [['environment_id','a'],[0,0]],
            [['learner_id','b'],[0,1]],
            [['evaluator_id','c'],[0,2]],
            [['environment_id','learner_id','evaluator_id','index'],[0,0,0,1]])

        triples   = [ (ParamObj(a=0),ParamObj(b=1),ParamObj(c=2))]
        new_tasks = list(MakeTasks(triples,restored).read())

        self.assertEqual([], new_tasks)

    def test_unfinished_environment(self):

        restored = Result(
            None,
            [['learner_id','b'],[0,1]],
            [['evaluator_id','c'],[0,2]],
            [['environment_id','learner_id','evaluator_id','index'],[0,0,0,1]])

        triple   = (ParamObj(a=0),ParamObj(b=1),ParamObj(c=2))
        actual   = list(MakeTasks([triple],restored).read())
        expected = [
            Task((0,triple[0]),None,None),
        ]
        self.assertEqual(expected, actual)

    def test_unfinished_learner(self):

        restored = Result(
            [['environment_id','a'],[0,0]],
            None,
            [['evaluator_id','c'],[0,2]],
            [['environment_id','learner_id','evaluator_id','index'],[0,0,0,1]])

        triple   = (ParamObj(a=0),ParamObj(b=1),ParamObj(c=2))
        actual   = list(MakeTasks([triple],restored).read())
        expected = [
            Task(None,(0,triple[1]),None),
        ]
        self.assertEqual(expected, actual)

    def test_unfinished_evaluator(self):

        restored = Result(
            [['environment_id','a'],[0,0]],
            [['learner_id','b'],[0,1]],
            None,
            [['environment_id','learner_id','evaluator_id','index'],[0,0,0,1]])

        triple   = (ParamObj(a=0),ParamObj(b=1),ParamObj(c=2))
        actual   = list(MakeTasks([triple],restored).read())
        expected = [
            Task(None,None,(0,triple[2])),
        ]
        self.assertEqual(expected, actual)

    def test_unfinished_interactions(self):

        restored = Result(
            [['environment_id','a'],[0,0]],
            [['learner_id','b'],[0,1]],
            [['evaluator_id','c'],[0,2]],
            None)

        triple    = (ParamObj(a=0),ParamObj(b=1),ParamObj(c=2))
        new_tasks = list(MakeTasks([triple],restored).read())

        self.assertEqual([Task((0,triple[0]),(0,triple[1]),(0,triple[2]))], new_tasks)

    def test_unfinished_environment_and_interactions(self):

        restored = Result(
            None,
            [['learner_id','b'],[0,1]],
            [['evaluator_id','c'],[0,2]],
            None)

        triple   = (ParamObj(a=1),ParamObj(b=1),ParamObj(c=2))
        actual   = list(MakeTasks([triple],restored).read())
        expected = [
            Task((0,triple[0]),None,None),
            Task((0,triple[0]),(0,triple[1]),(0,triple[2]))
        ]
        self.assertEqual(expected, actual)

class ChunkTasks_Tests(unittest.TestCase):

    def test_no_chunks(self):
        src1 = Environments.from_linear_synthetic(10)
        src2 = Environments.from_linear_synthetic(10)

        envs = (src1+src2)

        tasks = [
            Task(None, (0,None), None),
            Task(None, (1,None), None),
            Task((1,envs[0]), None, None),
            Task((0,envs[1]), None, None),
            Task((1,envs[0]), (1,None), None),
            Task((0,envs[1]), (0,None), None),
            Task((2,envs[1]), (0,None), None),
            Task((0,envs[1]), (1,None), None),
        ]

        groups = list(ChunkTasks().filter(tasks))

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
            Task(None, (0,None), None),
            Task(None, (1,None), None),
            Task((1,envs[0]), None, None),
            Task((0,envs[1]), None, None),
            Task((1,envs[0]), (1,None), None),
            Task((0,envs[1]), (0,None), None),
            Task((2,envs[1]), (0,None), None),
            Task((0,envs[1]), (1,None), None),
        ]

        groups = list(ChunkTasks().filter(tasks))

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
            Task(None, (0,None), None),
            Task(None, (1,None), None),
            Task((1,envs[0]), None, None),
            Task((0,envs[1]), None, None),
            Task((1,envs[2]), (1,None), None),
            Task((0,envs[1]), (0,None), None),
            Task((2,envs[3]), (0,None), None),
            Task((0,envs[3]), (1,None), None),
        ]

        groups = list(ChunkTasks().filter(tasks))

        self.assertEqual(len(groups), 4)
        self.assertEqual(groups[0], tasks[0:1])
        self.assertEqual(groups[1], tasks[1:2])
        self.assertCountEqual(groups[2], [tasks[3],tasks[5],tasks[7],tasks[6]])
        self.assertCountEqual(groups[3], [tasks[2],tasks[4]])

    def test_max_size_two(self):
        src1 = Environments.from_linear_synthetic(10)
        src2 = Environments.from_linear_synthetic(10)
        envs = (src1+src2).chunk().shuffle(n=2)

        tasks = [
            Task(None, (0,None), None),
            Task(None, (1,None), None),
            Task((1,envs[0]), None, None),
            Task((0,envs[1]), None, None),
            Task((1,envs[2]), (1,None), None),
            Task((0,envs[1]), (0,None), None),
            Task((2,envs[3]), (0,None), None),
            Task((0,envs[3]), (1,None), None),
        ]

        groups = list(ChunkTasks(2).filter(tasks))

        self.assertEqual(len(groups), 5)
        self.assertEqual(groups[0], tasks[0:1])
        self.assertEqual(groups[1], tasks[1:2])
        self.assertCountEqual(groups[2], [tasks[3],tasks[5]])
        self.assertCountEqual(groups[3], [tasks[7],tasks[6]])
        self.assertCountEqual(groups[4], [tasks[2],tasks[4]])

class ProcessTasks_Tests(unittest.TestCase):

    def setUp(self) -> None:
        CobaContext.logger = BasicLogger(ListSink())
        ModuloLearner.n_finish = 0

    def test_simple(self):
        env1 = LambdaSimulation(1, lambda i: i, lambda i,c: [0,1], lambda i,c,a: cast(float,a))
        lrn1 = ModuloLearner("1")
        evl1 = ObserveEvaluator()

        tasks = [Task((1,env1), (1,lrn1), (1,evl1))]
        transactions = list(ProcessTasks().filter(tasks))

        self.assertEqual(list(evl1.observed[0].read()),[{'context':0,'actions':[0,1],'rewards':[0,1]}])
        self.assertIs(evl1.observed[1], lrn1)
        self.assertEqual(['T4', (1,1,1), []], transactions[0])

    def test_env_task(self):
        env1 = SupervisedSimulation([1,2],[1,2],label_type='c')
        env2 = SupervisedSimulation([1,2],[1,2],label_type='c')
        list(env2.read())

        tasks = [Task((1,env1), None, None)]

        transactions = list(ProcessTasks().filter(tasks))
        self.assertEqual(['T1', 1, env2.params], transactions[0])

    def test_environment_reused(self):
        sim1 = Pipes.join(CountReadSimulation(), Cache())

        list(sim1.read())

        lrn1 = ModuloLearner("1")
        lrn2 = ModuloLearner("2")

        val1 = ObserveEvaluator()
        val2 = ObserveEvaluator()

        tasks = [
            Task((0,sim1), None, None),
            Task((0,sim1), (0,lrn1), (0,val1)),
            Task((0,sim1), (1,lrn2), (1,val2))
        ]

        transactions = list(ProcessTasks().filter(tasks))

        self.assertIs(val1.observed[0].env[0], sim1[0])
        self.assertIs(val1.observed[0].env[1], sim1[1])
        self.assertIs(val1.observed[1]       , lrn1   )

        self.assertIs(val2.observed[0].env[0], sim1[0])
        self.assertIs(val2.observed[0].env[1], sim1[1])
        self.assertIs(val2.observed[1]       , lrn2   )

        self.assertEqual(['T1', 0      , {'env_type': 'CountReadSimulation'}], transactions[0])
        self.assertEqual(['T4', (0,0,0), []                                 ], transactions[1])
        self.assertEqual(['T4', (0,1,1), []                                 ], transactions[2])

        self.assertEqual(sim1[0].n_reads, 1)

    def test_task_copy_true(self):
        lrn1 = ModuloLearner("1")

        sim1 = CountReadSimulation()
        sim2 = CountReadSimulation()

        val1 = ObserveEvaluator()
        val2 = ObserveEvaluator()

        tasks = [ Task((0,sim1), (0,lrn1), (0,val1), True), Task((1,sim2), (0,lrn1), (1,val2), True) ]

        list(ProcessTasks().filter(tasks))

        self.assertIsNot(val1.observed[1], lrn1)
        self.assertIsNot(val2.observed[1], lrn1)

        self.assertEqual(2,ModuloLearner.n_finish)

    def test_task_copy_false(self):
        lrn1 = ModuloLearner("1")

        sim1 = CountReadSimulation()
        sim2 = CountReadSimulation()

        task1 = ObserveEvaluator()
        task2 = ObserveEvaluator()

        tasks = [ Task((0,sim1), (0,lrn1), (0,task1), False), Task((1,sim2), (0,lrn1), (1,task2), False) ]

        list(ProcessTasks().filter(tasks))

        self.assertIs(task1.observed[1], lrn1)
        self.assertIs(task2.observed[1], lrn1)

    def test_empty_env_skipped(self):
        lrn1 = ModuloLearner("1")
        src1  = LinearSyntheticSimulation(n_interactions=0)

        task1 = ObserveEvaluator()

        tasks = [ Task((0,src1), (0,lrn1), (0,task1)) ]

        transactions = list(ProcessTasks().filter(tasks))

        self.assertEqual(len(task1.observed), 0)
        self.assertEqual(len(transactions), 0)

    def test_exception_during_tasks(self):
        env0 = ExceptionSimulation(params_ex=True)
        env1 = ExceptionSimulation(read_ex=True)
        env2 = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        lrn1 = ModuloLearner("1")

        val2 = ExceptionEvaluator()
        val3 = ObserveEvaluator()

        CobaContext.logger.sink = ListSink()

        tasks = [Task((0,env0), None, None), Task((0,env1),(0,lrn1), (1,val2)), Task((1,env2),(0,lrn1), (2,val3)) ]

        expected = [ ["T4", (1,0,2), [] ] ]
        actual   = list(ProcessTasks().filter(tasks))

        self.assertIs(val3.observed[1], lrn1)
        self.assertEqual(expected, actual)
        self.assertEqual('ExceptionSimulation.params', str(CobaContext.logger.sink.items[4]))
        self.assertEqual('ExceptionSimulation.read', str(CobaContext.logger.sink.items[7]))

if __name__ == '__main__':
    unittest.main()