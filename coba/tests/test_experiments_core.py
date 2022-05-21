import unittest
import math

from pathlib import Path
from typing import cast

from coba.environments import Environment, LambdaSimulation
from coba.experiments.tasks import OnlineOnPolicyEvalTask
from coba.pipes import Source, ListSink
from coba.learners import Learner
from coba.contexts import CobaContext, InteractionContext, CobaContext, IndentLogger, BasicLogger, NullLogger
from coba.experiments import Experiment

class NoParamsLearner:
    def predict(self, context, actions):
        return [ int(i == actions.index(actions[context%len(actions)])) for i in range(len(actions)) ]
    def learn(self, context, action, reward, probability, info):
        pass

class NoParamsEnvironment:
    def read(self):
        return LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a)).read()

class ModuloLearner(Learner):
    def __init__(self, param:str="0"):
        self._param = param

    @property
    def params(self):
        return {"family": "Modulo", "p":self._param}

    def predict(self, context, actions):
        return [ int(i == actions.index(actions[context%len(actions)])) for i in range(len(actions)) ]

    def learn(self, context, action, reward, probability, info):
        pass

class BrokenLearner(Learner):

    @property
    def params(self):
        return {"family": "Broken"}

    def predict(self, context, actions):
        raise Exception("Broken Learner")

    def learn(self, context, action, reward, probability, info):
        pass

class PredictInfoLearner(Learner):
    def __init__(self, param:str="0"):
        self._param = param

    @property
    def params(self):
        return {"family": "Modulo", "p":self._param}

    def predict(self, context, actions):
        return [ int(i == actions.index(actions[context%len(actions)])) for i in range(len(actions)) ], (0,1)

    def learn(self, context, action, reward, probability, info):
        assert info == (0,1)

class LearnInfoLearner(Learner):
    def __init__(self, param:str="0"):
        self._param = param

    @property
    def params(self):
        return {"family": "Modulo", "p":self._param}

    def predict(self, context, actions):
        return [ int(i == actions.index(actions[context%len(actions)])) for i in range(len(actions)) ]

    def learn(self, context, action, reward, probability, info):
        InteractionContext.learner_info.update({"Modulo": self._param})

class NotPicklableLearner(ModuloLearner):
    def __init__(self):
        self._val = lambda x: 1
        super().__init__()

class NotPicklableLearnerWithReduce(NotPicklableLearner):
    def __init__(self):
        self._val = lambda x: 1
        super().__init__()

    def __reduce__(self):
        return (NotPicklableLearnerWithReduce, ())

class WrappedLearner(Learner):

    @property
    def params(self):
        return {"family": "family"}

    def __init__(self, learner):
        self._learner = learner

    def predict(self, key, context, actions):
        return self._learner.predict(key, context, actions)

    def learn(self, key, context, action, reward, probability) -> None:
        return self._learner.learn(key, context, action, reward, probability)

class OneTimeSource(Source):

    def __init__(self, source: Source) -> None:
        self._source = source
        self._read_count = 0

    def read(self):
        if self._read_count > 0: raise Exception("Read more than once")

        self._read_count += 1

        return self._source.read()

class ExceptionEnvironment(Environment):

    def __init__(self, exception: Exception):
        self._exc = exception

    def params(self):
        return {}

    def read(self):
        raise self._exc

class Experiment_Single_Tests(unittest.TestCase):

    @classmethod
    def setUp(cls) -> None:
        CobaContext.logger = NullLogger()
        CobaContext.experiment.processes = 1
        CobaContext.experiment.maxchunksperchild = 0

    def test_sim(self):
        sim1       = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner    = ModuloLearner()
        experiment = Experiment([sim1], [learner], evaluation_task=OnlineOnPolicyEvalTask(False))

        result              = experiment.evaluate()
        actual_learners     = result.learners.to_dicts()
        actual_environments = result.environments.to_dicts()
        actual_interactions = result.interactions.to_dicts()

        expected_learners     = [
            {"learner_id":0, "family":"Modulo", "full_name":"Modulo(p=0)", "p":'0'}
        ]
        expected_environments = [
            {"environment_id":0, "type":'LambdaSimulation'}
        ]
        expected_interactions = [
            {"environment_id":0, "learner_id":0, "index":1, "reward":0, "min_reward":0, "max_reward":2, 'min_rank':1, 'max_rank':3, "rank":3, "n_actions":3},
            {"environment_id":0, "learner_id":0, "index":2, "reward":1, "min_reward":0, "max_reward":2, 'min_rank':1, 'max_rank':3, "rank":2, "n_actions":3}
        ]

        self.assertDictEqual({"description":None, "n_learners":1, "n_environments":1}, result.experiment)
        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_environments, expected_environments)
        self.assertCountEqual(actual_interactions, expected_interactions)

    def test_sims(self):
        sim1       = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        sim2       = LambdaSimulation(3, lambda i: i, lambda i,c: [3,4,5], lambda i,c,a: cast(float,a))
        learner    = ModuloLearner()
        experiment = Experiment([sim1,sim2], [learner], "abc", evaluation_task=OnlineOnPolicyEvalTask(False))

        result              = experiment.evaluate()
        actual_learners     = result.learners.to_dicts()
        actual_environments = result.environments.to_dicts()
        actual_interactions = result.interactions.to_dicts()

        expected_learners     = [
            {"learner_id":0, "family":"Modulo", "full_name":"Modulo(p=0)", "p":'0'}
        ]
        expected_environments = [
            {"environment_id":0, "type":'LambdaSimulation'},
            {"environment_id":1, "type":'LambdaSimulation'}
        ]
        expected_interactions = [
            {"environment_id":0, "learner_id":0, "index":1, "reward":0, "min_reward":0, "max_reward":2, 'min_rank':1, 'max_rank':3, "rank":3, "n_actions":3},
            {"environment_id":0, "learner_id":0, "index":2, "reward":1, "min_reward":0, "max_reward":2, 'min_rank':1, 'max_rank':3, "rank":2, "n_actions":3},
            {"environment_id":1, "learner_id":0, "index":1, "reward":3, "min_reward":3, "max_reward":5, 'min_rank':1, 'max_rank':3, "rank":3, "n_actions":3},
            {"environment_id":1, "learner_id":0, "index":2, "reward":4, "min_reward":3, "max_reward":5, 'min_rank':1, 'max_rank':3, "rank":2, "n_actions":3},
            {"environment_id":1, "learner_id":0, "index":3, "reward":5, "min_reward":3, "max_reward":5, 'min_rank':1, 'max_rank':3, "rank":1, "n_actions":3}
        ]

        self.assertDictEqual({"description":"abc", "n_learners":1, "n_environments":2}, result.experiment)
        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_environments, expected_environments)
        self.assertCountEqual(actual_interactions, expected_interactions)

    def test_learners(self):
        sim        = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner1   = ModuloLearner("0") #type: ignore
        learner2   = ModuloLearner("1") #type: ignore
        experiment = Experiment([sim], [learner1, learner2], evaluation_task=OnlineOnPolicyEvalTask(False))

        expected_learners     = [
            {"learner_id":0, "family":"Modulo", "full_name":"Modulo(p=0)", "p":'0'},
            {"learner_id":1, "family":"Modulo", "full_name":"Modulo(p=1)", "p":'1'}
        ]
        expected_environments = [
            {"environment_id":0, "type":'LambdaSimulation'},
        ]
        expected_interactions = [
            {"environment_id":0, "learner_id":0, "index":1, "reward":0, "min_reward":0, "max_reward":2, 'min_rank':1, 'max_rank':3, "rank":3, "n_actions":3},
            {"environment_id":0, "learner_id":0, "index":2, "reward":1, "min_reward":0, "max_reward":2, 'min_rank':1, 'max_rank':3, "rank":2, "n_actions":3},
            {"environment_id":0, "learner_id":1, "index":1, "reward":0, "min_reward":0, "max_reward":2, 'min_rank':1, 'max_rank':3, "rank":3, "n_actions":3},
            {"environment_id":0, "learner_id":1, "index":2, "reward":1, "min_reward":0, "max_reward":2, 'min_rank':1, 'max_rank':3, "rank":2, "n_actions":3},

        ]

        result              = experiment.evaluate()
        actual_learners     = result._learners.to_dicts()
        actual_environments = result._environments.to_dicts()
        actual_interactions = result.interactions.to_dicts()

        self.assertDictEqual({"description":None, "n_learners":2, "n_environments":1}, result.experiment)
        self.assertEqual(actual_learners, expected_learners)
        self.assertEqual(actual_environments, expected_environments)
        self.assertEqual(actual_interactions, expected_interactions)

    def test_learner_info(self):
        sim        = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner1   = LearnInfoLearner("0") #type: ignore
        experiment = Experiment([sim],[learner1], evaluation_task=OnlineOnPolicyEvalTask(False))

        actual_result       = experiment.evaluate()
        actual_learners     = actual_result._learners.to_dicts()
        actual_environments = actual_result._environments.to_dicts()
        actual_interactions = actual_result.interactions.to_dicts()

        expected_learners     = [
            {"learner_id":0, "family":"Modulo", "full_name":"Modulo(p=0)", "p":'0'}
        ]
        expected_environments = [
            {"environment_id":0, "type":'LambdaSimulation'},
        ]
        expected_interactions = [
            {"environment_id":0, "learner_id":0, "index":1, "reward":0, "min_reward":0, "max_reward":2, 'min_rank':1, 'max_rank':3, "rank":3, "n_actions":3, "Modulo":"0"},
            {"environment_id":0, "learner_id":0, "index":2, "reward":1, "min_reward":0, "max_reward":2, 'min_rank':1, 'max_rank':3, "rank":2, "n_actions":3, "Modulo":"0"},
        ]

        self.assertDictEqual({"description":None, "n_learners":1, "n_environments":1}, actual_result.experiment)
        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_environments, expected_environments)
        self.assertCountEqual(actual_interactions, expected_interactions)

    def test_predict_info(self):
        sim        = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner1   = PredictInfoLearner("0") #type: ignore
        experiment = Experiment([sim],[learner1],evaluation_task=OnlineOnPolicyEvalTask(False))

        actual_result       = experiment.evaluate()
        
        actual_learners     = actual_result._learners.to_dicts()
        actual_environments = actual_result._environments.to_dicts()
        actual_interactions = actual_result.interactions.to_dicts()

        expected_learners     = [
            {"learner_id":0, "family":"Modulo", "full_name":"Modulo(p=0)", "p":'0'}
        ]
        expected_environments = [
            {"environment_id":0, "type":'LambdaSimulation'},
        ]
        expected_interactions = [
            {"environment_id":0, "learner_id":0, "index":1, "reward":0, "min_reward":0, "max_reward":2, 'min_rank':1, 'max_rank':3, "rank":3, "n_actions":3},
            {"environment_id":0, "learner_id":0, "index":2, "reward":1, "min_reward":0, "max_reward":2, 'min_rank':1, 'max_rank':3, "rank":2, "n_actions":3},
        ]

        self.assertDictEqual({"description":None, "n_learners":1, "n_environments":1}, actual_result.experiment)
        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_environments, expected_environments)
        self.assertCountEqual(actual_interactions, expected_interactions)

    def test_transaction_resume_1(self):
        sim             = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        working_learner = ModuloLearner()
        broken_learner  = BrokenLearner()

        #the second Experiment shouldn't ever call broken_factory() because
        #we're resuming from the first experiment's transaction.log
        try:
            first_result  = Experiment([sim],[working_learner],evaluation_task=OnlineOnPolicyEvalTask(False)).evaluate("coba/tests/.temp/transactions.log")
            second_result = Experiment([sim],[broken_learner ],evaluation_task=OnlineOnPolicyEvalTask(False)).evaluate("coba/tests/.temp/transactions.log")
        finally:
            if Path('coba/tests/.temp/transactions.log').exists(): Path('coba/tests/.temp/transactions.log').unlink()

        actual_learners     = second_result.learners.to_dicts()
        actual_environments = second_result.environments.to_dicts()
        actual_interactions = second_result.interactions.to_dicts()

        expected_learners     = [
            {"learner_id":0, "family":"Modulo", "full_name":"Modulo(p=0)", "p":'0'}
        ]
        expected_environments = [
            {"environment_id":0, "type":'LambdaSimulation'},
        ]
        expected_interactions = [
            {"environment_id":0, "learner_id":0, "index":1, "reward":0, "min_reward":0, "max_reward":2, 'min_rank':1, 'max_rank':3, "rank":3, "n_actions":3},
            {"environment_id":0, "learner_id":0, "index":2, "reward":1, "min_reward":0, "max_reward":2, 'min_rank':1, 'max_rank':3, "rank":2, "n_actions":3},
        ]

        self.assertDictEqual({"description":None, "n_learners":1, "n_environments":1}, second_result.experiment)
        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_environments, expected_environments)
        self.assertCountEqual(actual_interactions, expected_interactions)

    def test_no_params(self):
        sim1       = NoParamsEnvironment()
        learner    = NoParamsLearner()
        experiment = Experiment([sim1], [learner], evaluation_task=OnlineOnPolicyEvalTask(False))

        result              = experiment.evaluate()
        actual_learners     = result.learners.to_dicts()
        actual_environments = result.environments.to_dicts()
        actual_interactions = result.interactions.to_dicts()

        expected_learners     = [
            {"learner_id":0, "family":"NoParamsLearner", "full_name":"NoParamsLearner" }
        ]
        expected_environments = [
            {"environment_id":0, "type":'NoParamsEnvironment'},
        ]
        expected_interactions = [
            {"environment_id":0, "learner_id":0, "index":1, "reward":0, "min_reward":0, "max_reward":2, 'min_rank':1, 'max_rank':3, "rank":3, "n_actions":3},
            {"environment_id":0, "learner_id":0, "index":2, "reward":1, "min_reward":0, "max_reward":2, 'min_rank':1, 'max_rank':3, "rank":2, "n_actions":3},
        ]

        self.assertDictEqual({"description":None, "n_learners":1, "n_environments":1}, result.experiment)
        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_environments, expected_environments)
        self.assertCountEqual(actual_interactions, expected_interactions)

    def test_ignore_raise(self):

        CobaContext.logger = IndentLogger(ListSink())

        sim1       = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        sim2       = LambdaSimulation(3, lambda i: i, lambda i,c: [3,4,5], lambda i,c,a: cast(float,a))
        experiment = Experiment([sim1,sim2], [ModuloLearner(), BrokenLearner()],evaluation_task=OnlineOnPolicyEvalTask(False))

        result              = experiment.evaluate()
        actual_learners     = result.learners.to_dicts()
        actual_environments = result.environments.to_dicts()
        actual_interactions = result.interactions.to_dicts()

        expected_learners     = [
            {"learner_id":0, "family":"Modulo", "full_name":"Modulo(p=0)", "p":'0'},
            {"learner_id":1, "family":"Broken", "full_name":"Broken", "p":None    }
        ]
        expected_environments = [
            {"environment_id":0, "type":'LambdaSimulation'},
            {"environment_id":1, "type":'LambdaSimulation'},
        ]
        expected_interactions = [
            {"environment_id":0, "learner_id":0, "index":1, "reward":0, "min_reward":0, "max_reward":2, 'min_rank':1, 'max_rank':3, "rank":3, "n_actions":3},
            {"environment_id":0, "learner_id":0, "index":2, "reward":1, "min_reward":0, "max_reward":2, 'min_rank':1, 'max_rank':3, "rank":2, "n_actions":3},
            {"environment_id":1, "learner_id":0, "index":1, "reward":3, "min_reward":3, "max_reward":5, 'min_rank':1, 'max_rank':3, "rank":3, "n_actions":3},
            {"environment_id":1, "learner_id":0, "index":2, "reward":4, "min_reward":3, "max_reward":5, 'min_rank':1, 'max_rank':3, "rank":2, "n_actions":3},
            {"environment_id":1, "learner_id":0, "index":3, "reward":5, "min_reward":3, "max_reward":5, 'min_rank':1, 'max_rank':3, "rank":1, "n_actions":3},
        ]

        self.assertIsInstance(CobaContext.logger, IndentLogger)
        self.assertEqual(2, sum([int("Unexpected exception:" in item) for item in CobaContext.logger.sink.items]))

        self.assertDictEqual({"description":None, "n_learners":2, "n_environments":2}, result.experiment)
        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_environments, expected_environments)
        self.assertCountEqual(actual_interactions, expected_interactions)

    def test_config_set(self):

        exp = Experiment([], [])

        CobaContext.experiment.processes = 10
        self.assertEqual(10,exp.processes)

        CobaContext.experiment.maxchunksperchild = 3
        self.assertEqual(3,exp.maxchunksperchild)

        CobaContext.experiment.maxtasksperchunk = 2
        self.assertEqual(2,exp.maxtasksperchunk)

        CobaContext.experiment.chunk_by = 'source'
        self.assertEqual('source',exp.chunk_by)

        exp.config(processes=2, maxchunksperchild=5, maxtasksperchunk=3, chunk_by='task')
        self.assertEqual(2,exp.processes)
        self.assertEqual(5,exp.maxchunksperchild)
        self.assertEqual(3,exp.maxtasksperchunk)
        self.assertEqual('task',exp.chunk_by)

    def test_restore_not_matched_environments(self):

        path = Path("coba/tests/.temp/experiment.log")

        if path.exists(): path.unlink()
        path.write_text('["version",4]\n["experiment",{"n_environments":1,"n_learners":1}]')

        try:
            sim1       = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
            learner    = ModuloLearner()

            with self.assertRaises(AssertionError) as e:
                result = Experiment([sim1,sim1], [learner]).evaluate(str(path))

            with self.assertRaises(AssertionError) as e:
                result = Experiment([sim1], [learner,learner]).evaluate(str(path))

        finally:
            path.unlink()

class Experiment_Multi_Tests(Experiment_Single_Tests):

    @classmethod
    def setUp(cls) -> None:
        CobaContext.logger = NullLogger()
        CobaContext.experiment.processes = 2
        CobaContext.experiment.maxchunksperchild = 0

    def test_not_picklable_learner_sans_reduce(self):
        sim1       = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner    = NotPicklableLearner()
        experiment = Experiment([sim1],[learner])

        CobaContext.logger = BasicLogger(ListSink())

        experiment.evaluate()

        self.assertEqual(1, len(CobaContext.logger.sink.items))
        self.assertIn("pickle", CobaContext.logger.sink.items[0])

    def test_wrapped_not_picklable_learner_sans_reduce(self):
        sim1       = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner    = WrappedLearner(NotPicklableLearner())
        experiment = Experiment([sim1],[learner])

        CobaContext.logger = BasicLogger(ListSink())

        experiment.evaluate()

        self.assertEqual(1, len(CobaContext.logger.sink.items))
        self.assertIn("pickle", CobaContext.logger.sink.items[0])

    def test_not_picklable_learner_with_reduce(self):
        sim1       = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner    = NotPicklableLearnerWithReduce()
        experiment = Experiment([sim1],[learner])

        experiment.evaluate()

    def test_wrapped_not_picklable_learner_with_reduce(self):
        sim1       = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner    = WrappedLearner(NotPicklableLearnerWithReduce())
        experiment = Experiment([sim1],[learner])

        experiment.evaluate()

if __name__ == '__main__':
    unittest.main()