from copy import deepcopy
from itertools import groupby, product, count
from collections import defaultdict
from typing import Iterable, Sequence, Any, Optional, Dict, Tuple, Hashable

from coba.random import CobaRandom
from coba.learners import Learner
from coba.config import CobaConfig
from coba.pipes import Pipe, Filter, Source, IdentityFilter
from coba.simulations import Context, Action, Key, Simulation

from coba.benchmarks.transactions import Transaction
from coba.benchmarks.results import Result

class BenchmarkTask:

    class BenchmarkTaskLearner(Learner):

        @property
        def family(self) -> str:
            try:
                return self._learner.family
            except AttributeError:
                return self._learner.__class__.__name__

        @property
        def params(self) -> Dict[str, Any]:
            try:
                return self._learner.params
            except AttributeError:
                return {}

        def __init__(self, learner: Learner, seed: Optional[int]) -> None:
            self._learner = learner
            self._random  = CobaRandom(seed)

        def predict(self, key: Key, context: Context, actions: Sequence[Action]) -> Sequence[float]:
            return self._learner.predict(key, context, actions) #type: ignore

        def learn(self, key: Key, context: Context, action: Action, reward: float, probability: float) -> None:
            self._learner.learn(key, context, action, reward, probability) #type: ignore

    class BenchmarkTaskSimulation(Source[Simulation]):

        def __init__(self, pipe: Source[Simulation]) -> None:
            self._pipe = pipe

        @property
        def source(self) -> Source[Simulation]:
            return self._pipe._source if isinstance(self._pipe, (Pipe.SourceFilters)) else self._pipe

        @property
        def filter(self) -> Filter[Simulation,Simulation]:
            return self._pipe._filter if isinstance(self._pipe, Pipe.SourceFilters) else IdentityFilter()

        def read(self) -> Simulation:
            return self._pipe.read()

        def __repr__(self) -> str:
            return self._pipe.__repr__()

    def __init__(self, src_id:int, sim_id: int, lrn_id: int, simulation: Source[Simulation], learner: Learner, seed: int = None) -> None:
        self.src_id     = src_id
        self.sim_id     = sim_id
        self.lrn_id     = lrn_id
        
        self.seed       = seed
        self.simulation = BenchmarkTask.BenchmarkTaskSimulation(simulation)
        self.learner    = BenchmarkTask.BenchmarkTaskLearner(learner, seed)

class Tasks(Source[Iterable[BenchmarkTask]]):

    def __init__(self, simulations: Sequence[Source[Simulation]], learners: Sequence[Learner], seed: int = None) -> None:
        self._simulations = simulations
        self._learners    = learners
        self._seed        = seed

    def read(self) -> Iterable[BenchmarkTask]:

        #we rely on sim_id to make sure we don't do duplicate work. So long as self._simulations
        #is always in the exact same order we should be fine. In the future we may want to consider.
        #adding a better check for simulations other than assigning an index based on their order.

        source_ids: Dict[Hashable, int]  = defaultdict(lambda x=count(): next(x)) # type: ignore

        for (sim_id,sim), (lrn_id,lrn) in product(enumerate(self._simulations), enumerate(self._learners)):
            sim_source = sim._source if isinstance(sim, (Pipe.SourceFilters)) else sim
            yield BenchmarkTask(source_ids[sim_source], sim_id, lrn_id, sim, lrn, self._seed)                

class Unfinished(Filter[Iterable[BenchmarkTask], Iterable[BenchmarkTask]]):
    def __init__(self, restored: Result) -> None:
        self._restored = restored

    def filter(self, tasks: Iterable[BenchmarkTask]) -> Iterable[BenchmarkTask]:

        def is_not_complete(sim_id: int, learn_id: int):
            return (sim_id,learn_id) not in self._restored._interactions

        for task in tasks:
            if is_not_complete(task.sim_id, task.lrn_id):
                yield task

class ChunkBySource(Filter[Iterable[BenchmarkTask], Iterable[Iterable[BenchmarkTask]]]):

    def filter(self, tasks: Iterable[BenchmarkTask]) -> Iterable[Iterable[BenchmarkTask]]:

        srt_key = lambda t: t.src_id
        grp_key = lambda t: t.src_id

        tasks = list(tasks)

        for _, group in groupby(sorted(tasks, key=srt_key), key=grp_key):
            yield list(group)

class ChunkByTask(Filter[Iterable[BenchmarkTask], Iterable[Iterable[BenchmarkTask]]]):

    def filter(self, tasks: Iterable[BenchmarkTask]) -> Iterable[Iterable[BenchmarkTask]]:

        for task in tasks:
            yield [ task ]

class ChunkByNone(Filter[Iterable[BenchmarkTask], Iterable[Iterable[BenchmarkTask]]]):

    def filter(self, tasks: Iterable[BenchmarkTask]) -> Iterable[Iterable[BenchmarkTask]]:
        yield list(tasks)

class Transactions(Filter[Iterable[Iterable[BenchmarkTask]], Iterable[Any]]):

    def filter(self, chunks: Iterable[Iterable[BenchmarkTask]]) -> Iterable[Any]:

        for chunk in chunks:
            for transaction in self._process_chunk(chunk):
                yield transaction

    def _process_chunk(self, task_group: Iterable[BenchmarkTask]) -> Iterable[Any]:

        source_by_id = { t.src_id: t.simulation.source for t in task_group }
        filter_by_id = { t.sim_id: t.simulation.filter for t in task_group }

        srt_src = lambda t: t.src_id
        grp_src = lambda t: t.src_id
        srt_sim = lambda t: t.sim_id
        grp_sim = lambda t: t.sim_id

        with CobaConfig.Logger.log(f"Processing chunk..."):

            for src_id, tasks_by_src in groupby(sorted(task_group, key=srt_src), key=grp_src):

                try:

                    with CobaConfig.Logger.time(f"Creating source {src_id} from {source_by_id[src_id]}..."):
                        loaded_source = source_by_id[src_id].read()

                    for sim_id, tasks_by_src_sim in groupby(sorted(tasks_by_src, key=srt_sim), key=grp_sim):

                        tasks_by_src_sim_list = list(tasks_by_src_sim)
                        learner_ids           = [t.lrn_id  for t in tasks_by_src_sim_list]
                        learners              = [t.learner for t in tasks_by_src_sim_list]
                        seeds                 = [t.seed    for t in tasks_by_src_sim_list]

                        learner_ids.reverse()
                        learners.reverse() 

                        with CobaConfig.Logger.time(f"Creating simulation {sim_id} from source {src_id}..."):
                            simulation = filter_by_id[sim_id].filter(loaded_source)

                        if not simulation.interactions:
                            CobaConfig.Logger.log(f"Simulation {sim_id} has nothing to evaluate (likely due to `take` being larger than the simulation).")
                            continue

                        for i in sorted(range(len(learners)), reverse=True):

                            lrn_id  = learner_ids[i]
                            learner = deepcopy(learners[i])
                            random  = CobaRandom(seeds[i])

                            try:
                                with CobaConfig.Logger.time(f"Evaluating learner {lrn_id} on Simulation {sim_id}..."):

                                    rewards = []

                                    for interaction in simulation.interactions:
                                        probs  = learner.predict(interaction.key, interaction.context, interaction.actions)
                                        
                                        assert abs(sum(probs) - 1) < .0001, "The learner returned invalid proabilities for action choices."
                                        
                                        action = random.choice(interaction.actions, probs)
                                        reward = simulation.reward.observe([(interaction.key, interaction.context, action)])[0]
                                        prob   = probs[interaction.actions.index(action)]
                                        
                                        learner.learn(interaction.key, interaction.context, action, reward, prob)
                                        rewards.append(reward)

                                    yield Transaction.interactions(sim_id, lrn_id, _packed={"reward":rewards})

                            except Exception as e:
                                CobaConfig.Logger.log_exception(e)

                            finally:
                                del learner_ids[i]
                                del learners[i]

                except Exception as e:
                    CobaConfig.Logger.log_exception(e)