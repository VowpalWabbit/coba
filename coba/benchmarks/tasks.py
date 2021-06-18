from copy import deepcopy
from statistics import mean
from itertools import groupby, product, count, chain
from collections import defaultdict
from statistics import median
from typing import Iterable, Sequence, Any, Optional, Dict, Tuple, Hashable

from coba.random import CobaRandom
from coba.learners import Learner
from coba.config import CobaConfig
from coba.pipes import Pipe, Filter, Source, IdentityFilter
from coba.simulations import Context, Action, Key, Interaction, Simulation, BatchedSimulation

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

        def choose(self, key: Key, context: Context, actions: Sequence[Action]) -> Tuple[Action, float]:
            p = self.predict(key,context,actions)
            c = list(zip(actions,p))

            assert abs(sum(p) - 1) < .0001, "The learner returned invalid proabilities for action choices."

            return self._random.choice(c, p)
        
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
            yield BenchmarkTask(source_ids[sim_source], sim_id, lrn_id, sim, deepcopy(lrn), self._seed)                

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

    def filter(self, task_chunks: Iterable[Iterable[BenchmarkTask]]) -> Iterable[Any]:

        for task_chunk in task_chunks:
            for transaction in self._process_group(task_chunk):
                yield transaction

    def _process_group(self, task_group: Iterable[BenchmarkTask]) -> Iterable[Any]:

        def batchify(simulation: Simulation) -> Sequence[Sequence[Interaction]]:
            if isinstance(simulation, BatchedSimulation):
                return simulation.interaction_batches
            else:
                return [ [interaction] for interaction in simulation.interactions ]

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

                        with CobaConfig.Logger.time(f"Creating simulation {sim_id} from source {src_id}..."):
                            simulation = filter_by_id[sim_id].filter(loaded_source)
                            batches    = batchify(simulation)

                        if not batches:
                            CobaConfig.Logger.log(f"Simulation {sim_id} has nothing to evaluate (likely due to `take` being larger than the simulation).")
                            continue

                        for i in sorted(range(len(learners)), reverse=True):

                            lrn_id  = learner_ids[i]
                            learner = learners[i]

                            try:
                                with CobaConfig.Logger.time(f"Evaluating learner {lrn_id} on Simulation {sim_id}..."):
                                    context_sizes = [ int(median(self._context_sizes(batch)))                for batch in batches ]
                                    action_counts = [ int(median(self._action_counts(batch)))                for batch in batches ]
                                    batch_sizes   = [ len(batch)                                             for batch in batches ]
                                    mean_rewards  = [ self._process_batch(batch, learner, simulation.reward) for batch in batches ]
                                    yield Transaction.batch(sim_id, lrn_id, C=context_sizes, A=action_counts, N=batch_sizes, reward=mean_rewards)
                            
                            except Exception as e:
                                CobaConfig.Logger.log_exception(e)

                            finally:
                                del learner_ids[i]
                                del learners[i]
                
                except Exception as e:
                    CobaConfig.Logger.log_exception(e)


    def _process_batch(self, batch, learner, reward) -> float:
        
        keys     = []
        contexts = []
        actions  = []
        probs    = []

        for interaction in batch:

            action, prob = learner.choose(interaction.key, interaction.context, interaction.actions)

            keys    .append(interaction.key)
            contexts.append(interaction.context)
            probs   .append(prob)
            actions .append(action)

        rewards = reward.observe(list(zip(keys, contexts, actions))) 

        for (key,context,action,reward,prob) in zip(keys,contexts,actions,rewards,probs):
            learner.learn(key,context,action,reward,prob)

        return round(mean(rewards),5)

    def _context_sizes(self, interactions) -> Iterable[int]:
        for context in [i.context for i in interactions]:
            yield 0 if context is None else len(context) if isinstance(context,tuple) else 1

    def _action_counts(self, interactions) -> Iterable[int]:
        if len(interactions) == 0:
            yield 0

        for actions in [i.actions for i in interactions]:
            yield len(actions)
