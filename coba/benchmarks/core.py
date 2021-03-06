from copy import deepcopy
from statistics import mean
from itertools import groupby, product, count, chain
from statistics import median
from typing import Iterable, Tuple, Sequence, Dict, Any, cast, Optional, overload, List, Union

from coba.random import CobaRandom
from coba.learners import Learner
from coba.simulations import Context, Action, Key, Interaction, Simulation, BatchedSimulation, Take, Shuffle, Batch
from coba.tools import CobaRegistry, CobaConfig
from coba.data.filters import Filter, IdentityFilter, JsonDecode, ResponseToText
from coba.data.sources import HttpSource, Source, MemorySource, DiskSource
from coba.data.pipes import Pipe

from coba.benchmarks.results import Result, Transaction, TransactionSink

class BenchmarkLearner(Learner):

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

    def init(self) -> None:
        try:
            self._learner.init()
        except AttributeError:
            pass

    def choose(self, key: Key, context: Context, actions: Sequence[Action]) -> Tuple[Action, float]:
        p = self.predict(key,context,actions)
        c = list(zip(actions,p))
        
        return self._random.choice(c, p)
    
    def predict(self, key: Key, context: Context, actions: Sequence[Action]) -> Sequence[float]:
        return self._learner.predict(key, context, actions)

    def learn(self, key: Key, context: Context, action: Action, reward: float, probability: float) -> None:
        self._learner.learn(key, context, action, reward, probability)

class BenchmarkSimulation(Source[Simulation]):

    def __init__(self, source: Source[Simulation], filters: Sequence[Filter[Simulation,Simulation]] = None) -> None:
        self._pipe = source if filters is None else Pipe.join(source, filters)

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

class BenchmarkTask:
    def __init__(self, src_id:int, sim_id: int, lrn_id: int, simulation: BenchmarkSimulation, learner: BenchmarkLearner) -> None:
        self.src_id     = src_id
        self.sim_id     = sim_id
        self.lrn_id     = lrn_id
        self.simulation = simulation
        self.learner    = learner

class Tasks(Source[Iterable[BenchmarkTask]]):
    
    def __init__(self, simulations: Sequence[Source[Simulation]], learners: Sequence[Learner], seed: int = None) -> None:
        self._simulations = simulations
        self._learners    = learners
        self._seed        = seed

    def read(self) -> Iterable[BenchmarkTask]:

        benchmark_sims = [BenchmarkSimulation(sim)                       for sim in self._simulations]
        benchmark_lrns = [BenchmarkLearner(lrn, self._seed) for lrn in self._learners   ]

        #this could be made more sophisticated in the future
        sources    = set([ s.source for s in benchmark_sims ])        
        source_ids = cast(Dict[Source[Simulation],int], { src:idx for src,idx in zip(sources, count()) })

        for (sim_id,sim), (lrn_id,lrn) in product(enumerate(benchmark_sims), enumerate(benchmark_lrns)):
            yield BenchmarkTask(source_ids[sim.source], sim_id, lrn_id, sim, deepcopy(lrn))                

class Unfinished(Filter[Iterable[BenchmarkTask], Iterable[BenchmarkTask]]):
    def __init__(self, restored: Result) -> None:
        self._restored = restored

    def filter(self, tasks: Iterable[BenchmarkTask]) -> Iterable[BenchmarkTask]:
        
        no_batch_sim_rows = self._restored.simulations.get_where(batch_count=0)
        no_batch_sim_ids  = [ row['simulation_id'] for row in no_batch_sim_rows ]

        def is_not_complete(sim_id: int, learn_id: int):
            return (sim_id,learn_id) not in self._restored.batches and sim_id not in no_batch_sim_ids

        for task in tasks:
            if is_not_complete(task.sim_id, task.lrn_id):
                yield task

class GroupBySource(Filter[Iterable[BenchmarkTask], Iterable[Iterable[BenchmarkTask]]]):

    def filter(self, tasks: Iterable[BenchmarkTask]) -> Iterable[Iterable[BenchmarkTask]]:

        srt_key = lambda t: t.src_id
        grp_key = lambda t: t.src_id

        tasks = list(tasks)

        for _, group in groupby(sorted(tasks, key=srt_key), key=grp_key):
            a = list(group)
            yield a

class GroupByNone(Filter[Iterable[BenchmarkTask], Iterable[Iterable[BenchmarkTask]]]):

    def filter(self, tasks: Iterable[BenchmarkTask]) -> Iterable[Iterable[BenchmarkTask]]:

        for task in tasks:
            yield [ task ]

class Transactions(Filter[Iterable[Iterable[BenchmarkTask]], Iterable[Any]]):

    def __init__(self, ignore_raise: bool) -> None:
        self._ignore_raise = ignore_raise

    def filter(self, task_groups: Iterable[Iterable[BenchmarkTask]]) -> Iterable[Any]:
        
        task_group = chain.from_iterable(task_groups)

        for transaction in self._process_group(task_group):
            yield transaction

    def _process_group(self, task_group: Iterable[BenchmarkTask]) -> Iterable[Any]:

        def batchify(simulation: Simulation) -> Sequence[Sequence[Interaction]]:
            if isinstance(simulation, BatchedSimulation):
                return simulation.interaction_batches
            else:
                return [ [interaction] for interaction in simulation.interactions ]

        def sim_transaction(sim_id, pipe, interactions, batches):
            return Transaction.simulation(sim_id,
                pipe              = str(pipe),
                interaction_count = len(interactions),
                batch_count       = len(batches),
                context_size      = int(median(self._context_sizes(interactions))),
                action_count      = int(median(self._action_counts(interactions))))        

        srt_src = lambda t: t.src_id
        grp_src = lambda t: (t.src_id, t.simulation.source)
        srt_sim = lambda t: t.sim_id
        grp_sim = lambda t: (t.sim_id, t.simulation)

        try:
            for (src_id, source), tasks_by_src in groupby(sorted(task_group, key=srt_src), key=grp_src):
                with CobaConfig.Logger.log(f"Processing group..."):

                    with CobaConfig.Logger.time(f"Creating source {src_id} from {source}..."):
                        loaded_source = source.read()

                    for (sim_id,sim), tasks_by_src_sim in groupby(sorted(tasks_by_src, key=srt_sim), key=grp_sim):

                        tasks_by_src_sim = list(tasks_by_src_sim)
                        learner_ids      = [t.lrn_id  for t in tasks_by_src_sim] 
                        learners         = [t.learner for t in tasks_by_src_sim] 

                        with CobaConfig.Logger.time(f"Creating simulation {sim_id} from source {src_id}..."):
                            simulation = sim.filter.filter(loaded_source)

                            interactions = simulation.interactions
                            batches      = batchify(simulation)

                            yield sim_transaction(sim_id, sim, interactions, batches)

                        if not batches:
                            CobaConfig.Logger.log(f"Simulation {sim_id} has nothing to evaluate. (likely due to `Take` being larger than the source)")
                            continue

                        for lrn_id,learner in zip(learner_ids,learners):

                            try:

                                learner.init()

                                with CobaConfig.Logger.time(f"Evaluating learner {lrn_id} on Simulation {sim_id}..."):
                                    batch_sizes  = [ len(batch)                                             for batch in batches ]
                                    mean_rewards = [ self._process_batch(batch, learner, simulation.reward) for batch in batches ]

                                    yield Transaction.batch(sim_id, lrn_id, N=batch_sizes, reward=mean_rewards)

                            except Exception as e:
                                CobaConfig.Logger.log_exception("Unhandled exception:", e)
                                if not self._ignore_raise: raise e
        except KeyboardInterrupt:
            raise
        except Exception as e:
            CobaConfig.Logger.log_exception("unhandled exception:", e)
            if not self._ignore_raise: raise e

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

        rewards = reward.observe(list(zip(keys, actions))) 

        for (key,context,action,reward,prob) in zip(keys,contexts,actions,rewards,probs):
            learner.learn(key,context,action,reward,prob)

        return round(mean(rewards),5)

    def _context_sizes(self, interactions) -> Iterable[int]:
        if len(interactions) == 0:
            yield 0

        for context in [i.context for i in interactions]:
            yield 0 if context is None else len(context) if isinstance(context,tuple) else 1

    def _action_counts(self, interactions) -> Iterable[int]:
        if len(interactions) == 0:
            yield 0

        for actions in [i.actions for i in interactions]:
            yield len(actions)

class Benchmark:
    """An on-policy Benchmark using samples drawn from simulations to estimate performance statistics."""
    
    @overload
    @staticmethod
    def from_file(filesource:Union[Source[str], Source[Iterable[str]]]) -> 'Benchmark': ...

    @overload
    @staticmethod
    def from_file(filename:str) -> 'Benchmark': ...
    
    @staticmethod #type: ignore #(this apppears to be a mypy bug https://github.com/python/mypy/issues/7781)
    def from_file(arg) -> 'Benchmark': #type: ignore
        """Instantiate a Benchmark from a config file."""

        if isinstance(arg,str) and arg.startswith('http'):
            content = ResponseToText().filter(HttpSource(arg).read())
        
        elif isinstance(arg,str) and not arg.startswith('http'):
            content = '\n'.join(DiskSource(arg).read())

        else:
            content = arg.read() #type: ignore

        return CobaRegistry.construct(CobaConfig.Benchmark['file_fmt']).filter(JsonDecode().filter(content))

    @overload
    def __init__(self, 
        simulations : Sequence[Source[Simulation]],
        *,
        batch_size      : int = 1,
        take            : int = None,
        shuffle         : Sequence[Optional[int]] = [None],
        ignore_raise    : bool = True,
        processes       : int = None,
        maxtasksperchild: int = None) -> None: ...

    @overload
    def __init__(self,
        simulations : Sequence[Source[Simulation]],
        *,
        batch_count     : int,
        take            : int = None,
        shuffle         : Sequence[Optional[int]] = [None],
        ignore_raise    : bool = True,
        processes       : int = None,
        maxtasksperchild: int = None) -> None: ...

    @overload
    def __init__(self, 
        simulations : Sequence[Source[Simulation]],
        *,
        batch_sizes     : Sequence[int],
        shuffle         : Sequence[Optional[int]] = [None],
        ignore_raise    : bool = True,
        processes       : int = None,
        maxtasksperchild: int = None) -> None: ...

    def __init__(self,*args, **kwargs) -> None:
        """Instantiate a UniversalBenchmark.

        Args:
            simulations: The sequence of simulations to benchmark against.
            batcher: How each simulation is broken into evaluation batches.
            ignore_raise: Should exceptions be raised or logged during evaluation.
            shuffle: A sequence of seeds for simulation shuffling. None means no shuffle.
            processes: The number of process to spawn during evalution (overrides coba config).
            maxtasksperchild: The number of tasks each process will perform before a refresh.
        
        See the overloads for more information.
        """

        sources = cast(Sequence[Source[Simulation]], args[0])
        filters: List[Sequence[Filter[Simulation,Simulation]]] = []

        if 'shuffle' in kwargs and kwargs['shuffle'] != [None]:
            filters.append([ Shuffle(seed) for seed in kwargs['shuffle'] ])

        if 'take' in kwargs:
            filters.append([ Take(kwargs['take']) ])

        if 'batch_count' in kwargs:
            filters.append([ Batch(count=kwargs['batch_count']) ])
        elif 'batch_size' in kwargs:
            filters.append([ Batch(size=kwargs['batch_size']) ])
        elif 'batch_sizes' in kwargs:
            filters.append([ Batch(sizes=kwargs['batch_sizes']) ])

        if len(filters) > 0:
            simulations = [cast(Source[Simulation],Pipe.join(s,f)) for s,f in product(sources, product(*filters))]
        else:
            simulations = list(sources)

        self._simulations      = simulations
        self._ignore_raise     = cast(bool         , kwargs.get('ignore_raise'    , True))
        self._processes        = cast(Optional[int], kwargs.get('processes'       , None))
        self._maxtasksperchild = cast(Optional[int], kwargs.get('maxtasksperchild', None))

    def ignore_raise(self, value:bool=True) -> 'Benchmark':
        self._ignore_raise = value
        return self

    def processes(self, value:int) -> 'Benchmark':
        self._processes = value
        return self

    def maxtasksperchild(self, value:int) -> 'Benchmark':
        self._maxtasksperchild = value
        return self

    def evaluate(self, learners: Sequence[Learner], transaction_log:str = None, seed:int = None) -> Result:
        """Collect observations of a Learner playing the benchmark's simulations to calculate Results.

        Args:
            factories: See the base class for more information.

        Returns:
            See the base class for more information.
        """
        restored         = Result.from_file(transaction_log)
        tasks            = Tasks(self._simulations, learners, seed)
        unfinished       = Unfinished(restored)
        grouped          = GroupByNone() if CobaConfig.Benchmark.get("group_by","source") == "none" else GroupBySource()
        process          = Transactions(self._ignore_raise)
        transaction_sink = TransactionSink(transaction_log, restored)

        n_given_learners    = len(learners)
        n_given_simulations = len(self._simulations)
 
        if len(restored.benchmark) != 0:
            assert n_given_learners    == restored.benchmark['n_learners'   ], "The currently evaluating benchmark doesn't match the given transaction log"
            assert n_given_simulations == restored.benchmark['n_simulations'], "The currently evaluating benchmark doesn't match the given transaction log"

        preamble = []
        preamble.append(Transaction.version())
        preamble.append(Transaction.benchmark(n_given_learners, n_given_simulations))
        preamble.extend(Transaction.learners(learners))

        mp = self._processes        if self._processes        else CobaConfig.Benchmark['processes']
        mt = self._maxtasksperchild if self._maxtasksperchild else CobaConfig.Benchmark['maxtasksperchild']
        
        grouped_tasks = Pipe.join(tasks, [unfinished,grouped])

        Pipe.join(MemorySource(preamble), []       , transaction_sink).run(1,None)
        Pipe.join(grouped_tasks         , [process], transaction_sink).run(mp,mt)

        return transaction_sink.result