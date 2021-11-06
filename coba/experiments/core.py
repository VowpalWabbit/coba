from pathlib import Path
from itertools import product
from typing import Iterable, Sequence, Optional, overload, List, Union

from coba.learners import Learner, SafeLearner
from coba.environments import Simulation, Take, Shuffle, SimSourceFilters, SimulatedInteraction
from coba.registry import CobaRegistry
from coba.config import CobaConfig
from coba.exceptions import CobaFatal
from coba.pipes import Pipe, Filter, Source, JsonDecode, ResponseToLines, HttpIO, MemoryIO, DiskIO
from coba.multiprocessing import CobaMultiprocessFilter

from coba.experiments.tasks import ChunkByNone, CreateTasks, FilterFinished, ChunkByTask, ChunkBySource, ProcessTasks
from coba.experiments.transactions import Transaction, TransactionSink
from coba.experiments.results import Result

class Experiment:
    """A Benchmark which uses simulations to calculate performance statistics for learners."""
    
    @overload
    @staticmethod
    def from_file(filesource:Union[Source[str], Source[Iterable[str]]]) -> 'Experiment': ...

    @overload
    @staticmethod
    def from_file(filename:str) -> 'Experiment': ...
    
    @staticmethod #type: ignore #(this apppears to be a mypy bug https://github.com/python/mypy/issues/7781)
    def from_file(arg) -> 'Experiment': #type: ignore
        """Instantiate a Benchmark from a config file."""

        if isinstance(arg,str) and arg.startswith('http'):
            content = '\n'.join(ResponseToLines().filter(HttpIO(arg).read()))
        
        elif isinstance(arg,str) and not arg.startswith('http'):
            content = '\n'.join(DiskIO(arg).read())

        else:
            content = arg.read() #type: ignore

        return CobaRegistry.construct(CobaConfig.Benchmark['file_fmt']).filter(JsonDecode().filter(content))

    def __init__(self, 
        simulations: Sequence[Simulation],
        shuffle    : Sequence[Optional[int]] = [None],
        take       : int = None,
        isWarmStart: bool = False) -> None:
        """Instantiate a Benchmark.

        Args:
            simulations: The collection of simulations to benchmark against.
            shuffle: A collection of seeds to use for simulation shuffling. A seed of `None` means no shuffle will be applied.
            take: The number of interactions to take from each simulation for evaluation.
            isWarmStart: Indicates if the benchmark will engage Warm Start.
        """

        sources: List[Simulation] = simulations
        filters: List[Sequence[Filter[Iterable[SimulatedInteraction],Iterable[SimulatedInteraction]]]] = []

        if shuffle != [None]:
            filters.append([ Shuffle(seed) for seed in shuffle ])

        if take is not None:
            filters.append([ Take(take) ])

        simulation_sources = [SimSourceFilters(s,f) for s,f in product(sources, product(*filters))]

        self._simulations         : Sequence[Simulation] = simulation_sources
        self._processes           : Optional[int]        = None
        self._maxtasksperchild    : Optional[int]        = None
        self._maxtasksperchild_set: bool                 = False
        self._chunk_by            : Optional[str]        = None
        self._isWarmStart         : bool                 = False

    def chunk_by(self, value: str = 'source') -> 'Experiment':
        """Determines how tasks are chunked for processing.
        
        Args:
            value: Allowable values are 'task', 'source' and 'none'.
        """

        assert value in ['task', 'source', 'none'], "The given chunk_by value wasn't recognized. Allowed values are 'task', 'source' and 'none'"

        self._chunk_by = value

        return self

    def processes(self, value:int = 1) -> 'Experiment':
        """Determines how many processes will be utilized for processing Benchmark chunks.
        
        Args:
            value: This is the number of processes Benchmark will use.
        """

        self._processes = value
        return self

    def maxtasksperchild(self, value: Optional[int] = None) -> 'Experiment':
        """Determines how many chunks a process can handle before it will be torn down and recreated.
        
        Args:
            value: This is the number of chunks a process will handle before being recreated. If this
                value is None then processes will remain alive for the life of the Benchmark evaluation.
        """

        self._maxtasksperchild_set = True
        self._maxtasksperchild = value
        return self

    def evaluate(self, learners: Sequence[Learner], result_file:str = None, seed:int = 1) -> Result:
        """Collect observations of a Learner playing the benchmark's simulations to calculate Results.

        Args:
            learners: The collection of learners that we'd like to evalute.
            result_file: The file we'd like to use for writing/restoring results for the requested evaluation.
            seed: The random seed we'd like to use when choosing which action to take from the learner's predictions.

        Returns:
            See the base class for more information.
        """

        restored = Result.from_file(result_file) if result_file and Path(result_file).exists() else Result()
        safe_learners = list(map(SafeLearner, learners))

        n_given_learners    = len(safe_learners)
        n_given_simulations = len(self._simulations)
 
        if len(restored.benchmark) != 0:
            assert n_given_learners    == restored.benchmark['n_learners'   ], "The currently evaluating benchmark doesn't match the given transaction log"
            assert n_given_simulations == restored.benchmark['n_simulations'], "The currently evaluating benchmark doesn't match the given transaction log"

        preamble = []
        preamble.append(Transaction.version())
        preamble.append(Transaction.benchmark(n_given_learners, n_given_simulations))
        preamble.extend(Transaction.learners(safe_learners))

        cb = self._chunk_by         if self._chunk_by             else CobaConfig.Benchmark['chunk_by']
        mp = self._processes        if self._processes            else CobaConfig.Benchmark['processes']
        mt = self._maxtasksperchild if self._maxtasksperchild_set else CobaConfig.Benchmark['maxtasksperchild']
        
        tasks            = CreateTasks(self._simulations, safe_learners, seed, self._isWarmStart)
        unfinished       = FilterFinished(restored)
        chunked          = ChunkByTask() if cb == 'task' else ChunkByNone() if cb == 'none' else ChunkBySource()
        process          = ProcessTasks()
        transaction_sink = TransactionSink(result_file, restored)

        if mp > 1 or mt is not None  : process = CobaMultiprocessFilter([process], mp, mt) #type: ignore

        try:
            Pipe.join(MemoryIO(preamble), []                            , transaction_sink).run()
            Pipe.join(tasks             , [unfinished, chunked, process], transaction_sink).run()
        except KeyboardInterrupt:
            CobaConfig.Logger.log("Benchmark evaluation was manually aborted via Ctrl-C")
        except CobaFatal:
            raise
        except Exception as ex:
            CobaConfig.Logger.log_exception(ex)

        return transaction_sink.result