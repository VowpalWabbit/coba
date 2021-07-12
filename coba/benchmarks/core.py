from coba.simulations.core import Interaction
from pathlib import Path
from itertools import product
from typing import Iterable, Sequence, cast, Optional, overload, List, Union

from coba.learners import Learner
from coba.simulations import Simulation, Take, Shuffle
from coba.registry import CobaRegistry
from coba.config import CobaConfig, CobaFatal
from coba.pipes import Pipe, Filter, Source, JsonDecode, ResponseToLines, HttpSource, MemorySource, DiskSource
from coba.multiprocessing import MultiprocessFilter

from coba.benchmarks.tasks import ChunkByNone, Tasks, Unfinished, ChunkByTask, ChunkBySource, Transactions
from coba.benchmarks.transactions import Transaction, TransactionSink
from coba.benchmarks.results import Result

class Benchmark:
    """A Benchmark which uses simulations to calculate performance statistics for learners."""
    
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
            content = '\n'.join(ResponseToLines().filter(HttpSource(arg).read()))
        
        elif isinstance(arg,str) and not arg.startswith('http'):
            content = '\n'.join(DiskSource(arg).read())

        else:
            content = arg.read() #type: ignore

        return CobaRegistry.construct(CobaConfig.Benchmark['file_fmt']).filter(JsonDecode().filter(content))

    def __init__(self, 
        simulations: Sequence[Simulation],
        shuffle    : Sequence[Optional[int]] = [None],
        take       : int = None) -> None:
        """Instantiate a Benchmark.

        Args:
            simulations: The collection of simulations to benchmark against.
            shuffle: A collection of seeds to use for simulation shuffling. A seed of `None` means no shuffle will be applied.
            take: The number of interactions to take from each simulation for evaluation.
        """
        ...

        sources: List[Simulation] = simulations
        filters: List[Sequence[Filter[Iterable[Interaction],Iterable[Interaction]]]] = []

        if shuffle != [None]:
            filters.append([ Shuffle(seed) for seed in shuffle ])

        if take is not None:
            filters.append([ Take(take) ])

        if len(filters) > 0:
            simulation_sources = [cast(Source[Simulation],Pipe.join(s,f)) for s,f in product(sources, product(*filters))]
        else:
            simulation_sources = list(sources)

        self._simulations         : Sequence[Source[Simulation]] = simulation_sources
        self._processes           : Optional[int]                = None
        self._maxtasksperchild    : Optional[int]                = None
        self._maxtasksperchild_set: bool                         = False
        self._chunk_by            : Optional[str]                = None

    def chunk_by(self, value: str = 'source') -> 'Benchmark':
        """Determines how tasks are chunked for processing.
        
        Args:
            value: Allowable values are 'task', 'source' and 'none'.
        """

        assert value in ['task', 'source', 'none'], "The given chunk_by value wasn't recognized. Allowed values are 'task', 'source' and 'none'"

        self._chunk_by = value

        return self

    def processes(self, value:int = 1) -> 'Benchmark':
        """Determines how many processes will be utilized for processing Benchmark chunks.
        
        Args:
            value: This is the number of processes Benchmark will use.
        """

        self._processes = value
        return self

    def maxtasksperchild(self, value: Optional[int] = None) -> 'Benchmark':
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

        n_given_learners    = len(learners)
        n_given_simulations = len(self._simulations)
 
        if len(restored.benchmark) != 0:
            assert n_given_learners    == restored.benchmark['n_learners'   ], "The currently evaluating benchmark doesn't match the given transaction log"
            assert n_given_simulations == restored.benchmark['n_simulations'], "The currently evaluating benchmark doesn't match the given transaction log"

        preamble = []
        preamble.append(Transaction.version())
        preamble.append(Transaction.benchmark(n_given_learners, n_given_simulations))
        preamble.extend(Transaction.learners(learners))
        preamble.extend(Transaction.simulations(self._simulations))

        cb = self._chunk_by         if self._chunk_by             else CobaConfig.Benchmark['chunk_by']
        mp = self._processes        if self._processes            else CobaConfig.Benchmark['processes']
        mt = self._maxtasksperchild if self._maxtasksperchild_set else CobaConfig.Benchmark['maxtasksperchild']
            
        tasks            = Tasks(self._simulations, learners, seed)
        unfinished       = Unfinished(restored)
        chunked          = ChunkByTask() if cb == 'task' else ChunkByNone() if cb == 'none' else ChunkBySource()
        process          = Transactions()
        transaction_sink = TransactionSink(result_file, restored)

        if mp > 1 or mt is not None  : process = MultiprocessFilter([process], mp, mt) #type: ignore

        try:
            Pipe.join(MemorySource(preamble), []                            , transaction_sink).run()
            Pipe.join(tasks                 , [unfinished, chunked, process], transaction_sink).run()
        except KeyboardInterrupt:
            CobaConfig.Logger.log("Benchmark evaluation was manually aborted via Ctrl-C")
        except CobaFatal:
            raise
        except Exception as ex:
            CobaConfig.Logger.log_exception(ex)

        return transaction_sink.result