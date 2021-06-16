from pathlib import Path
from itertools import product
from typing import Iterable, Sequence, cast, Optional, overload, List, Union

from coba.learners import Learner
from coba.simulations import Simulation, Take, Shuffle, Batch
from coba.registry import CobaRegistry
from coba.config import CobaConfig
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

    @overload
    def __init__(self, 
        simulations: Sequence[Source[Simulation]],
        *,
        batch_size : int = 1,
        shuffle    : Sequence[Optional[int]] = [None],
        take       : int = None) -> None:
        """Instantiate a Benchmark.

        Args:
            simulations: The collection of simulations to benchmark against.
            batch_size: The number of interactions to predict before receiving reward feedback to learn from.
            shuffle: A collection of seeds to use for simulation shuffling. A seed of `None` means no shuffle will be applied.
            take: The number of interactions to take from each simulation for evaluation.
        """
        ...

    @overload
    def __init__(self,
        simulations: Sequence[Source[Simulation]],
        *,
        batch_count: int,
        shuffle    : Sequence[Optional[int]] = [None],
        take       : int = None) -> None:
        """Instantiate a Benchmark.

        Args:
            simulations: The collection of simulations to benchmark against.
            batch_count: The number of times feedback will be given to each learner during a simulation.
            shuffle: A collection of seeds to use for simulation shuffling. A seed of `None` means no shuffle will be applied.
            take: The number of interactions to take from each simulation for evaluation.
        """
        ...

    @overload
    def __init__(self, 
        simulations : Sequence[Source[Simulation]],
        *,
        batch_sizes     : Sequence[int],
        shuffle         : Sequence[Optional[int]] = [None]) -> None:
        """Instantiate a Benchmark.

        Args:
            simulations: The collection of simulations to benchmark against.
            batch_sizes: The number of interactions to predict on each learning iteration before providing feedback to learners.
            shuffle: A collection of seeds to use for simulation shuffling. A seed of `None` means no shuffle will be applied.
        """
        ...

    def __init__(self,*args, **kwargs) -> None:

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

        self._simulations: Sequence[Source[Simulation]] = simulations
        self._ignore_raise: bool                        = True
        self._processes: Optional[int]                  = None
        self._maxtasksperchild: Optional[int]           = None
        self._maxtasksperchild_set: bool                = False
        self._chunk_by: Optional[str]                   = None

    def chunk_by(self, value: str = 'source') -> 'Benchmark':
        """Determines how tasks are chunked for processing.
        
        Args:
            value: Allowable values are 'task', 'source' and 'none'.
        """

        assert value in ['task', 'source', 'none'], "The given chunk_by value wasn't recognized. Allowed values are 'task', 'source' and 'none'"

        self._chunk_by = value

        return self

    def ignore_raise(self, value:bool=True) -> 'Benchmark':
        """Determines how unexpected Exceptions are handled by Benchmark.
        
        Args:
            value: If the value is `True` then Benchmark will exit on an exception. Otherwise
                Benchmark will log the exception and continue running the rest of the tasks.
        """

        self._ignore_raise = value
        return self

    def processes(self, value:int = 1) -> 'Benchmark':
        """Determines how many processes will be utilized for processing Benchmark chunks.
        
        Args:
            value: This is the number of processes Benchmark will use.
        """

        self._processes = value
        return self

    def maxtasksperchild(self, value: Optional[int] = 1) -> 'Benchmark':
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
        process          = Transactions(self._ignore_raise)
        transaction_sink = TransactionSink(result_file, restored)

        if mp > 1 or mt is not None  : process = MultiprocessFilter([process], mp, mt) #type: ignore

        try:
            Pipe.join(MemorySource(preamble), []                            , transaction_sink).run()
            Pipe.join(tasks                 , [unfinished, chunked, process], transaction_sink).run()
        except KeyboardInterrupt:
            CobaConfig.Logger.log("Benchmark evaluation was canceled via a keyboard interrupt command.")

        return transaction_sink.result