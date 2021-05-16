from itertools import product
from typing import Iterable, Sequence, cast, Optional, overload, List, Union

from coba.learners import Learner
from coba.simulations import Simulation, Take, Shuffle, Batch
from coba.tools import CobaRegistry, CobaConfig
from coba.data.filters import Filter, JsonDecode, ResponseToText
from coba.data.sources import HttpSource, Source, MemorySource, DiskSource
from coba.data.pipes import Pipe

from coba.benchmarks.tasks import Tasks, Unfinished, GroupByNone, GroupBySource, Transactions
from coba.benchmarks.transactions import Transaction, TransactionSink
from coba.benchmarks.results import Result

class Benchmark:
    """A Benchmark which uses simulations to estimate performance statistics of learners."""
    
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
        preamble.extend(Transaction.simulations(self._simulations))

        mp = self._processes        if self._processes        else CobaConfig.Benchmark['processes']
        mt = self._maxtasksperchild if self._maxtasksperchild else CobaConfig.Benchmark['maxtasksperchild']
        
        grouped_tasks = Pipe.join(tasks, [unfinished,grouped])

        Pipe.join(MemorySource(preamble), []       , transaction_sink).run(1,None)
        Pipe.join(grouped_tasks         , [process], transaction_sink).run(mp,mt)

        return transaction_sink.result