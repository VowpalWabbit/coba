from pathlib import Path
from typing_extensions import Literal
from typing import Sequence, Optional

from coba.learners import Learner
from coba.environments import Simulation
from coba.config import CobaConfig
from coba.exceptions import CobaFatal
from coba.pipes import Pipe, MemoryIO
from coba.multiprocessing import CobaMultiprocessFilter

from coba.experiments.tasks import CreateWorkItems, EnvironmentTask, EvaluationTask, RemoveFinished, ChunkByTask, ChunkBySource, LearnerTask, ProcessWorkItems
from coba.experiments.tasks import SimpleLearnerTask, SimpleEnvironmentTask, OnPolicyEvaluationTask

from coba.experiments.transactions import Transaction, TransactionSink
from coba.experiments.results import Result

class Experiment:
    """A Benchmark which uses simulations to calculate performance statistics for learners."""

    def __init__(self,
        simulations     : Sequence[Simulation],
        learners        : Sequence[Learner],
        learner_task    : LearnerTask     = SimpleLearnerTask(), 
        environment_task: EnvironmentTask = SimpleEnvironmentTask(),
        evaluation_task : EvaluationTask  = OnPolicyEvaluationTask()) -> None:
        """Instantiate a Benchmark.

        Args:
            simulations: The collection of simulations to benchmark against.
        """

        self._simulations      = simulations
        self._learners         = learners
        self._learner_task     = learner_task
        self._environment_task = environment_task
        self._evaluation_task  = evaluation_task

        self._processes       : Optional[int] = None
        self._maxtasksperchild: Optional[int] = None
        self._chunk_by        : Optional[str] = None

    def config(self, 
        chunk_by: Literal['source','task'] = None,
        processes: int = None,
        maxtasksperchild: Optional[int] = None) -> 'Experiment':
        """Configure how the experiment will be executed. A value of `None` means the global config will be used.
        
        Args:
            chunk_by: Determines how tasks are chunked for processing.
            processes: Determines how many processes to use when processing task chunks.
            maxtasksperchild: Determines how many tasks each process will complete before being restarted. A value of -1 means infinite.
        """

        assert chunk_by is None or chunk_by in ['task', 'source'], "The given chunk_by value wasn't recognized. Allowed values are 'task', 'source' and 'none'"
        assert processes is None or processes > 0, "The given number of processes is invalid. Must be greater than 0."
        assert maxtasksperchild is None or maxtasksperchild >= 0, "The given number of taks per child is invalid. Must be greater than or equal to 0 (0 for infinite)."

        self._chunk_by         = chunk_by
        self._processes        = processes
        self._maxtasksperchild = maxtasksperchild

        return self

    def evaluate(self, result_file:str = None, seed:int = 1) -> Result:
        """Collect observations of a Learner playing the benchmark's simulations to calculate Results.

        Args:
            learners: The collection of learners that we'd like to evalute.
            result_file: The file we'd like to use for writing/restoring results for the requested evaluation.
            seed: The random seed we'd like to use when choosing which action to take from the learner's predictions.

        Returns:
            See the base class for more information.
        """

        restored = Result.from_file(result_file) if result_file and Path(result_file).exists() else Result()

        n_given_learners    = len(self._learners)
        n_given_simulations = len(self._simulations)
 
        if len(restored.benchmark) != 0:
            assert n_given_learners    == restored.benchmark['n_learners'   ], "The currently evaluating benchmark doesn't match the given transaction log"
            assert n_given_simulations == restored.benchmark['n_simulations'], "The currently evaluating benchmark doesn't match the given transaction log"

        preamble = []
        preamble.append(Transaction.version())
        preamble.append(Transaction.benchmark(n_given_learners, n_given_simulations))

        cb = self._chunk_by         if self._chunk_by         else CobaConfig.experiment.chunk_by
        mp = self._processes        if self._processes        else CobaConfig.experiment.processes
        mt = self._maxtasksperchild if self._maxtasksperchild else CobaConfig.experiment.maxtasksperchild

        workitems  = CreateWorkItems(self._simulations, self._learners, self._learner_task, self._environment_task, self._evaluation_task, seed)
        unfinished = RemoveFinished(restored)
        chunk      = ChunkByTask() if cb == 'task' else ChunkBySource()
        sink       = TransactionSink(result_file, restored)

        single_process = ProcessWorkItems()
        multi_process  = Pipe.join([chunk, CobaMultiprocessFilter([ProcessWorkItems()], mp, mt)])
        process        = multi_process if mp > 1 or mt != -1 else single_process

        try:
            Pipe.join(MemoryIO(preamble), [], sink).run()
            Pipe.join(workitems, [unfinished, process], sink).run()
        
        except KeyboardInterrupt:
            CobaConfig.logger.log("Benchmark evaluation was manually aborted via Ctrl-C")
        except CobaFatal:
            raise
        except Exception as ex:
            CobaConfig.logger.log_exception(ex)

        return sink.result