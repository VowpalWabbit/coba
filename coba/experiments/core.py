from pathlib import Path
from typing import Sequence, Optional
from coba.backports import Literal

from coba.pipes import Pipe, Foreach
from coba.learners import Learner
from coba.environments import Environment
from coba.multiprocessing import CobaMultiprocessor
from coba.contexts import CobaContext, ExceptLog, StampLog, NameLog, DecoratedLogger

from coba.experiments.process import CreateWorkItems,  RemoveFinished, ChunkByTask, ChunkBySource, ProcessWorkItems
from coba.experiments.tasks   import EnvironmentTask, EvaluationTask, LearnerTask
from coba.experiments.tasks   import SimpleLearnerTask, SimpleEnvironmentTask, OnlineOnPolicyEvalTask
from coba.experiments.results import Result, TransactionIO

class Experiment:
    """An Experiment using a collection of environments and learners."""

    def __init__(self,
        environments    : Sequence[Environment],
        learners        : Sequence[Learner],
        learner_task    : LearnerTask     = SimpleLearnerTask(),
        environment_task: EnvironmentTask = SimpleEnvironmentTask(),
        evaluation_task : EvaluationTask  = OnlineOnPolicyEvalTask()) -> None:
        """Instantiate an Experiment.

        Args:
            environments: The collection of environments to use in the experiment.
            learners: The collection of learners to use in the experiment.
            learner_task: A task which describes a learner.
            environment_task: A task which describes an environment.
            evaluation_task: A task which evaluates a learner on an environment.
        """

        self._environments     = environments
        self._learners         = learners
        self._learner_task     = learner_task
        self._environment_task = environment_task
        self._evaluation_task  = evaluation_task

        self._processes        : Optional[int] = None
        self._maxchunksperchild: Optional[int] = None
        self._chunk_by         : Optional[str] = None

    def config(self, 
        processes: int = None,
        chunk_by: Literal['source','task'] = None,
        maxchunksperchild: Optional[int] = None) -> 'Experiment':
        """Configure how the experiment will be executed. 
        
        A value of `None` for any item means the CobaContext.experiment will be used.

        Args:
            chunk_by: The method for chunking tasks before processing.
            processes: The number of processes to create for evaluating the experiment.
            maxchunkssperchild: The number of chunks each process evaluate before being restarted. A value of 0 means infinite.
        """

        assert chunk_by is None or chunk_by in ['task', 'source'], "The given chunk_by value wasn't recognized. Allowed values are 'task', 'source' and 'none'"
        assert processes is None or processes > 0, "The given number of processes is invalid. Must be greater than 0."
        assert maxchunksperchild is None or maxchunksperchild >= 0, "The given number of taks per child is invalid. Must be greater than or equal to 0 (0 for infinite)."

        self._chunk_by          = chunk_by
        self._processes         = processes
        self._maxchunksperchild = maxchunksperchild

        return self

    @property
    def chunk_by(self) -> str:
        """The method for chunking tasks before sending them to processes for execution. 

        This option is only relevant if the experiment is being executed on multiple processes.
        """
        return self._chunk_by if self._chunk_by is not None else CobaContext.experiment.chunk_by

    @property
    def processes(self) -> int:
        """The number of processes to use when evaluating the experiment."""
        return self._processes if self._processes is not None else CobaContext.experiment.processes

    @property
    def maxchunksperchild(self) -> int:
        """The number of chunks to perform per process before restarting an evaluation process."""
        return self._maxchunksperchild if self._maxchunksperchild is not None else CobaContext.experiment.maxchunksperchild

    def evaluate(self, result_file:str = None) -> Result:
        """Evaluate the experiment and return the results.

        Args:
            result_file: The file for writing and restoring results .
        """
        cb, mp, mt = self.chunk_by, self.processes, self.maxchunksperchild

        if mp > 1 or mt != 0:
            CobaContext.logger = DecoratedLogger([ExceptLog()], CobaContext.logger, [NameLog(), StampLog()])
        else:
            CobaContext.logger = DecoratedLogger([ExceptLog()], CobaContext.logger, [StampLog()])

        restored = Result.from_file(result_file) if result_file and Path(result_file).exists() else Result()

        n_given_learners     = len(self._learners)
        n_given_environments = len(self._environments)
 
        if len(restored.experiment) != 0:
            assert n_given_learners     == restored.experiment['n_learners'    ], "The current experiment doesn't match the given transaction log."
            assert n_given_environments == restored.experiment['n_environments'], "The current experiment doesn't match the given transaction log."

        workitems  = CreateWorkItems(self._environments, self._learners, self._learner_task, self._environment_task, self._evaluation_task)
        unfinished = RemoveFinished(restored)
        chunk      = ChunkByTask() if cb == 'task' else ChunkBySource()
        sink       = TransactionIO(result_file)

        single_process = ProcessWorkItems()
        multi_process  = Pipe.join([chunk, CobaMultiprocessor(ProcessWorkItems(), mp, mt)])
        process        = multi_process if mp > 1 or mt != 0 else single_process

        try:
            if not restored: sink.write(["T0", n_given_learners, n_given_environments])
            Pipe.join(workitems, [unfinished, process], Foreach(sink)).run()

        except KeyboardInterrupt as e: # pragma: no cover
            CobaContext.logger.log("Experiment execution was manually aborted via Ctrl-C")
        
        except Exception as ex: # pragma: no cover
            CobaContext.logger.log(ex)

        if isinstance(CobaContext.logger, DecoratedLogger):
            CobaContext.logger = CobaContext.logger.undecorate()

        return sink.read()
