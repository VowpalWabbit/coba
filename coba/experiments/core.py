from pathlib import Path
from typing import Sequence, Optional
from coba.backports import Literal

from coba.pipes import Pipes, Foreach
from coba.learners import Learner
from coba.environments import Environment
from coba.multiprocessing import CobaMultiprocessor
from coba.contexts import CobaContext, ExceptLog, StampLog, NameLog, DecoratedLogger

from coba.experiments.process import CreateWorkItems,  RemoveFinished, ChunkByTask, ChunkBySource, ProcessWorkItems, MaxChunkSize
from coba.experiments.tasks   import EnvironmentTask, EvaluationTask, LearnerTask
from coba.experiments.tasks   import SimpleLearnerTask, SimpleEnvironmentTask, OnlineOnPolicyEvalTask
from coba.experiments.results import Result, TransactionIO

class Experiment:
    """An Experiment using a collection of environments and learners."""

    def __init__(self,
        environments    : Sequence[Environment],
        learners        : Sequence[Learner],
        description     : str = None,
        learner_task    : LearnerTask     = SimpleLearnerTask(),
        environment_task: EnvironmentTask = SimpleEnvironmentTask(),
        evaluation_task : EvaluationTask  = OnlineOnPolicyEvalTask()) -> None:
        """Instantiate an Experiment.

        Args:
            environments: The collection of environments to use in the experiment.
            learners: The collection of learners to use in the experiment.
            description: A description of the experiment for documentaiton purposes.
            learner_task: A task which describes a learner.
            environment_task: A task which describes an environment.
            evaluation_task: A task which evaluates a learner on an environment.
        """

        self._environments     = environments
        self._learners         = learners
        self._description      = description
        self._learner_task     = learner_task
        self._environment_task = environment_task
        self._evaluation_task  = evaluation_task

        self._processes        : Optional[int] = None
        self._maxchunksperchild: Optional[int] = None
        self._maxtasksperchunk : Optional[int] = None 
        self._chunk_by         : Optional[str] = None

    def config(self,
        processes: int = None,
        chunk_by: Literal['source','task'] = None,
        maxchunksperchild: Optional[int] = None,
        maxtasksperchunk: Optional[int] = None) -> 'Experiment':
        """Configure how the experiment will be executed.

        A value of `None` for any item means the CobaContext.experiment will be used.

        Args:
            chunk_by: The method for chunking tasks before processing.
            processes: The number of processes to create for evaluating the experiment.
            maxchunksperchild: The number of chunks each process evaluate before being restarted. A 
                value of 0 means that all processes will survive until the end of the experiment.
            maxtasksperchunk: The maximum number of tasks a chunk can have. If a chunk has too many 
                tasks it will be split into smaller chunks. A value of 0 means that chunks are never
                broken down into smaller chunks.
        """

        assert chunk_by is None or chunk_by in ['task', 'source'], "The given chunk_by value wasn't recognized. Allowed values are 'task', 'source' and 'none'"
        assert processes is None or processes > 0, "The given number of processes is invalid. Must be greater than 0."
        assert maxchunksperchild is None or maxchunksperchild >= 0, "The given number of chunks per child is invalid. Must be greater than or equal to 0 (0 for infinite)."
        assert maxtasksperchunk is None or maxtasksperchunk >= 0, "The given number of tasks per chunk is invalid. Must be greater than or equal to 0 (0 for infinite)."

        self._chunk_by          = chunk_by
        self._processes         = processes
        self._maxchunksperchild = maxchunksperchild
        self._maxtasksperchunk  = maxtasksperchunk

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
        """The number of tasks chunks to perform per process before restarting an evaluation process."""
        return self._maxchunksperchild if self._maxchunksperchild is not None else CobaContext.experiment.maxchunksperchild

    @property
    def maxtasksperchunk(self) -> int:
        """The maximum number of tasks allowed in a chunk before breaking a chunk into smaller chunks."""
        return self._maxtasksperchunk if self._maxtasksperchunk is not None else CobaContext.experiment.maxtasksperchunk

    def run(self, result_file:str = None) -> Result:
        """Run the experiment and return the results.

        Args:
            result_file: The file for writing and restoring results .
        """
        cb, mp, mc, mt = self.chunk_by, self.processes, self.maxchunksperchild, self.maxtasksperchunk

        if mp > 1 or mc != 0:
            #Add name so that we know which process-id the logs came from in addition to the time of the log
            CobaContext.logger = DecoratedLogger([ExceptLog()], CobaContext.logger, [NameLog(), StampLog()])
        else:
            CobaContext.logger = DecoratedLogger([ExceptLog()], CobaContext.logger, [StampLog()])

        restored = Result.from_file(result_file) if result_file and Path(result_file).exists() else None

        n_given_learners     = len(self._learners)
        n_given_environments = len(self._environments)

        if restored:
            assert n_given_learners     == restored.experiment.get('n_learners',n_given_learners)        , "The current experiment doesn't match the given transaction log."
            assert n_given_environments == restored.experiment.get('n_environments',n_given_environments), "The current experiment doesn't match the given transaction log."

        workitems  = CreateWorkItems(self._environments, self._learners, self._learner_task, self._environment_task, self._evaluation_task)
        unfinished = RemoveFinished(restored)
        chunk      = ChunkByTask() if cb == 'task' else ChunkBySource()
        max_chunk  = MaxChunkSize(mt)
        sink       = TransactionIO(result_file)

        single_process = ProcessWorkItems()
        multi_process  = Pipes.join(chunk, max_chunk, CobaMultiprocessor(ProcessWorkItems(), mp, mc))
        process        = multi_process if mp > 1 or mc != 0 else single_process

        try:
            if not restored: sink.write(["T0", {'n_learners':n_given_learners, 'n_environments':n_given_environments, 'description':self._description }])
            Pipes.join(workitems, unfinished, process, Foreach(sink)).run()
        except KeyboardInterrupt as e: # pragma: no cover
            CobaContext.logger.log("Experiment execution was manually aborted via Ctrl-C")

        except Exception as ex: # pragma: no cover
            CobaContext.logger.log(ex)

        if isinstance(CobaContext.logger, DecoratedLogger):
            CobaContext.logger = CobaContext.logger.undecorate()

        return sink.read()

    def evaluate(self, result_file:str = None) -> Result:
        """Evaluate the experiment and return the results (this is a backwards compatible proxy for the run method).

        Args:
            result_file: The file for writing and restoring results .
        """

        return self.run(result_file=result_file)
