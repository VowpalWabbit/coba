from inspect import getmodule
from collections import abc
from pathlib import Path
from itertools import product
from typing import Sequence, Optional, Union, overload, Tuple

from coba.evaluators import SequentialCB

from coba.pipes import Pipes, DiskSink, ListSink, DiskSource, ListSource, Identity, Insert
from coba.results import Result, TransactionDecode, TransactionEncode, TransactionResult
from coba.context import CobaContext, ExceptLog, StampLog, NameLog, DecoratedLogger, ExceptionLogger
from coba.exceptions import CobaException
from coba.multiprocessing import CobaMultiprocessor
from coba.primitives import Learner, Environment, Evaluator
from coba.utilities import PackageChecker

from coba.experiments.process import MakeTasks, SelectTasks, ChunkTasks, ProcessTasks

class Experiment:
    """Experiment for environments, learners and evaluators."""

    @overload
    def __init__(self,
        environments : Union[Environment, Sequence[Environment]],
        learners     : Union[Learner,Sequence[Learner]],
        evaluator    : Union[Evaluator,Sequence[Evaluator]] = SequentialCB(),
        description  : str = None) -> None:
        """Instantiate an Experiment.

        Args:
            environments: The collection of environments to use in the experiment.
            learners: The collection of learners to use in the experiment.
            evaluator: The evaluation task we wish to perform on learners and environments.
            description: A description of the experiment for documentaiton purposes.
        """

    @overload
    def __init__(self,
        eval_tuples: Sequence[Union[Tuple[Learner,Environment],Tuple[Learner,Environment,Evaluator]]],
        description: str = None) -> None:
        ...
        """Instantiate an Experiment.

        Args:
            eval_tuples: The learner-environment-evaluator triples we wish to evaluate. If no evaluator
                is provided the default SequentialCB() will be used for evaluation.
            description: A description of the experiment for documentaiton purposes.
        """

    def __init__(self, *args,**kwargs) -> None:
        """Instantiate an Experiment.

        Args:
            environments: The collection of environments to use in the experiment.
            learners: The collection of learners to use in the experiment.
            evaluator: The evaluation task we wish to perform on learners and environments.
            eval_tuples: The learner-environment-evaluator triples we wish to evaluate.
            description: A description of the experiment for documentaiton purposes.
        """

        if 'evaluation_task' in kwargs:
            raise CobaException(
                "The `evaluation_task` argument has been deprecated. Instead of `Experiment(x,y,evaluation_task=t)` "
                "you should call `Experiment(x,y,t)`. Additionally if you are using a custom evaluator then the "
                "main evaluation method should be `def evaluate(self, learner, environment)` rather than "
                "`def process(self,learner,interactions)`."
            )

        args = list(args)
        if len(args) > 0 and not isinstance(args[0],(abc.Sequence,abc.Iterable)): args[0] = [args[0]]

        triples,description = self._parse_init_args(*args,**kwargs)

        self._triples     = triples
        self._description = description

        # if any([lrn is None for _,lrn,_ in self._triples]):
        #     raise CobaException("A Learner was given whose value was None, which can't be processed.")

        # if any([env is None for env,_,_ in self._triples]):
        #     raise CobaException("An Environment was given whose value was None, which can't be processed.")

        self._processes        : Optional[int] = None
        self._maxchunksperchild: Optional[int] = None
        self._maxtasksperchunk : Optional[int] = None

    def __len__(self) -> int:
        """The total number of environment-learner-evaluator tasks in the experiment."""
        return len(self._triples)

    def config(self,
        processes: int = None,
        maxchunksperchild: int = None,
        maxtasksperchunk: int = None) -> 'Experiment':
        """Configure how the experiment will be executed.

        Args:
            processes: The number of processes to create for evaluating the experiment.
            maxchunksperchild: The number of chunks each process evaluate before being restarted. A
                value of 0 means that all processes will survive until the end of the experiment.
            maxtasksperchunk: The maximum number of tasks a chunk can have. If a chunk has too many
                tasks it will be split into smaller chunks. A value of 0 means that chunks are never
                broken down into smaller chunks.

        Returns:
            The configured Experiment.
        """

        assert processes is None or processes > 0, "The given number of processes is invalid. Must be greater than 0."
        assert maxchunksperchild is None or maxchunksperchild >= 0, "The given number of chunks per child is invalid. Must be greater than or equal to 0 (0 for infinite)."
        assert maxtasksperchunk is None or maxtasksperchunk >= 0, "The given number of tasks per chunk is invalid. Must be greater than or equal to 0 (0 for infinite)."

        self._processes         = processes
        self._maxchunksperchild = maxchunksperchild
        self._maxtasksperchunk  = maxtasksperchunk

        return self

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

    def run(self,
            result_file:str = None,
            quiet:bool = False,
            processes:int = None,
            maxchunksperchild: int = None,
            maxtasksperchunk: int = None,
            seed: Optional[int] = 1,
            start: Optional[int] = None,
            end: Optional[int] = None,
            batch_index: Optional[int] = None,
            batch_count: Optional[int] = None) -> Result:
        """Run the experiment and return the results.

        Args:
            result_file: The file for writing and restoring results.
            quiet: Indicates that logged output should be turned off.
            processes: The number of processes to create for evaluating the experiment.
            maxchunksperchild: The number of chunks each process evaluates before being restarted. A
                value of 0 means that processes will survive until the end of the experiment. Unless
                otherwise specified chunks are individual environment-learner-evaluator triples.
            maxtasksperchunk: The maximum number of tasks a chunk can have. If a chunk has too many
                tasks it will be split into smaller chunks. A value of 0 means that chunks are never
                broken down into smaller chunks.
            seed: The seed that will determine all randomness within the experiment.
            start: The index (0-based, inclusive) of the first environment-learner-evaluator task to
                run. Tasks are indexed according to their position in the full experiment definition
                (i.e., the full cartesian product of environments, learners and evaluators), not the
                subset being run. Defaults to 0 if not given. Cannot be given with batch_index/batch_count.
            end: The index (0-based, exclusive) of the last environment-learner-evaluator task to run.
                Defaults to the total number of tasks in the experiment if not given. Cannot be given
                with batch_index/batch_count.
            batch_index: Which batch of tasks to run when the experiment's tasks are split into
                batch_count contiguous, (as close as possible to) equally sized batches. Valid values
                are 0 to batch_count-1. Cannot be given with start/end.
            batch_count: The total number of batches the experiment's tasks should be split into. Must
                be given together with batch_index. Cannot be given with start/end.

        Note:
            A start/end slice or batch only restricts which evaluations actually run; every
            environment/learner/evaluator in the full experiment still gets registered (and, for
            environments, peeked) by every slice/batch, exactly as a full run would register them. This
            keeps ids stable across separate slices/batches, at the cost of some redundant registration
            work for entities the given slice/batch doesn't actually evaluate.

            Give each slice/batch its own result_file (e.g. one per batch_index inside a shared
            folder) and combine them afterward with Result.from_files or Result.from_folder, rather
            than pointing multiple slices/batches at one shared result_file. DiskSink does not use
            any file locking, so concurrent writers appending to one shared file can silently lose
            data (not just duplicate/interleave it) -- this was confirmed empirically, not just
            theoretical -- and even running slices/batches one after another into a shared file only
            works if they are guaranteed to never overlap in time. Separate files sidestep this
            entirely: nothing ever writes concurrently to the same file, so no coordination between
            slices/batches is needed regardless of where or how they run (including truly parallel
            jobs on a cluster, which is the scenario this option exists for).

        Returns:
            Result of the experiment.
        """

        self.config(processes,maxchunksperchild,maxtasksperchunk)
        mp,mc,mt = self.processes,self.maxchunksperchild,self.maxtasksperchunk

        n_tasks = len(self._triples)

        if batch_index is not None or batch_count is not None:
            if start is not None or end is not None:
                raise CobaException("The batch_index/batch_count and start/end arguments cannot be used together.")
            if batch_index is None or batch_count is None:
                raise CobaException("Both batch_index and batch_count must be given together.")
            assert batch_count > 0, "batch_count must be greater than 0."
            assert 0 <= batch_index < batch_count, "batch_index must be between 0 and batch_count-1."

            base_size, n_large = divmod(n_tasks, batch_count)
            start = batch_index * base_size + min(batch_index, n_large)
            end   = start + base_size + (1 if batch_index < n_large else 0)
        else:
            start = 0       if start is None else start
            end   = n_tasks if end   is None else end

        assert 0 <= start <= end <= n_tasks, (
            f"The given start/end ({start}/{end}) is invalid for an experiment with {n_tasks} tasks."
        )
        is_partial = (start,end) != (0,n_tasks)
        select     = set(self._triples[start:end]) if is_partial else None

        CobaContext.store['experiment_seed'] = seed
        is_multiproc = mp > 1 or mc != 0

        if is_multiproc: self._check_for_cloudpickle_dependency()

        old_logger = CobaContext.logger

        if quiet:
            CobaContext.logger = DecoratedLogger([], ExceptionLogger(CobaContext.logger.sink), [NameLog(),StampLog()] if is_multiproc else [StampLog()])
        else:
            CobaContext.logger = DecoratedLogger([ExceptLog()], CobaContext.logger, [NameLog(),StampLog()] if is_multiproc else [StampLog()])

        CobaContext.logger.log("Experiment Started")

        if result_file and Path(result_file).exists():
            CobaContext.logger.log("Restoring Results")
            restored = Result.from_file(result_file)
        else:
            restored = None

        n_given_lrns = len(set([l for _,l,_ in self._triples]))
        n_given_envs = len(set([e for e,_,_ in self._triples]))

        meta = {'n_learners':n_given_lrns,'n_environments':n_given_envs,'description':self._description,'seed':seed}

        workitems = MakeTasks(self._triples,restored)
        selector  = SelectTasks(select)
        chunker   = ChunkTasks(mt)
        process   = CobaMultiprocessor(ProcessTasks(), mp, mc, False)
        encode    = TransactionEncode(restored)
        sink      = DiskSink(result_file,batch=1) if result_file else ListSink(foreach=True)
        source    = DiskSource(result_file) if result_file else ListSource(sink.items)
        decode    = TransactionDecode()
        result    = TransactionResult()
        preamble  = Identity() if restored else Insert([["T0",meta]])

        try:
            lrn_mismatch = restored and n_given_lrns != restored.experiment.get('n_learners',n_given_lrns)
            env_mismatch = restored and n_given_envs != restored.experiment.get('n_environments',n_given_envs)
            if lrn_mismatch or env_mismatch: raise CobaException("The experiment does not match the given logs")
            Pipes.join(workitems, selector, chunker, process, preamble, encode, sink).run()
        except KeyboardInterrupt: #pragma: no cover
            CobaContext.logger.log("Experiment Aborted (aborted via Ctrl-C)")
        except Exception as ex: #pragma: no cover
            CobaContext.logger.log(ex)
            CobaContext.logger.log("Experiment Failed")
        else:
            CobaContext.logger.log("Experiment Finished")

        CobaContext.logger = old_logger
        del CobaContext.store['experiment_seed']

        return Pipes.join(source,decode,result).read()

    def _parse_init_args(self,*args,**kwargs) -> Tuple[Sequence[Tuple[Environment,Learner]], Evaluator, Optional[str]]:
        #we know this with 100% certainty
        definite_triples = ({'eval_tuples'} & kwargs.keys()            ) or \
                           (len(args) == 1 and 'learners' not in kwargs) or \
                           (len(args) == 2 and isinstance(args[1],str) )
        try:
            if definite_triples:
                triples = args[0] if len(args) > 0 else kwargs['eval_tuples']
                descr   = args[1] if len(args) > 1 else kwargs.get('description',None)
                triples = [t if len(t) ==3 else (*t,SequentialCB()) for t in triples]
                return triples, descr

            else:
                envs  = args[0] if len(args) > 0 else kwargs['environments']
                lrns  = args[1] if len(args) > 1 else kwargs['learners']
                vals  = args[2] if len(args) > 2 else kwargs.get('evaluator',SequentialCB())
                descr = args[3] if len(args) > 3 else kwargs.get('description',None)
                if not isinstance(envs,abc.Sequence): envs = [envs]
                if not isinstance(lrns,abc.Sequence): lrns = [lrns]
                if not isinstance(vals,abc.Sequence): vals = [vals]
                return list(product(envs,lrns,vals)), descr

        except KeyError as e:
            raise TypeError(f'Experiment.__init__ missing required arguments')

    def _check_for_cloudpickle_dependency(self) -> None:
        objs = set()

        for e,l,v in self._triples:
            objs.add(l)
            objs.add(v)

            try:
                objs.update(e)
            except:
                objs.add(e)

        if any(not hasattr(getmodule(o),'__file__') for o in objs):
            PackageChecker.cloudpickle('Coba multiprocessing in Jupyter')
