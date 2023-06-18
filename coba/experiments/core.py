
from collections import abc
from pathlib import Path
from itertools import product
from typing import Sequence, Optional, Union, overload, Tuple

from coba.environments import Environment
from coba.learners     import Learner
from coba.evaluators   import Evaluator, OnPolicyEvaluator, LambdaEvaluator

from coba.pipes import Pipes, DiskSink, ListSink, DiskSource, ListSource, Identity, Insert
from coba.contexts import CobaContext, ExceptLog, StampLog, NameLog, DecoratedLogger, ExceptionLogger
from coba.exceptions import CobaException
from coba.multiprocessing import CobaMultiprocessor

from coba.experiments.process import MakeTasks,  ResumeTasks, ChunkTasks, MaxChunk, ProcessTasks
from coba.experiments.results import Result, TransactionDecode, TransactionEncode, TransactionResult

class Experiment:
    """An Experiment using a collection of environments and learners."""

    @overload
    def __init__(self,
        environments : Union[Environment, Sequence[Environment]],
        learners     : Union[Learner,Sequence[Learner]],
        evaluator    : Evaluator = OnPolicyEvaluator(),
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
        eval_tuples: Sequence[Tuple[Learner,Environment]],
        evaluator  : Evaluator = OnPolicyEvaluator(),
        description: str = None) -> None:
        ...
        """Instantiate an Experiment.

        Args:
            eval_pairs: The learner-environment pairs we wish to evaluate.
            evaluator: The evaluation task we wish to perform on learners and environments.
            description: A description of the experiment for documentaiton purposes.
        """

    def __init__(self, *args,**kwargs) -> None:
        """Instantiate an Experiment."""

        args = list(args)
        if len(args) > 0 and not isinstance(args[0],abc.Sequence): args[0] = [args[0]]

        pairs,evaluator,description = self._parse_init_args(*args,**kwargs)

        if callable(evaluator): evaluator = LambdaEvaluator(evaluator)

        self._triples     = [(e,l,evaluator) for e,l in pairs]
        self._description = description

        if any([lrn is None for _,lrn,_ in self._triples]):
            raise CobaException("A Learner was given whose value was None, which can't be processed.")

        if any([env is None for env,_,_ in self._triples]):
            raise CobaException("An Environment was given whose value was None, which can't be processed.")

        self._processes        : Optional[int] = None
        self._maxchunksperchild: Optional[int] = None
        self._maxtasksperchunk : Optional[int] = None

    def config(self,
        processes: int = None,
        maxchunksperchild: Optional[int] = None,
        maxtasksperchunk: Optional[int] = None) -> 'Experiment':
        """Configure how the experiment will be executed.

        A value of `None` for any item means the CobaContext.experiment will be used.

        Args:
            processes: The number of processes to create for evaluating the experiment.
            maxchunksperchild: The number of chunks each process evaluate before being restarted. A 
                value of 0 means that all processes will survive until the end of the experiment.
            maxtasksperchunk: The maximum number of tasks a chunk can have. If a chunk has too many 
                tasks it will be split into smaller chunks. A value of 0 means that chunks are never
                broken down into smaller chunks.
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

    def run(self, result_file:str = None, quiet:bool = False, processes:int = None, seed: Optional[int] = 1) -> Result:
        """Run the experiment and return the results.

        Args:
            result_file: The file for writing and restoring results.
            quiet: Indicates that logged output should be turned off.
            processes: The number of processes to create for evaluating the experiment.
            seed: The seed that will determine all randomness within the experiment.
        """
        mp, mc, mt = (processes or self.processes), self.maxchunksperchild, self.maxtasksperchunk

        CobaContext.store['experiment_seed'] = seed
        is_multiproc = mp > 1 or mc != 0

        old_logger = CobaContext.logger

        if quiet:
            CobaContext.logger = DecoratedLogger([], ExceptionLogger(CobaContext.logger.sink), [NameLog(),StampLog()] if is_multiproc else [StampLog()])
        else:
            CobaContext.logger = DecoratedLogger([ExceptLog()], CobaContext.logger, [NameLog(),StampLog()] if is_multiproc else [StampLog()])

        if result_file and Path(result_file).exists():
            CobaContext.logger.log("Restoring existing experiment logs...")
            restored = Result.from_file(result_file)
        else:
            restored = None

        n_given_learners     = len(set([l for _,l,_ in self._triples]))
        n_given_environments = len(set([e for e,_,_ in self._triples]))

        if restored:
            assert n_given_learners     == restored.experiment.get('n_learners',n_given_learners)        , "The current experiment doesn't match the given transaction log."
            assert n_given_environments == restored.experiment.get('n_environments',n_given_environments), "The current experiment doesn't match the given transaction log."

        meta = {'n_learners':n_given_learners,'n_environments':n_given_environments,'description':self._description,'seed':seed}

        workitems  = MakeTasks(self._triples)
        unfinished = ResumeTasks(restored)
        chunker    = ChunkTasks(mp)
        max_chunk  = MaxChunk(mt)
        process    = CobaMultiprocessor(ProcessTasks(), mp, mc, False)
        encode     = TransactionEncode()
        sink       = DiskSink(result_file) if result_file else ListSink(foreach=True)
        source     = DiskSource(result_file) if result_file else ListSource(sink.items)
        decode     = TransactionDecode()
        result     = TransactionResult()
        preamble   = Identity() if restored else Insert([["T0",meta]])

        try:
            CobaContext.logger.log("Experiment Started")
            Pipes.join(workitems, unfinished, chunker, max_chunk, process, preamble, encode, sink).run()
            CobaContext.logger.log("Experiment Finished")
        except KeyboardInterrupt: #pragma: no cover
            CobaContext.logger.log("Experiment Aborted (aborted via Ctrl-C)")
        except Exception as ex: #pragma: no cover
            CobaContext.logger.log(ex)
            CobaContext.logger.log("Experiment Failed")

        CobaContext.logger = old_logger
        del CobaContext.store['experiment_seed']

        return Pipes.join(source,decode,result).read()

    def evaluate(self, result_file:str = None) -> Result:
        """Evaluate the experiment and return the results (this is a backwards compatible proxy for the run method).

        Args:
            result_file: The file for writing and restoring results .
        """

        return self.run(result_file=result_file)

    def _parse_init_args(self,*args,**kwargs) -> Tuple[Sequence[Tuple[Environment,Learner]], Evaluator, Optional[str]]:
        #we know this with 100% certainty
        definite_paired  = ({'eval_pairs'} & kwargs.keys()             ) or (len(args) == 1 and 'learners' not in kwargs)
        definite_product = ({'environments','learners'} & kwargs.keys()) or (len(args) == 4)

        #we are making reasonable guesses
        likely_paired  = not definite_product and len(args)>0 and len(args[0])>0 and isinstance(args[0][0],tuple)
        likely_product = not definite_paired  and len(args)>0 and len(args[0])>0 and not isinstance(args[0][0],tuple)

        try:
            if definite_paired or (likely_paired and not definite_product):
                pairs       = args[0] if len(args) > 0 else kwargs['eval_pairs']
                evaluator   = args[1] if len(args) > 1 else kwargs.get('evaluator',OnPolicyEvaluator())
                description = args[2] if len(args) > 2 else kwargs.get('description',None)
                return pairs, evaluator, description

            if definite_product or (likely_product and not definite_paired):
                envs        = args[0] if len(args) > 0 else kwargs['environments']
                lrns        = args[1] if len(args) > 1 else kwargs['learners']
                evaluator   = args[2] if len(args) > 2 else kwargs.get('evaluator',OnPolicyEvaluator())
                description = args[3] if len(args) > 3 else kwargs.get('description',None)
                if not isinstance(envs,abc.Sequence): envs = [envs]
                if not isinstance(lrns,abc.Sequence): lrns = [lrns]
                pairs = list(product(envs,lrns))
                return pairs, evaluator, description

            if len(args) > 0 and len(args[0]) == 0:
                pairs       = []
                evaluator   = None #this doesn't matter because pairs is empty
                description = args[-1] if len(args) > 2 and isinstance(args[-1],str) else kwargs.get('description',None)
                return pairs, evaluator, description

            raise CobaException(f"We were unable to construct Experiment given *{args} **{kwargs}")
        except KeyError as e:
            raise TypeError(f'Experiment missing required argument {str(e)}')
