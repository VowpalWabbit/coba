import gc

from itertools import groupby, product
from collections import defaultdict
from typing import Iterable, Sequence, Any, Optional, Tuple, Union

from coba.learners import Learner
from coba.config import CobaConfig
from coba.pipes import Source, Pipe, Filter
from coba.environments import SimulatedEnvironment, EnvironmentPipe

from coba.experiments.tasks import LearnerTask, EnvironmentTask, EvaluationTask
from coba.experiments.results import Result, Transaction

class WorkItem:

    def __init__(self,
        learner: Optional[Tuple[int, Learner]], 
        environ: Optional[Tuple[int, SimulatedEnvironment]],
        task   : Union[LearnerTask, EnvironmentTask, EvaluationTask]) -> None:

        self.environ = environ
        self.learner = learner
        self.task    = task

class CreateWorkItems(Source[Iterable[WorkItem]]):

    def __init__(self, 
        environs        : Sequence[SimulatedEnvironment], 
        learners        : Sequence[Learner],
        learner_task    : LearnerTask,
        environment_task: EnvironmentTask,
        evaluation_task : EvaluationTask) -> None:
                
        self._environs = environs
        self._learners = learners
        
        self._learner_task     = learner_task
        self._environment_task = environment_task
        self._evaluation_task  = evaluation_task        

    def read(self) -> Iterable[WorkItem]:

        #we rely on sim_id to make sure we don't do duplicate work. So long as self._environs
        #is always in the exact same order we should be fine. In the future we may want to consider.
        #adding a better check for environments other than assigning an index based on their order.

        keyed_learners = dict(enumerate(self._learners))
        keyed_environs = dict(enumerate(self._environs))

        for lrn in keyed_learners.items():
            yield WorkItem(lrn, None, self._learner_task)

        for env in keyed_environs.items():
            yield WorkItem(None, env, self._environment_task)
            
        for lrn, env in product(keyed_learners.items(), keyed_environs.items()):
            yield WorkItem(lrn, env, self._evaluation_task)

class RemoveFinished(Filter[Iterable[WorkItem], Iterable[WorkItem]]):
    def __init__(self, restored: Result) -> None:
        self._restored = restored

    def filter(self, tasks: Iterable[WorkItem]) -> Iterable[WorkItem]:

        def is_not_complete(task: WorkItem):

            if not task.environ:
                return task.learner[0] not in self._restored.learners

            if not task.learner:
                return task.environ[0] not in self._restored.environments

            return (task.environ[0],task.learner[0]) not in self._restored._interactions

        return filter(is_not_complete, tasks)

class ChunkBySource(Filter[Iterable[WorkItem], Iterable[Sequence[WorkItem]]]):

    def filter(self, tasks: Iterable[WorkItem]) -> Iterable[Iterable[WorkItem]]:

        tasks  = list(tasks)
        chunks = defaultdict(list)

        for env_task in [t for t in tasks if t.environ]:
            chunks[self._get_source(env_task.environ[1])].append(env_task)

        for lrn_task in [t for t in tasks if not t.environ]:
            yield [lrn_task]

        for chunk in chunks.values():
            yield chunk
    
    def _get_source(self, env):
        return env._source  if isinstance(env, (EnvironmentPipe, Pipe.SourceFilters)) else env

class ChunkByTask(Filter[Iterable[WorkItem], Iterable[Iterable[WorkItem]]]):

    def filter(self, tasks: Iterable[WorkItem]) -> Iterable[Iterable[WorkItem]]:

        for task in tasks:
            yield [ task ]

class ProcessWorkItems(Filter[Iterable[WorkItem], Iterable[Any]]):

    def filter(self, chunk: Iterable[WorkItem]) -> Iterable[Any]:

        chunk = list(chunk)

        if not chunk: return

        with CobaConfig.logger.log(f"Processing chunk..."):

            for env_source, work_for_env_source in groupby(sorted(chunk, key=self._get_source_sort), key=self._get_source):

                try:

                    if env_source is None:
                        loaded_source = None
                    else:
                        with CobaConfig.logger.time(f"Loading source {env_source}..."):
                            
                            #This is not ideal. I'm not sure how it should be improved so it is being left for now.
                            #Maybe add a flag to the Experiment to say whether the source should be stashed in mem?
                            loaded_source = list(env_source.read())

                    filter_groups = [ (k,list(g)) for k,g in groupby(sorted(work_for_env_source, key=self._get_id_filter_sort), key=self._get_id_filter) ]

                    for (env_id, env_filter), work_for_env_filter in filter_groups:

                        if loaded_source is None:
                            interactions = []
                        else:
                            with CobaConfig.logger.time(f"Creating Environment {env_id} from {env_source}..."):
                                interactions = list(env_filter.filter(loaded_source)) if env_filter else loaded_source

                            if len(filter_groups) == 1:
                                #this will hopefully help with memory...
                                loaded_source = None
                                gc.collect()

                            if not interactions:
                                CobaConfig.logger.log(f"Environment {env_id} has nothing to evaluate (this is often due to `take` being larger than source).")
                                return

                        for workitem in work_for_env_filter:
                            try:

                                if workitem.environ is None:
                                    with CobaConfig.logger.time(f"Recording Learner {workitem.learner[0]} parameters..."):
                                        yield Transaction.learner(workitem.learner[0], **workitem.task.filter(workitem.learner[1]))

                                if workitem.learner is None:
                                    with CobaConfig.logger.time(f"Calculating Environment {workitem.environ[0]} statistics..."):
                                        yield Transaction.environment(workitem.environ[0], **workitem.task.filter((workitem.environ[1],interactions)))

                                if workitem.environ and workitem.learner:
                                    with CobaConfig.logger.time(f"Evaluating Learner {workitem.learner[0]} on Environment {workitem.environ[0]}..."):
                                        yield Transaction.interactions(workitem.environ[0], workitem.learner[0], workitem.task.filter((workitem.learner[1], interactions)))

                            except Exception as e:
                                CobaConfig.logger.log_exception(e)

                except Exception as e:
                    CobaConfig.logger.log_exception(e)

    def _get_source(self, task:WorkItem) -> SimulatedEnvironment:
        if task.environ is None: 
            return None
        elif isinstance(task.environ[1], (Pipe.SourceFilters, EnvironmentPipe)):
            return task.environ[1]._source 
        else:
            return task.environ[1]
    
    def _get_source_sort(self, task:WorkItem) -> int:
        return id(self._get_source(task))

    def _get_id_filter(self, task:WorkItem) -> Tuple[int, Filter[SimulatedEnvironment,SimulatedEnvironment]]:
        if task.environ is None:
            return (-1,None)
        elif isinstance(task.environ[1], (Pipe.SourceFilters, EnvironmentPipe)):
            return (task.environ[0], task.environ[1]._filter) 
        else:
            return (task.environ[0], None)

    def _get_id_filter_sort(self, task:WorkItem) -> int:
        return self._get_id_filter(task)[0]
