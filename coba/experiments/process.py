import gc

from copy import deepcopy
from itertools import groupby, islice
from collections import defaultdict
from typing import Iterable, Sequence, Any, Optional, Tuple, Union

from coba.pipes import Source, Filter, SourceFilters
from coba.learners import Learner
from coba.contexts import CobaContext
from coba.environments import Environment, Cache, EnvironmentFilter, Finalize, BatchSafe

from coba.experiments.tasks import LearnerTask, EnvironmentTask, EvaluationTask
from coba.experiments.results import Result

class WorkItem:

    def __init__(self,
        env_id: Optional[int],
        lrn_id: Optional[int],
        env   : Optional[Environment],
        lrn   : Optional[Learner],
        task  : Union[LearnerTask, EnvironmentTask, EvaluationTask]) -> None:

        self.lrn_id = lrn_id
        self.env_id = env_id
        self.lrn    = lrn
        self.env    = env
        self.task   = task

class CreateWorkItems(Source[Iterable[WorkItem]]):

    def __init__(self,
        environs        : Sequence[Environment],
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

        for lrn_id,lrn in keyed_learners.items():
            yield WorkItem(None, lrn_id, None, lrn, self._learner_task)

        for env_id,env in keyed_environs.items():
            yield WorkItem(env_id, None, env, None, self._environment_task)
            for lrn_id,lrn in keyed_learners.items():
                yield WorkItem(env_id, lrn_id, env, lrn, self._evaluation_task)

class RemoveFinished(Filter[Iterable[WorkItem], Iterable[WorkItem]]):
    def __init__(self, restored: Optional[Result]) -> None:
        self._restored = restored

    def filter(self, tasks: Iterable[WorkItem]) -> Iterable[WorkItem]:

        for task in tasks:

            is_learner_task = task.env_id is None
            is_environ_task = task.lrn_id is None
            is_eval_task    = not (is_learner_task or is_environ_task)

            if not self._restored:
                yield task
            elif is_learner_task and task.lrn_id not in self._restored.learners:
                yield task
            elif is_environ_task and task.env_id not in self._restored.environments:
                yield task
            elif is_eval_task and (task.env_id, task.lrn_id) not in self._restored._interactions:
                yield task

class ChunkBySource(Filter[Iterable[WorkItem], Iterable[Sequence[WorkItem]]]):

    def filter(self, items: Iterable[WorkItem]) -> Iterable[Sequence[WorkItem]]:

        items  = list(items)
        chunks = defaultdict(list)

        sans_source_items = [t for t in items if t.env_id is None]
        with_source_items = [t for t in items if t.env_id is not None]

        for env_item in with_source_items:
            chunks[self._get_source(env_item.env)].append(env_item)

        for lrn_item in sans_source_items:
            yield [lrn_item]

        for chunk in sorted(chunks.values(), key=lambda chunk: min([c.env_id for c in chunk])):
            yield list(sorted(chunk, key=lambda c: (c.env_id, -1 if c.lrn_id is None else c.lrn_id)))

    def _get_source(self, env):
        return env._source if isinstance(env, SourceFilters) else env

class ChunkByTask(Filter[Iterable[WorkItem], Iterable[Sequence[WorkItem]]]):

    def filter(self, workitems: Iterable[WorkItem]) -> Iterable[Sequence[WorkItem]]:

        #We used to sort by source before performing this action
        #This was nice because it meant a single openml data set
        #Could be loaded and then a bunch of tasks would be able
        #to use the cached version of it. Just leaving this note
        #here for now in case I want to change back in the future.

        for workitem in workitems:
            yield [workitem]

class MaxChunkSize(Filter[Iterable[Sequence[WorkItem]], Iterable[Sequence[WorkItem]]]):
    def __init__(self, max_tasks) -> None:
        self._max_tasks = max_tasks

    def filter(self, chunks: Iterable[Sequence[WorkItem]]) -> Iterable[Sequence[WorkItem]]:

        for chunk in chunks:
            chunk = iter(chunk)
            max_task_chunk = list(islice(chunk,self._max_tasks or None))
            while max_task_chunk:
                yield max_task_chunk
                max_task_chunk = list(islice(chunk,self._max_tasks or None))

class ProcessWorkItems(Filter[Iterable[WorkItem], Iterable[Any]]):

    def filter(self, chunk: Iterable[WorkItem]) -> Iterable[Any]:

        chunk = list(chunk)

        self._source_id = {}

        for item in chunk:
            if item.env_id is not None:
                source = id(self._get_source(item))
                self._source_id[source] = min(self._source_id.get(source,item.env_id), item.env_id)

        if not chunk: return

        with CobaContext.logger.log(f"Processing chunk..."):

            for env_source, work_for_env_source in groupby(sorted(chunk, key=self._get_source_sort), key=self._get_source):

                try:
                    if env_source is None:
                        loaded_source = None
                    else:
                        with CobaContext.logger.time(f"Loading {env_source._source if isinstance(env_source,SourceFilters) else env_source}..."):
                            #This is not ideal. I'm not sure how it should be improved so it is being left for now.
                            #Maybe add a flag to the Experiment to say whether the source should be stashed in mem?
                            loaded_source = list(env_source.read())

                    #if a learner only has one eval 

                    filter_groups = [ (k,list(g)) for k,g in groupby(sorted(work_for_env_source, key=self._get_id_filter_sort), key=self._get_id_filter) ]

                    for (env_id, env_filter), work_for_env_filter in filter_groups:

                        if loaded_source is None:
                            interactions = []
                        else:
                            with CobaContext.logger.time(f"Creating Environment {env_id} from Loaded Source..."):
                                interactions = list(env_filter.filter(loaded_source)) if env_filter else loaded_source

                            interactions = list(BatchSafe(Finalize()).filter(interactions))

                            if len(filter_groups) == 1:
                                #this will hopefully help with memory...
                                loaded_source = None
                                gc.collect()

                            if not interactions:
                                CobaContext.logger.log(f"Environment {env_id} has nothing to evaluate (this is likely due to having too few interactions).")
                                break

                        for workitem in work_for_env_filter:
                            try:

                                if workitem.env is None:
                                    with CobaContext.logger.time(f"Recording Learner {workitem.lrn_id} parameters..."):
                                        row = workitem.task.process(workitem.lrn)
                                        yield ["T1", workitem.lrn_id, row]

                                if workitem.lrn is None:
                                    with CobaContext.logger.time(f"Recording Environment {workitem.env_id} statistics..."):
                                        row = workitem.task.process(workitem.env,interactions)
                                        yield ["T2", workitem.env_id, row]

                                if workitem.env and workitem.lrn:
                                    with CobaContext.logger.time(f"Evaluating Learner {workitem.lrn_id} on Environment {workitem.env_id}..."):
 
                                        if len([i for i in chunk if i.env and i.lrn and i.lrn_id == workitem.lrn_id]) > 1:
                                            learner = deepcopy(workitem.lrn)
                                        else:
                                            learner = workitem.lrn

                                        row = list(workitem.task.process(learner, interactions))
                                        yield ["T3", (workitem.env_id, workitem.lrn_id), row]

                            except Exception as e:
                                CobaContext.logger.log(e)

                except Exception as e:
                    CobaContext.logger.log(e)

    def _get_source(self, task:WorkItem) -> Environment:
        if task.env is None:
            return None
        elif isinstance(task.env, SourceFilters) and not isinstance(task.env[-1], Cache):
            return task.env._source
        else:
            return task.env

    def _get_source_sort(self, task:WorkItem) -> int:
        return self._source_id.get(id(self._get_source(task)),-1)

    def _get_id_filter(self, task:WorkItem) -> Tuple[int, EnvironmentFilter]:
        if task.env is None:
            return (-1,None)
        elif isinstance(task.env, SourceFilters):
            return (task.env_id, task.env._filter)
        else:
            return (task.env_id, None)

    def _get_id_filter_sort(self, task:WorkItem) -> int:
        return self._get_id_filter(task)[0]
