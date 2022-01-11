import gc

from copy import deepcopy
from itertools import groupby
from collections import defaultdict
from typing import Iterable, Sequence, Any, Optional, Tuple, Union

from coba.learners import Learner
from coba.contexts import CobaContext
from coba.pipes import Source, Filter, SourceFilters
from coba.environments import SimulatedEnvironment, FilteredEnvironment

from coba.experiments.tasks import LearnerTask, EnvironmentTask, EvaluationTask
from coba.experiments.results import Result

class WorkItem:

    def __init__(self,
        environ_id: Optional[int],
        learner_id: Optional[int],
        environ   : Optional[SimulatedEnvironment],
        learner   : Optional[Learner],        
        task      : Union[LearnerTask, EnvironmentTask, EvaluationTask]) -> None:

        self.learner_id = learner_id
        self.environ_id = environ_id
        self.learner = learner
        self.environ = environ
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

        for lrn_id,lrn in keyed_learners.items():
            yield WorkItem(None, lrn_id, None, lrn, self._learner_task)

        for env_id, env in keyed_environs.items():
            yield WorkItem(env_id, None, env, None, self._environment_task)

        for env_id,env in keyed_environs.items():
            for lrn_id,lrn in keyed_learners.items():
                yield WorkItem(env_id, lrn_id, env, lrn, self._evaluation_task)

class RemoveFinished(Filter[Iterable[WorkItem], Iterable[WorkItem]]):
    def __init__(self, restored: Result) -> None:
        self._restored = restored

    def filter(self, tasks: Iterable[WorkItem]) -> Iterable[WorkItem]:

        for task in tasks:

            is_learner_task = task.environ_id is None
            is_environ_task = task.learner_id is None
            is_eval_task    = not (is_learner_task or is_environ_task)

            if is_learner_task and task.learner_id not in self._restored.learners:
                yield task
            if is_environ_task and task.environ_id not in self._restored.environments:
                yield task
            if is_eval_task and (task.environ_id, task.learner_id) not in self._restored._interactions:
                yield task

class ChunkBySource(Filter[Iterable[WorkItem], Iterable[Sequence[WorkItem]]]):

    def filter(self, items: Iterable[WorkItem]) -> Iterable[Iterable[WorkItem]]:

        items  = list(items)
        chunks = defaultdict(list)

        sans_source_items = [t for t in items if t.environ_id is None]
        with_source_items = [t for t in items if t.environ_id is not None]

        for env_item in with_source_items:
            chunks[self._get_source(env_item.environ)].append(env_item)

        for lrn_item in sans_source_items:
            yield [lrn_item]

        for chunk in sorted(chunks.values(), key=lambda chunk: min([c.environ_id for c in chunk])):
            yield list(sorted(chunk, key=lambda c: (c.environ_id, -1 if c.learner_id is None else c.learner_id)))

    def _get_source(self, env):
        return env._source if isinstance(env, (FilteredEnvironment, SourceFilters)) else env

class ChunkByTask(Filter[Iterable[WorkItem], Iterable[Iterable[WorkItem]]]):

    def filter(self, workitems: Iterable[WorkItem]) -> Iterable[Iterable[WorkItem]]:

        workitems = list(workitems)

        learner_items = [w for w in workitems if w.environ_id is None]
        environ_items = [w for w in workitems if w.learner_id is None]
        eval_items    = [w for w in workitems if w.environ_id is not None and w.learner_id is not None]

        for item in sorted(learner_items, key = lambda t: t.learner_id):
            yield [ item ]

        for item in sorted(environ_items+eval_items, key=lambda t:( t.environ_id, t.learner_id if t.learner_id is not None else -1)):
            yield [ item ]

class ProcessWorkItems(Filter[Iterable[WorkItem], Iterable[Any]]):

    def filter(self, chunk: Iterable[WorkItem]) -> Iterable[Any]:

        chunk = list(chunk)

        self._source_id = {}

        for item in chunk:
            if item.environ_id is not None:
                source = id(self._get_source(item))
                self._source_id[source] = min(self._source_id.get(source,float('inf')), item.environ_id)

        if not chunk: return

        with CobaContext.logger.log(f"Processing chunk..."):

            for env_source, work_for_env_source in groupby(sorted(chunk, key=self._get_source_sort), key=self._get_source):

                try:

                    if env_source is None:
                        loaded_source = None
                    else:
                        with CobaContext.logger.time(f"Loading {env_source}..."):
                            #This is not ideal. I'm not sure how it should be improved so it is being left for now.
                            #Maybe add a flag to the Experiment to say whether the source should be stashed in mem?
                            loaded_source = list(env_source.read())

                    filter_groups = [ (k,list(g)) for k,g in groupby(sorted(work_for_env_source, key=self._get_id_filter_sort), key=self._get_id_filter) ]

                    for (env_id, env_filter), work_for_env_filter in filter_groups:

                        if loaded_source is None:
                            interactions = []
                        else:
                            with CobaContext.logger.time(f"Creating Environment {env_id} from Loaded Source..."):
                                interactions = list(env_filter.filter(loaded_source)) if env_filter else loaded_source

                            if len(filter_groups) == 1:
                                #this will hopefully help with memory...
                                loaded_source = None
                                gc.collect()

                            if not interactions:
                                CobaContext.logger.log(f"Environment {env_id} has nothing to evaluate (this is often due to `take` being larger than source).")
                                break

                        for workitem in work_for_env_filter:
                            try:

                                if workitem.environ is None:
                                    with CobaContext.logger.time(f"Recording Learner {workitem.learner_id} parameters..."):
                                        row = workitem.task.process(deepcopy(workitem.learner))
                                        yield ["T1", workitem.learner_id, row]

                                if workitem.learner is None:
                                    with CobaContext.logger.time(f"Recording Environment {workitem.environ_id} statistics..."):
                                        row = workitem.task.process(workitem.environ,interactions)
                                        yield ["T2", workitem.environ_id, row]

                                if workitem.environ and workitem.learner:
                                    with CobaContext.logger.time(f"Evaluating Learner {workitem.learner_id} on Environment {workitem.environ_id}..."):
                                        row = list(workitem.task.process(deepcopy(workitem.learner), interactions))
                                        yield ["T3", (workitem.environ_id, workitem.learner_id), row]

                            except Exception as e:
                                CobaContext.logger.log(e)

                except Exception as e:
                    CobaContext.logger.log(e)

    def _get_source(self, task:WorkItem) -> SimulatedEnvironment:
        if task.environ is None:
            return None
        elif isinstance(task.environ, (SourceFilters, FilteredEnvironment)):
            return task.environ._source 
        else:
            return task.environ
    
    def _get_source_sort(self, task:WorkItem) -> int:
        return self._source_id.get(id(self._get_source(task)),-1)

    def _get_id_filter(self, task:WorkItem) -> Tuple[int, Filter[SimulatedEnvironment,SimulatedEnvironment]]:
        if task.environ is None:
            return (-1,None)
        elif isinstance(task.environ, (SourceFilters, FilteredEnvironment)):
            return (task.environ_id, task.environ._filter) 
        else:
            return (task.environ_id, None)

    def _get_id_filter_sort(self, task:WorkItem) -> int:
        return self._get_id_filter(task)[0]
