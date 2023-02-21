from copy import deepcopy
from itertools import islice
from collections import defaultdict, Counter
from typing import Iterable, Sequence, Any, Optional, Union, Tuple

from coba.pipes import Source, Filter, SourceFilters
from coba.learners import Learner
from coba.contexts import CobaContext
from coba.environments import Environment, Finalize, BatchSafe, Chunk
from coba.utilities import peek_first

from coba.experiments.tasks import LearnerTask, EnvironmentTask, EvaluationTask
from coba.experiments.results import Result

class WorkItem:

    def __init__(self,
        env_id: Optional[int],
        lrn_id: Optional[int],
        env   : Optional[Environment],
        lrn   : Optional[Learner],
        task  : Union[LearnerTask, EnvironmentTask, EvaluationTask]) -> None:

        self.env_id = env_id
        self.lrn_id = lrn_id
        self.env    = env
        self.lrn    = lrn
        self.task   = task

class CreateWorkItems(Source[Iterable[WorkItem]]):

    def __init__(self,
        evaluation_pairs: Sequence[Tuple[Learner,Environment]],
        learner_task    : LearnerTask,
        environment_task: EnvironmentTask,
        evaluation_task : EvaluationTask) -> None:

        self._evaluation_pairs = evaluation_pairs
        self._environment_task = environment_task
        self._learner_task     = learner_task
        self._evaluation_task  = evaluation_task

    def read(self) -> Iterable[WorkItem]:

        #we rely on ids to make sure we don't do duplicate work. So long as self.evaluation_pairs
        #is always in the exact same order we should be fine. In the future we may want to consider.
        #adding a better check for environments other than assigning an index based on their order.

        lrns = dict()
        envs = dict()

        for lrn,env in self._evaluation_pairs:

            if lrn not in lrns:
                lrns[lrn] = len(lrns)
                yield WorkItem(None, lrns[lrn], None, lrn, self._learner_task)

            if env not in envs:
                envs[env] = len(envs)
                yield WorkItem(envs[env], None, env, None, self._environment_task)

            yield WorkItem(envs[env], lrns[lrn], env, lrn, self._evaluation_task) 

class RemoveFinished(Filter[Iterable[WorkItem], Iterable[WorkItem]]):
    def __init__(self, restored: Optional[Result]) -> None:
        self._restored = restored

    def filter(self, tasks: Iterable[WorkItem]) -> Iterable[WorkItem]:

        finished_learners = set(self._restored.learners.col_values()[0]) if self._restored else set()
        finished_environments = set(self._restored.environments.col_values()[0]) if self._restored else set()
        finished_evaluations = set(zip(*self._restored.interactions.col_values()[:2])) if self._restored else set()


        for task in tasks:

            is_learner_task = task.env_id is None
            is_environ_task = task.lrn_id is None
            is_eval_task    = not (is_learner_task or is_environ_task)

            if not self._restored:
                yield task
            elif is_learner_task and task.lrn_id not in finished_learners:
                yield task
            elif is_environ_task and task.env_id not in finished_environments:
                yield task
            elif is_eval_task and (task.env_id, task.lrn_id) not in finished_evaluations:
                yield task

class ChunkByChunk(Filter[Iterable[WorkItem], Iterable[Sequence[WorkItem]]]):

    def filter(self, items: Iterable[WorkItem]) -> Iterable[Sequence[WorkItem]]:

        items  = list(items)
        chunks = defaultdict(list)

        workitems_sans_env = [t for t in items if t.env_id is None    ]
        workitems_with_env = [t for t in items if t.env_id is not None]

        for env_item in workitems_with_env:
            chunks[self._get_last_chunk(env_item.env)].append(env_item)

        for lrn_item in workitems_sans_env:
            yield [lrn_item]

        for workitem in chunks.pop('not_chunked',[]):
            yield [workitem]

        for chunk in sorted(chunks.values(), key=lambda chunk: min([c.env_id for c in chunk])):
            yield list(sorted(chunk, key=lambda c: (c.env_id, -1 if c.lrn_id is None else c.lrn_id)))

    def _get_last_chunk(self, env):
        if isinstance(env, SourceFilters):
            for pipe in reversed(list(env)):
                if isinstance(pipe, Chunk):
                    return pipe
        return 'not_chunked'

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

    def filter(self, chunks: Iterable[Iterable[WorkItem]]) -> Iterable[Any]:

        for chunk in chunks:

            chunk = list(chunk)
            empty_envs = set()

            finalizer = BatchSafe(Finalize())

            if not chunk: return

            if len(chunk) > 1:
                #We sort in case there are multiple chunks in the pipe. Sorting means we can free chunks from memory as we go.
                chunk = sorted(chunk, key=lambda item: self._env_ids(item)+self._lrn_ids(item))
                chunk = list(reversed(chunk))

            learner_eval_counts = Counter([item.lrn_id for item in chunk if item.lrn and item.env])

            with CobaContext.logger.log(f"Processing chunk..."):

                while chunk:
                    try:
                        item = chunk.pop()

                        if item.env is None:
                            with CobaContext.logger.time(f"Recording Learner {item.lrn_id} parameters..."):
                                row = item.task.process(item.lrn)
                                yield ["T1", item.lrn_id, row]

                        if item.lrn is None:
                            with CobaContext.logger.time(f"Recording Environment {item.env_id} statistics..."):
                                row = item.task.process(item.env,finalizer.filter(item.env.read()))
                                yield ["T2", item.env_id, row]

                        if item.env and item.lrn and item.env_id not in empty_envs:

                            with CobaContext.logger.time(f"Peeking at Environment {item.env_id}..."):
                                interactions = peek_first(item.env.read())[1]

                            if not interactions: 
                                CobaContext.logger.log(f"Environment {item.env_id} has nothing to evaluate (this is likely due to having too few interactions).")
                                empty_envs.add(item.env_id)
                                continue

                            with CobaContext.logger.time(f"Evaluating Learner {item.lrn_id} on Environment {item.env_id}..."):
                                lrn = item.lrn if learner_eval_counts[item.lrn_id] == 1 else deepcopy(item.lrn)
                                row = list(item.task.process(lrn, finalizer.filter(interactions)))
                                yield ["T3", (item.env_id, item.lrn_id), row]

                    except Exception as e:
                        CobaContext.logger.log(e)

    # def _cache_ids(self, item: WorkItem) -> Tuple[int,...]:
        #I'm not sure this is necessary and it makes sorting by env-ids difficult which 
        #isn't necessary but is nice to have when you're running experiments on a single 
        #process. Therefore, we're not going to use this code for now.
    #   return tuple(id(pipe) for pipe in reversed(list(item.env)) if isinstance(pipe,Cache) ) if isinstance(item.env, SourceFilters) else ()

    def _env_ids(self, item: WorkItem):
        return (-1,) if item.env_id is None else (item.env_id,)

    def _lrn_ids(self, item: WorkItem):
        return (-1,) if item.lrn_id is None else (item.lrn_id,)
 