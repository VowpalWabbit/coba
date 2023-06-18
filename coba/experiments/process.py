from copy import deepcopy
from itertools import islice
from collections import defaultdict, Counter
from typing import Any, Iterable, Sequence, Optional, Tuple

from coba.learners import Learner, SafeLearner
from coba.environments import Environment, SafeEnvironment, Finalize, BatchSafe, Chunk
from coba.evaluators import Evaluator

from coba.pipes import Source, Filter, SourceFilters
from coba.contexts import CobaContext
from coba.utilities import peek_first

from coba.experiments.results import Result

class Task:

    def __init__(self,
        env_id: Optional[int],
        lrn_id: Optional[int],
        env   : Optional[Environment],
        lrn   : Optional[Learner],
        task  : Evaluator,
        copy  : bool = False) -> None:

        self.env_id = env_id
        self.lrn_id = lrn_id
        self.env    = env
        self.lrn    = lrn
        self.task   = task
        self.copy   = copy

class MakeTasks(Source[Iterable[Task]]):

    def __init__(self,triples: Sequence[Tuple[Environment,Learner,Evaluator]]) -> None:
        self._triples = triples

    def read(self) -> Iterable[Task]:

        #we rely on ids to make sure we don't do duplicate work. So long as self.evaluation_pairs
        #is always in the exact same order we should be fine. In the future we may want to consider.
        #adding a better check for environments other than assigning an index based on their order.

        #we rely on ids to make sure we don't do duplicate work. So long as self.evaluation_pairs
        #is always in the exact same order we should be fine. In the future we may want to consider.
        #adding a better check for environments other than assigning an index based on their order.

        lrns = dict()
        envs = dict()

        learner_counts = Counter([l for _,l,_ in self._triples])

        for env, lrn, evl in self._triples:

            if lrn not in lrns:
                lrns[lrn] = len(lrns)
                yield Task(None, lrns[lrn], None, lrn, None)

            if env not in envs:
                envs[env] = len(envs)
                yield Task(envs[env], None, env, None, None)

            yield Task(envs[env], lrns[lrn], env, lrn, evl, copy=learner_counts[lrn]>1)

class ResumeTasks(Filter[Iterable[Task], Iterable[Task]]):
    def __init__(self, restored: Optional[Result]) -> None:
        self._restored = restored

    def filter(self, tasks: Iterable[Task]) -> Iterable[Task]:

        finished_learners = set(self._restored.learners['learner_id']) if self._restored else set()
        finished_environments = set(self._restored.environments['environment_id']) if self._restored else set()
        finished_evaluations = set(zip(*self._restored.interactions[['environment_id','learner_id']])) if self._restored else set()

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

class ChunkTasks(Filter[Iterable[Task], Iterable[Sequence[Task]]]):

    def __init__(self, n_processes: int) -> None:
        self._n_processes = n_processes

    def filter(self, items: Iterable[Task]) -> Iterable[Sequence[Task]]:
        if self._n_processes ==1:
            return [sum(self._chunks(items),[])]
        else:
            return self._chunks(items)

    def _chunks(self, items: Iterable[Task]) -> Iterable[Sequence[Task]]:
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

class MaxChunk(Filter[Iterable[Sequence[Task]], Iterable[Sequence[Task]]]):
    def __init__(self, max_tasks) -> None:
        self._max_tasks = max_tasks

    def filter(self, chunks: Iterable[Sequence[Task]]) -> Iterable[Sequence[Task]]:

        for chunk in chunks:
            chunk = iter(chunk)
            max_task_chunk = list(islice(chunk,self._max_tasks or None))
            while max_task_chunk:
                yield max_task_chunk
                max_task_chunk = list(islice(chunk,self._max_tasks or None))

class ProcessTasks(Filter[Iterable[Task], Iterable[Any]]):

    def filter(self, chunks: Iterable[Iterable[Task]]) -> Iterable[Any]:

        for chunk in chunks:

            chunk = list(chunk)
            empty_envs = set()

            if not chunk: return

            if len(chunk) > 1:
                #We sort to make sure cached envs are grouped. Sorting means we can free envs from memory as we go.
                chunk = sorted(chunk, key=lambda item: self._env_ids(item)+self._lrn_ids(item))
                chunk = list(reversed(chunk))

            with CobaContext.logger.log(f"Processing chunk..."):

                while chunk:
                    try:
                        item = chunk.pop()

                        if item.env is None:
                            with CobaContext.logger.time(f"Recording Learner {item.lrn_id} parameters..."):
                                yield ["T1", item.lrn_id, SafeLearner(item.lrn).params]

                        if item.lrn is None:
                            with CobaContext.logger.time(f"Recording Environment {item.env_id} statistics..."):
                                yield ["T2", item.env_id, SafeEnvironment(item.env).params]

                        if item.env and item.lrn and item.env_id not in empty_envs:

                            with CobaContext.logger.time(f"Peeking at Environment {item.env_id}..."):
                                interactions = peek_first(item.env.read())[1]

                            if not interactions:
                                CobaContext.logger.log(f"Environment {item.env_id} has nothing to evaluate (this is likely due to having too few interactions).")
                                empty_envs.add(item.env_id)
                                continue

                            class dummy_env:
                                env = item.env
                                @property
                                def params(self): return SafeEnvironment(item.env).params #pragma: no cover
                                def read(self): return BatchSafe(Finalize()).filter(interactions)

                            with CobaContext.logger.time(f"Evaluating Learner {item.lrn_id} on Environment {item.env_id}..."):
                                lrn = item.lrn if not item.copy else deepcopy(item.lrn)
                                env = dummy_env()
                                yield ["T3", (item.env_id, item.lrn_id), list(item.task.evaluate(env,lrn))]
                                if hasattr(lrn,'finish') and item.copy: lrn.finish()

                    except Exception as e:
                        CobaContext.logger.log(e)

    def _env_ids(self, item: Task):
        return (-1,) if item.env_id is None else (item.env_id,)

    def _lrn_ids(self, item: Task):
        return (-1,) if item.lrn_id is None else (item.lrn_id,)
 