from copy import deepcopy
from itertools import islice
from collections import defaultdict, Counter
from typing import Any, Iterable, Sequence, Optional, Tuple

from coba.learners import Learner, SafeLearner
from coba.environments import Environment, SafeEnvironment, Finalize, BatchSafe, Chunk
from coba.evaluators import Evaluator, SafeEvaluator

from coba.pipes import Source, Filter, SourceFilters
from coba.contexts import CobaContext
from coba.utilities import peek_first

from coba.experiments.results import Result

class Task:

    def __init__(self,
        env : Optional[Tuple[int,Environment]],
        lrn : Optional[Tuple[int,Learner,bool]],
        val : Optional[Tuple[int,Evaluator]],
        copy: bool = False) -> None:
        (self.env_id, self.env) = env or (None,None)
        (self.lrn_id, self.lrn) = lrn or (None,None)
        (self.val_id, self.val) = val or (None,None)
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

        envs = {None:None}
        lrns = {None:None}
        evls = {None:None}

        learner_counts = Counter([l for _,l,_ in self._triples])

        for env, lrn, evl in self._triples:

            if env not in envs:
                envs[env] = len(envs)-1
                yield Task((envs[env],env), None, None)

            if lrn not in lrns:
                lrns[lrn] = len(lrns)-1
                yield Task(None, (lrns[lrn],lrn), None)

            if evl not in evls:
                evls[evl] = len(evls)-1
                yield Task(None, None, (evls[evl],evl))

            if evl:
                yield Task((envs[env],env), (lrns[lrn],lrn), (evls[evl],evl), copy=learner_counts[lrn]>1)

class ResumeTasks(Filter[Iterable[Task], Iterable[Task]]):
    def __init__(self, restored: Optional[Result]) -> None:
        self._restored = restored or Result()

    def filter(self, tasks: Iterable[Task]) -> Iterable[Task]:

        finished_lrns = set(self._restored.learners['learner_id'])
        finished_envs = set(self._restored.environments['environment_id'])
        finished_evls = set(self._restored.evaluators['evaluator_id'])
        finished_outs = set(zip(*self._restored.interactions[['environment_id','learner_id','evaluator_id']]))

        for task in tasks:

            is_env_task = bool(task.env and not task.lrn and not task.val)
            is_lrn_task = bool(task.lrn and not task.env and not task.val)
            is_val_task = bool(task.val and not task.env and not task.lrn)
            is_out_task = bool(task.env and task.lrn and task.val)

            if is_env_task: task_id = task.env_id
            if is_lrn_task: task_id = task.lrn_id
            if is_val_task: task_id = task.val_id
            if is_out_task: task_id = (task.env_id,task.lrn_id,task.val_id)

            if is_env_task and task_id not in finished_envs:
                yield task
            if is_lrn_task and task_id not in finished_lrns:
                yield task
            if is_val_task and task_id not in finished_evls:
                yield task
            if is_out_task and task_id not in finished_outs:
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

        tasks_sans_env = [t for t in items if not t.env ]
        tasks_with_env = [t for t in items if     t.env ]

        for task in tasks_with_env:
            chunks[self._get_last_chunk(task.env)].append(task)

        for task in tasks_sans_env:
            yield [task]

        for tasks in chunks.pop('not_chunked',[]):
            yield [tasks]

        chunks_sorter = lambda c: min([c.env_id for c in c])
        chunk_sorter  = lambda t: (t.env_id, t.lrn_id if t.lrn else -1)

        for chunk in sorted(chunks.values(), key=chunks_sorter):
            yield list(sorted(chunk, key=chunk_sorter))

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
                        task = chunk.pop()

                        env_id,env = (task.env_id,task.env)
                        lrn_id,lrn = (task.lrn_id,task.lrn)
                        val_id,val = (task.val_id,task.val)

                        if task.copy: lrn = deepcopy(lrn)

                        if env and not lrn and not val:
                            with CobaContext.logger.time(f"Recording Environment {env_id} parameters..."):
                                yield ["T1", env_id, SafeEnvironment(env).params]

                        if lrn and not env and not val:
                            with CobaContext.logger.time(f"Recording Learner {lrn_id} parameters..."):
                                yield ["T2", lrn_id, SafeLearner(lrn).params]

                        if val and not env and not lrn:
                            with CobaContext.logger.time(f"Recording Evaluator {val_id} parameters..."):
                                yield ["T3", val_id, SafeEvaluator(val).params]

                        if env and lrn and val and env_id not in empty_envs:

                            with CobaContext.logger.time(f"Peeking at Environment {env_id}..."):
                                interactions = peek_first(env.read())[1]

                            if not interactions:
                                CobaContext.logger.log(f"Environment {env_id} has nothing to evaluate (this is likely due to having too few interactions).")
                                empty_envs.add(env_id)
                                continue

                            class dummy_env:
                                _env = env
                                @property
                                def params(self): return SafeEnvironment(env).params #pragma: no cover
                                def read(self): return BatchSafe(Finalize()).filter(interactions)

                            with CobaContext.logger.time(f"Evaluating Learner {lrn_id} on Environment {env_id}..."):
                                env = dummy_env()
                                yield ["T4", (env_id, lrn_id, val_id), list(SafeEvaluator(val).evaluate(env,lrn))]
                                if hasattr(lrn,'finish') and task.copy: lrn.finish()

                    except Exception as e:
                        CobaContext.logger.log(e)

    def _env_ids(self, item: Task):
        return (item.env_id if item.env else -1,)

    def _lrn_ids(self, item: Task):
        return (item.lrn_id if item.lrn else -1,)
