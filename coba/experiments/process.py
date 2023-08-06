from copy import deepcopy
from itertools import islice
from operator import itemgetter
from collections import defaultdict, Counter
from typing import Any, Iterable, Sequence, Optional, Tuple, Mapping

from coba.learners import Learner, SafeLearner
from coba.environments import Environment, SafeEnvironment, Finalize, BatchSafe, Chunk
from coba.evaluators import Evaluator, SafeEvaluator

from coba.pipes import Source, Filter, SourceFilters
from coba.context import CobaContext
from coba.utilities import peek_first

from coba.experiments.results import Result, Table

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
    
    def __eq__(self, o: object) -> bool:
        return isinstance(o,Task) \
        and self.env_id == o.env_id \
        and self.env    == o.env    \
        and self.lrn_id == o.lrn_id \
        and self.lrn    == o.lrn    \
        and self.val_id == o.val_id \
        and self.val    == o.val    \
        and self.copy   == o.copy   \

class MakeTasks(Source[Iterable[Task]]):

    def __init__(self, 
        triples: Sequence[Tuple[Environment,Learner,Evaluator]],
        restored: Optional[Result] = None) -> None:
        
        self._triples = triples
        self._restored = restored or Result()

    def read(self) -> Iterable[Task]:

        #we rely on ids to make sure we don't do duplicate work. So long as self.evaluation_pairs
        #is always in the exact same order we should be fine. In the future we may want to consider.
        #adding a better check for environments other than assigning an index based on their order.

        envs = {None:None}
        lrns = {None:None}
        vals = {None:None}

        restored_lrns = set(self._restored.learners['learner_id'])
        restored_envs = set(self._restored.environments['environment_id'])
        restored_vals = set(self._restored.evaluators['evaluator_id'])
        restored_outs = set(zip(*self._restored.interactions[['environment_id','learner_id','evaluator_id']]))

        learner_counts = Counter([l for _,l,_ in self._triples])

        for env, lrn, val in self._triples:

            if env not in envs:
                envs[env] = len(envs)-1
                if envs[env] not in restored_envs:
                    yield Task((envs[env],env),None,None)

            if lrn not in lrns:
                lrns[lrn] = len(lrns)-1
                if lrns[lrn] not in restored_lrns:
                    yield Task(None,(lrns[lrn],lrn),None)

            if val not in vals:
                vals[val] = len(vals)-1
                if vals[val] not in restored_vals:
                    yield Task(None,None,(vals[val],val))

            eid,lid,vid = (envs[env],lrns[lrn],vals[val])
            if val and (eid,lid,vid) not in restored_outs:
                yield Task((eid,env),(lid,lrn),(vid,val),copy=learner_counts[lrn]>1)

class ChunkTasks(Filter[Iterable[Task], Iterable[Sequence[Task]]]):

    def __init__(self, max_tasks: int = None) -> None:
        self._max_tasks = max_tasks or None

    def filter(self, items: Iterable[Task]) -> Iterable[Sequence[Task]]:
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

        for task in chunks.pop('not_chunked',[]):
            yield [task]

        chunks_sorter = lambda c: min([c.env_id for c in c])
        chunk_sorter  = lambda t: (t.env_id, t.lrn_id if t.lrn else -1)

        for chunk in sorted(chunks.values(), key=chunks_sorter):
            yield from self._max_chunker(sorted(chunk, key=chunk_sorter), self._max_tasks)

    def _get_last_chunk(self, env):
        if isinstance(env, SourceFilters):
            for pipe in reversed(list(env)):
                if isinstance(pipe, Chunk):
                    return pipe
        return 'not_chunked'

    def _max_chunker(self, chunk, max_tasks):
        chunk = iter(chunk)
        batch = list(islice(chunk,max_tasks))
        while batch != []:
            yield batch
            batch = list(islice(chunk,max_tasks))

class ProcessTasks(Filter[Iterable[Task], Iterable[Any]]):

    def filter(self, chunk: Iterable[Task]) -> Iterable[Any]:

        chunk = list(chunk)
        empty_envs = set()

        if not chunk: return

        if len(chunk) > 1:
            #We sort to make sure cached envs are grouped. Sorting means we can free envs from memory as we go.
            chunk = sorted(chunk, key=lambda item: self._env_ids(item)+self._lrn_ids(item), reverse=True)

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
                        def params(self): return SafeEnvironment(dummy_env._env).params #pragma: no cover
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
