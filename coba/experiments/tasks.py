import collections
import gc

from abc import ABC, abstractmethod
from copy import deepcopy
from statistics import median
from itertools import groupby, product
from collections import defaultdict
from typing import Iterable, Sequence, Any, Optional, Dict, Tuple, Union

from numpy.lib.arraysetops import isin

from coba.random import CobaRandom
from coba.learners import Learner, SafeLearner
from coba.config import CobaConfig
from coba.pipes import Source, Pipe, Filter
from coba.environments import SimulatedEnvironment, EnvironmentPipe, LoggedInteraction, SimulatedInteraction
from coba.encodings import InteractionTermsEncoder

from coba.experiments.transactions import Transaction
from coba.experiments.results import Result

class LearnerTask(Filter[Learner,Dict[Any,Any]], ABC):

    @abstractmethod
    def filter(self, item: Learner) -> Dict[Any,Any]:
        ...

class EnvironmentTask(Filter[Tuple[SimulatedEnvironment,Iterable[Union[SimulatedInteraction, LoggedInteraction]]], Dict[Any,Any]]):

    @abstractmethod
    def filter(self, interactions: Iterable[Union[SimulatedInteraction, LoggedInteraction]]) -> Dict[Any,Any]:
        ...

class EvaluationTask(Filter[Tuple[Learner,Iterable[Union[SimulatedInteraction, LoggedInteraction]]], Iterable[Dict[Any,Any]]]):

    @abstractmethod
    def filter(self, item: Tuple[Learner,Iterable[Union[SimulatedInteraction, LoggedInteraction]]]) -> Iterable[Dict[Any,Any]]:
        return super().filter(item)

class WorkItem(Filter[Iterable[SimulatedInteraction], Iterable[Any]]):

    def __init__(self,
        learner: Optional[Tuple[int, Learner]], 
        environ: Optional[Tuple[int, SimulatedEnvironment]],
        task   : Union[LearnerTask, EnvironmentTask, EvaluationTask],
        seed   : int = None) -> None:

        self.environ = environ
        self.learner = learner
        self.task    = task
        self.seed    = seed

    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Tuple[str, Union[int,Tuple[int,...]], Dict[Any,Any] ]:
        if self.environ is None:
            with CobaConfig.logger.time(f"Recording Learner {self.learner[0]} parameters..."):
                return ["L", self.learner[0], self.task.filter(self.learner[1]) ]

        if self.learner is None:
            with CobaConfig.logger.time(f"Calculating Simulation {self.environ[0]} statistics..."):
                return ["S", self.environ[0], self.task.filter((self.environ[1],interactions)) ]

        if self.environ and self.learner:
            with CobaConfig.logger.time(f"Evaluating learner {self.learner[0]} on Simulation {self.environ[0]}..."):
                evals  = self.task.filter((self.learner[1], interactions, CobaRandom(self.seed)))
                rows_T = defaultdict(list)

                for row in evals:
                    for col,val in row.items():
                        if col == "rewards" : col="reward"
                        if col == "reveals" : col="reveal"
                        rows_T[col].append(val)

                return ["I", (self.environ[0], self.learner[0]), {"_packed": rows_T}]

class OnPolicyEvaluationTask(EvaluationTask):

    def filter(self, item: Tuple[Learner,Iterable[SimulatedInteraction], CobaRandom]) -> Iterable[Dict[Any,Any]]:

        learner      = deepcopy(item[0])
        interactions = item[1]
        random       = item[2]

        if not isinstance(learner, SafeLearner): learner = SafeLearner(learner)
        if not interactions: return

        for interaction in interactions:

            context = interaction.context
            actions = interaction.actions

            probs,info = learner.predict(context, actions)

            action = random.choice(actions, probs)
            reveal = interaction.kwargs.get("reveals", interaction.kwargs["rewards"])[actions.index(action)]
            prob   = probs[actions.index(action)]

            info = learner.learn(context, action, reveal, prob, info) or {}

            learn_info = info
            choice_info = {k:v[actions.index(action)] for k,v in interaction.kwargs.items()}

            yield {**choice_info, **learn_info}

class WarmStartEvaluationTask(EvaluationTask):

    def __init__(self, src_id:int, sim_id: int, lrn_id: int, simulation: SimulatedEnvironment, learner: Learner, seed: int) -> None:
        self._seed = seed
        super().__init__(src_id, sim_id, lrn_id, simulation, learner)

    def filter(self, interactions: Iterable[Union[LoggedInteraction,SimulatedInteraction]]) -> Iterable[Any]:

        if not interactions: return

        learner = deepcopy(self.learner)
        random  = CobaRandom(self._seed)

        with CobaConfig.logger.time(f"Evaluating learner {self.lrn_id} on Simulation {self.sim_id}..."):

            row_data = defaultdict(list)

            seen_keys = []

            for i,interaction in enumerate(interactions):

                probs, info, action, reveal, prob, info = None
                context = interaction.context
                actions = interaction.actions

                if isinstance(interaction, LoggedInteraction):
                    reveal = interaction.reward
                    prob = interaction.probability
                    probs,info = learner.predict(context, actions)
                    ratio = reveal * probs[actions.index(interaction.action)] / prob

                    info = learner.learn(context, actions, reveal, prob, info) or {}
                    interaction_data = {"reward": ratio*interaction.reward}
                else:
                    probs,info = learner.predict(context, actions)

                    action = random.choice(actions, probs)
                    reveal = interaction.reveals[actions.index(action)]
                    prob   = probs[actions.index(action)]
                    info = learner.learn(context, action, reveal, prob, info) or {}
                    interaction_data = {k:v[actions.index(action)] for k,v in interaction.results.items()}
                
                row_key_values = {}
                row_key_values.update(info)
                row_key_values.update(interaction_data)

                for key,value in row_key_values.items():
                    if key == "rewards": key = "reward"
                    if key == "reveals": key = "reveal"

                    if key in seen_keys:
                        row_data[key].append(value)
                    else:
                        seen_keys.append(key)
                        row_data[key].extend([None]*i)
                        row_data[key].append(value)

            yield Transaction.interactions(self.sim_id, self.lrn_id, _packed=row_data)

class SimpleLearnerTask(LearnerTask):
    def filter(self, item: Learner) -> Dict[Any,Any]:
        item = SafeLearner(item)
        return {"full_name": item.full_name, **item.params}

class SimpleEnvironmentTask(EnvironmentTask):

    def filter(self, item: Tuple[SimulatedEnvironment,Iterable[Union[SimulatedInteraction, LoggedInteraction]]]) -> Dict[Any,Any]:

            environment  = item[0]

            source = self._source_repr(environment)
            params = self._pipe_params(environment)

            return {"source": source, **params}
            
    def _source_repr(self, env) -> str:
        if isinstance(env, EnvironmentPipe):
            return env.source_repr
        else:
            return env.__class__.__name__
    
    def _pipe_params(self, env) -> Dict[str,Any]:
        if isinstance(env, EnvironmentPipe):
            return env.params
        else:
            return {}

class ClassEnvironmentTask(EnvironmentTask):

    def filter(self, item: Tuple[SimulatedEnvironment,Iterable[Union[SimulatedInteraction, LoggedInteraction]]]) -> Dict[Any,Any]:

        contexts,actions,rewards = zip(*[ (i.context, i.actions, i.kwargs["rewards"]) for i in item[1]])

        try:
            import numpy as np
            import scipy.sparse as sp
            import scipy.stats as st
            from sklearn.feature_extraction import FeatureHasher
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import pairwise_distances
            from sklearn.decomposition import TruncatedSVD, PCA

            env_statistics = {}

            X   = [ InteractionTermsEncoder('x').encode(x=c, a=[]) for c in contexts ]
            Y   = [ a[r.index(1)] for a,r in zip(actions,rewards)]
            C   = collections.defaultdict(list)
            clf = DecisionTreeClassifier(random_state=1)

            if isinstance(X[0][0],tuple):
                X = FeatureHasher(n_features=2**14, input_type="pair").fit_transform(X)

            if len(Y) > 5:
                scores = cross_val_score(clf, X, Y, cv=5)
                env_statistics["bayes_rate_avg"] = round(scores.mean(),4)
                env_statistics["bayes_rate_iqr"] = round(st.iqr(scores),4)

            svd = TruncatedSVD(n_components=8) if sp.issparse(X) else PCA()                    
            svd.fit(X)
            env_statistics["PcaVarExplained"] = svd.explained_variance_ratio_[:8].tolist()

            for x,y in zip(X,Y):
                C[y].append(x)

            if sp.issparse(X):
                centroids = sp.vstack([sp.csr_matrix(sp.vstack(c).mean(0)) for c in C.values()])
            else:
                centroids = np.vstack([np.vstack(c).mean(0) for c in C.values()])

            centroid_order = list(C.keys())
            centroid_index = [ centroid_order.index(y) for y in Y ]
            centroid_dists = pairwise_distances(X,centroids)
            closest_index  = centroid_dists.argmin(1)
            cluster_purity = (closest_index == centroid_index).mean()

            env_statistics["centroid_purity"]   = round(cluster_purity,4)
            env_statistics["centroid_distance"] = round(median(centroid_dists[range(centroid_dists.shape[0]),centroid_index]),4)

        except ImportError:
            pass

        labels     = set()
        features   = set()
        feat_cnts  = []
        label_cnts = defaultdict(int)

        for c,a,f in zip(contexts,actions,rewards):

            inter_label = a[f.index(1)]
            inter_feats = c.keys() if isinstance(c,dict) else range(len(c))

            labels.add(inter_label)
            features.update(inter_feats)
            feat_cnts.append(len(inter_feats))
            label_cnts[inter_label] += 1

        env_statistics["action_cardinality"] = len(labels)
        env_statistics["context_dimensions"] = len(features)
        env_statistics["context_median_nz" ] = median(feat_cnts)
        env_statistics["imbalance_ratio"   ] = round(max(label_cnts.values())/min(label_cnts.values()),4)

        return { **SimpleEnvironmentTask().filter(item), **env_statistics }

class CreateWorkItems(Source[Iterable[WorkItem]]):

    def __init__(self, 
        environs        : Sequence[SimulatedEnvironment], 
        learners        : Sequence[Learner],
        learner_task    : LearnerTask,
        environment_task: EnvironmentTask,
        evaluation_task : EvaluationTask,        
        seed            : int = None) -> None:
                
        self._environs = environs
        self._learners = learners
        
        self._learner_task     = learner_task
        self._environment_task = environment_task
        self._evaluation_task  = evaluation_task
        
        self._seed = seed

    def read(self) -> Iterable[WorkItem]:

        #we rely on sim_id to make sure we don't do duplicate work. So long as self._simulations
        #is always in the exact same order we should be fine. In the future we may want to consider.
        #adding a better check for simulations other than assigning an index based on their order.

        keyed_learners = dict(enumerate(self._learners))
        keyed_environs = dict(enumerate(self._environs))

        for lrn in keyed_learners.items():
            yield WorkItem(lrn, None, self._learner_task, self._seed)

        for env in keyed_environs.items():
            yield WorkItem(None, env, self._environment_task, self._seed)
            
        for lrn, env in product(keyed_learners.items(), keyed_environs.items()):
            yield WorkItem(lrn, env, self._evaluation_task, self._seed)
            
class RemoveFinished(Filter[Iterable[WorkItem], Iterable[WorkItem]]):
    def __init__(self, restored: Result) -> None:
        self._restored = restored

    def filter(self, tasks: Iterable[WorkItem]) -> Iterable[WorkItem]:

        def is_not_complete(task: WorkItem):

            if not task.environ:
                return task.learner[0] not in self._restored.learners

            if not task.learner:
                return task.environ[0] not in self._restored.simulations

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

        with CobaConfig.logger.log(f"Processing chunk..."):

            for env_source, tasks_for_env_source in groupby(sorted(chunk, key=self._get_source_sort), key=self._get_source):

                try:

                    if env_source is None:
                        loaded_source = None
                    else:
                        with CobaConfig.logger.time(f"Loading source {env_source}..."):
                            
                            #This is not ideal. I'm not sure how it should be improved so it is being left for now.
                            #Maybe add a flag to the Benchmark to say whether the source should be stashed in mem?
                            loaded_source = list(env_source.read())

                    filter_groups = [ (k,list(g)) for k,g in groupby(sorted(tasks_for_env_source, key=self._get_id_filter_sort), key=self._get_id_filter) ]

                    for (env_id,env_filter), tasks_for_env_filter in filter_groups:

                        if loaded_source is None:
                            interactions = []
                        else:
                            with CobaConfig.logger.time(f"Creating simulation {env_id} from {env_source}..."):
                                interactions = list(env_filter.filter(loaded_source)) if env_filter else loaded_source

                            if len(filter_groups) == 1:
                                #this will hopefully help with memory...
                                loaded_source = None
                                gc.collect()

                            if not interactions:
                                CobaConfig.logger.log(f"Simulation {env_id} has nothing to evaluate (this is often due to `take` being larger than source).")
                                return

                        for task in tasks_for_env_filter:
                            try:
                                yield task.filter(interactions)
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
