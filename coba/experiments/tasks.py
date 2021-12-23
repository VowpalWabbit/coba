import collections
import collections.abc

from abc import ABC, abstractmethod
from statistics import median
from itertools import takewhile, chain
from typing import Iterable, Any, Dict

from coba.pipes import QueueIO
from coba.exceptions import CobaExit
from coba.random import CobaRandom
from coba.learners import Learner, SafeLearner
from coba.encodings import InteractionsEncoder
from coba.utilities import PackageChecker
from coba.contexts import LearnerContext
from coba.environments import Environment, EnvironmentPipe, Interaction, SimulatedInteraction, LoggedInteraction

class LearnerTask(ABC):

    @abstractmethod
    def process(self, item: Learner) -> Dict[Any,Any]:
        ...

class EnvironmentTask(ABC):

    @abstractmethod
    def process(self, environment: Environment, interactions: Iterable[Interaction]) -> Dict[Any,Any]:
        ...

class EvaluationTask(ABC):

    @abstractmethod
    def process(self, learner: Learner, interactions: Iterable[Interaction]) -> Iterable[Dict[Any,Any]]:
        ...

class AssertDictQueueIO(QueueIO[Dict[str,Any]]):

    def write(self, item: Dict[str,Any] = {}, **kwargs: Any) -> None:

        assert isinstance(item,dict), "The on-policy LearnerContext.logger was passed a non-dictionary object."
        item.update(kwargs)
        return super().write(item)


class OnPolicyEvaluationTask(EvaluationTask):

    def __init__(self, seed:int = 1):
        self._seed = seed

    def process(self, learner: Learner, interactions: Iterable[SimulatedInteraction]) -> Iterable[Dict[Any,Any]]:

        random = CobaRandom(self._seed)

        LearnerContext.logger = AssertDictQueueIO(block=False)

        if not isinstance(learner, SafeLearner): learner = SafeLearner(learner)
        if not interactions: return

        for interaction in interactions:

            context = interaction.context
            actions = interaction.actions

            probs,info = learner.predict(context, actions)

            action = random.choice(actions, probs)
            reveal = interaction.kwargs.get("reveals", interaction.kwargs.get("rewards"))[actions.index(action)]
            prob   = probs[actions.index(action)]

            learner.learn(context, action, reveal, prob, info)

            learner_info  = { k:v for item in LearnerContext.logger.read() for k,v in item.items() }
            interaction_info = {}

            for k,v in interaction.kwargs.items():
                if isinstance(v,collections.abc.Sequence) and not isinstance(v,str):
                    interaction_info[k] = v[actions.index(action)]
                else:
                    interaction_info[k] = v

            yield {**interaction_info, **learner_info}

class OffPolicyEvaluationTask(EvaluationTask):

    def process(self, learner: Learner, interactions: Iterable[LoggedInteraction]) -> Iterable[Dict[Any,Any]]:

        LearnerContext.logger = AssertDictQueueIO(block=False)

        if not isinstance(learner, SafeLearner): learner = SafeLearner(learner)
        if not interactions: return

        for interaction in interactions:

            if "actions" not in interaction.kwargs:
                info             = None
                interaction_info = {}

            if "actions" in interaction.kwargs:
                actions          = list(interaction.kwargs["actions"])
                probs,info       = learner.predict(interaction.context, actions)
                interaction_info = {}

            if len(interaction.kwargs.keys() & {"probability", "actions", "reward"}) == 3:
                ratio            = probs[actions.index(interaction.action)] / interaction.kwargs["probability"]
                interaction_info = { 'reward': ratio*interaction.kwargs['reward'] }

            for k,v in interaction.kwargs.items():
                if k not in ["probability", "actions", "reward"]:
                    interaction_info[k] = v

            reveal = interaction.kwargs.get("reveal", interaction.kwargs.get("reward"))
            prob   = interaction.kwargs.get("probability")
            learner.learn(interaction.context, interaction.action, reveal, prob, info)

            learner_info  = { k:v for item in LearnerContext.logger.read() for k,v in item.items() }

            yield {**interaction_info, **learner_info}

class WarmStartEvaluationTask(EvaluationTask):

    def __init__(self, seed: int = 1) -> None:
        self._seed = seed

    def process(self, learner: Learner, interactions: Iterable[Interaction]) -> Iterable[Dict[Any,Any]]:

        if not isinstance(learner, SafeLearner): learner = SafeLearner(learner)
        if not interactions: return

        separable_interactions = iter(self._repeat_first_simulated_interaction(interactions))

        logged_interactions    = takewhile(lambda i: isinstance(i,LoggedInteraction), separable_interactions)
        simulated_interactions = separable_interactions

        for row in OffPolicyEvaluationTask().process(learner, logged_interactions):
            yield row

        for row in OnPolicyEvaluationTask(self._seed).process(learner, simulated_interactions):
            yield row

    def _repeat_first_simulated_interaction(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        any_simulated_found = False

        for interaction in interactions:
            yield interaction

            if isinstance(interaction, SimulatedInteraction) and not any_simulated_found:
                yield interaction

            any_simulated_found = any_simulated_found or isinstance(interaction, SimulatedInteraction)

class SimpleLearnerTask(LearnerTask):

    def process(self, item: Learner) -> Dict[Any,Any]:
        item = SafeLearner(item)
        return {"full_name": item.full_name, **item.params}

class SimpleEnvironmentTask(EnvironmentTask):

    def process(self, environment:Environment, interactions: Iterable[Interaction]) -> Dict[Any,Any]:

            source = self._source_repr(environment)
            params = self._pipe_params(environment)

            return {"source": source, **params}

    def _source_repr(self, env) -> str:
        if isinstance(env, EnvironmentPipe):
            return str(env._source)
        else:
            return str(env)

    def _pipe_params(self, env) -> Dict[str,Any]:
        if isinstance(env, EnvironmentPipe):
            return env.params
        else:
            return {}

class ClassEnvironmentTask(EnvironmentTask):

    def process(self, environment: Environment, interactions: Iterable[SimulatedInteraction]) -> Dict[Any,Any]:

        contexts,actions,rewards = zip(*[ (i.context, i.actions, i.kwargs["rewards"]) for i in interactions ])
        env_statistics = {}

        try:

            PackageChecker.sklearn("ClassEnvironmentTask.process")

            import numpy as np
            import scipy.sparse as sp
            import scipy.stats as st
            from sklearn.feature_extraction import FeatureHasher
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import pairwise_distances
            from sklearn.decomposition import TruncatedSVD, PCA

            X   = [ InteractionsEncoder('x').encode(x=c, a=[]) for c in contexts ]
            Y   = [ a[r.index(1)] for a,r in zip(actions,rewards)]
            C   = collections.defaultdict(list)
            clf = DecisionTreeClassifier(random_state=1)

            if isinstance(X[0],dict):
                X = FeatureHasher(n_features=2**14, input_type="dict").fit_transform(X)

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

            env_statistics["centroid_purity"  ] = round(cluster_purity,4)
            env_statistics["centroid_distance"] = round(median(centroid_dists[range(centroid_dists.shape[0]),centroid_index]),4)

        except CobaExit:
            pass

        labels     = set()
        features   = set()
        feat_cnts  = []
        label_cnts = collections.defaultdict(int)

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

        return { **SimpleEnvironmentTask().process(environment, interactions), **env_statistics }
