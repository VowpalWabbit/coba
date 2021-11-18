from abc import ABC, abstractmethod
from copy import deepcopy
from statistics import median
from collections import defaultdict
from itertools import takewhile, chain
from typing import Iterable, Any, Dict, Tuple

from coba.random import CobaRandom
from coba.learners import Learner, SafeLearner
from coba.pipes import Filter
from coba.environments import Environment, EnvironmentPipe, Interaction, SimulatedInteraction, LoggedInteraction
from coba.encodings import InteractionTermsEncoder

class LearnerTask(Filter[Learner,Dict[Any,Any]], ABC):

    @abstractmethod
    def filter(self, item: Learner) -> Dict[Any,Any]:
        ...

class EnvironmentTask(Filter[Tuple[Environment,Iterable[Interaction]], Dict[Any,Any]]):

    @abstractmethod
    def filter(self, item: Tuple[Environment,Iterable[Interaction]]) -> Dict[Any,Any]:
        ...

class EvaluationTask(Filter[Tuple[Learner,Iterable[Interaction]], Iterable[Dict[Any,Any]]]):

    @abstractmethod
    def filter(self, item: Tuple[Learner,Iterable[Interaction]]) -> Iterable[Dict[Any,Any]]:
        ...

class OnPolicyEvaluationTask(EvaluationTask):

    def __init__(self, seed:int = 1):
        self._seed = seed

    def filter(self, item: Tuple[Learner,Iterable[SimulatedInteraction]]) -> Iterable[Dict[Any,Any]]:

        learner      = deepcopy(item[0])
        interactions = item[1]
        random       = CobaRandom(self._seed)

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

            learn_info  = info
            action_info = {k:v[actions.index(action)] for k,v in interaction.kwargs.items()}

            yield {**action_info, **learn_info}

class OffPolicyEvaluationTask(EvaluationTask):

    def filter(self, item: Tuple[Learner,Iterable[LoggedInteraction]]) -> Iterable[Dict[Any,Any]]:

        learner      = deepcopy(item[0])
        interactions = item[1]

        if not isinstance(learner, SafeLearner): learner = SafeLearner(learner)
        if not interactions: return

        for interaction in interactions:

            if not interaction.probability or not interaction.actions:
                predict_info = {}
            else:
                probs = learner.predict(interaction.context, interaction.actions)[0]
                ratio = probs[interaction.actions.index(interaction.action)] / interaction.probability
                predict_info = { "reward": ratio * interaction.reward }

            learn_info = learner.learn(interaction.context, interaction.action, interaction.reward, interaction.probability)

            yield {**predict_info, **learn_info}

class WarmStartEvaluationTask(EvaluationTask):

    def __init__(self, seed: int) -> None:
        self._seed = seed

    def filter(self, item: Tuple[Learner,Iterable[Interaction]]) -> Iterable[Dict[Any,Any]]:

        learner      = deepcopy(item[0])
        interactions = item[1]

        if not isinstance(learner, SafeLearner): learner = SafeLearner(learner)
        if not interactions: return

        separable_interactions = iter(self._repeat_first_simulated(interactions))

        logged_interactions    = takewhile(lambda i: isinstance(i,LoggedInteraction), separable_interactions)
        simulated_interactions = separable_interactions

        for row in OffPolicyEvaluationTask().filter( (learner, logged_interactions)):
            yield row

        for row in OnPolicyEvaluationTask(self._seed).filter( (learner, simulated_interactions)):
            yield row

    def _repeat_first_simulated(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        for interaction in interactions:
            if isinstance(interaction, LoggedInteraction):
                yield interaction
            else:
                first_simulated_interaction = interaction
                break
        
        for interaction in chain([first_simulated_interaction], interactions):
            yield interaction

class SimpleLearnerTask(LearnerTask):
    def filter(self, item: Learner) -> Dict[Any,Any]:
        item = SafeLearner(item)
        return {"full_name": item.full_name, **item.params}

class SimpleEnvironmentTask(EnvironmentTask):

    def filter(self, item: Tuple[Environment,Iterable[Interaction]]) -> Dict[Any,Any]:

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

    def filter(self, item: Tuple[Environment,Iterable[SimulatedInteraction]]) -> Dict[Any,Any]:

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
            C   = defaultdict(list)
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
