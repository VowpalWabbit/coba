import time
import collections
import collections.abc

from abc import ABC, abstractmethod
from statistics import median
from itertools import takewhile
from typing import Iterable, Any, Dict

from coba.pipes import QueueIO
from coba.exceptions import CobaExit
from coba.random import CobaRandom
from coba.learners import Learner, SafeLearner
from coba.encodings import InteractionsEncoder
from coba.utilities import PackageChecker
from coba.contexts import LearnerContext
from coba.environments import Environment, FilteredEnvironment, Interaction, SimulatedInteraction, LoggedInteraction

class LearnerTask(ABC):
    """A task which describes a Learner."""
    
    @abstractmethod
    def process(self, learner: Learner) -> Dict[Any,Any]:
        """Process the LearnerTask.

        Args:
            learner: The learner that we wish to describe.

        Returns:
            A dictionary which describes the given learner.
        """
        ...

class EnvironmentTask(ABC):
    """A task which describes an Environment."""
    
    @abstractmethod
    def process(self, environment: Environment, interactions: Iterable[Interaction]) -> Dict[Any,Any]:
        """Process the EnvironmentTask.

        Args:
            environment: The environment that we wish to describe.
            interactions: The interactions which belong to the environment.

        Returns:
            A dictionary which describes the given environment.
        """
        ...

class EvaluationTask(ABC):
    """A task which evaluates a Learner on an Environment."""

    @abstractmethod
    def process(self, learner: Learner, interactions: Iterable[Interaction]) -> Iterable[Dict[Any,Any]]:
        """Process the EvaluationTask.

        Args:
            learner: The Learner that we wish to evaluate.
            interactions: The Interactions which we wish to evaluate on.

        Returns:
            An iterable of evaluation statistics (for online evaluation there should be one dict per interaction).
        """
        ...

class AssertDictQueueIO(QueueIO[Dict[str,Any]]):

    def write(self, item: Dict[str,Any] = {}, **kwargs: Any) -> None:

        assert isinstance(item,dict), "The on-policy LearnerContext.logger was passed a non-dictionary object."
        item.update(kwargs)
        return super().write(item)

class OnlineOnPolicyEvalTask(EvaluationTask):
    """Evaluate a Learner on a SimulatedEnvironment."""

    def __init__(self, time:bool = True, seed:int = 1) -> None:
        """Instantiate an OnlineOnPolicyEvalTask.

        Args:
            time: Indicator for whether learner predict and learn times should be recorded. 
            seed: A random seed which determines which action is taken given predictions.        
        """
        self._seed = seed
        self._time = time

    def process(self, learner: Learner, interactions: Iterable[SimulatedInteraction]) -> Iterable[Dict[Any,Any]]:

        random = CobaRandom(self._seed)

        LearnerContext.logger = AssertDictQueueIO(block=False)

        if not isinstance(learner, SafeLearner): learner = SafeLearner(learner)
        if not interactions: return

        for interaction in interactions:

            context = interaction.context
            actions = interaction.actions

            start_time = time.time()
            probs,info = learner.predict(context, actions)
            predict_time = time.time() - start_time

            action = random.choice(actions, probs)
            reveal = interaction.kwargs.get("reveals", interaction.kwargs.get("rewards"))[actions.index(action)]
            prob   = probs[actions.index(action)]

            start_time = time.time()
            learner.learn(context, action, reveal, prob, info)
            learn_time = time.time() - start_time

            learner_info  = { k:v for item in LearnerContext.logger.read() for k,v in item.items() }
            interaction_info = {}

            for k,v in interaction.kwargs.items():
                if isinstance(v,collections.abc.Sequence) and not isinstance(v,str):
                    interaction_info[k] = v[actions.index(action)]
                else:
                    interaction_info[k] = v

            time_info = {"predict_time": predict_time, "learn_time": learn_time} if self._time else {}

            yield {**interaction_info, **learner_info, **time_info}

class OnlineOffPolicyEvalTask(EvaluationTask):
    """Evaluate a Learner on a LoggedEnvironment."""

    def __init__(self, time:bool = True) -> None:
        self._time = time

    def process(self, learner: Learner, interactions: Iterable[LoggedInteraction]) -> Iterable[Dict[Any,Any]]:
        LearnerContext.logger = AssertDictQueueIO(block=False)

        learn_time = 0
        predict_time = 0

        if not isinstance(learner, SafeLearner): learner = SafeLearner(learner)
        if not interactions: return

        for interaction in interactions:

            if "actions" not in interaction.kwargs:
                info             = None
                interaction_info = {}

            if "actions" in interaction.kwargs:
                actions          = list(interaction.kwargs["actions"])
                start_time       = time.time()
                probs,info       = learner.predict(interaction.context, actions)
                predict_time     = time.time()-start_time
                interaction_info = {}

            if len(interaction.kwargs.keys() & {"probability", "actions", "reward"}) == 3:
                ratio            = probs[actions.index(interaction.action)] / interaction.kwargs["probability"]
                interaction_info = {'reward': ratio*interaction.kwargs['reward']}

            for k,v in interaction.kwargs.items():
                if k not in ["probability", "actions", "reward"]:
                    interaction_info[k] = v

            reveal = interaction.kwargs.get("reveal", interaction.kwargs.get("reward"))
            prob   = interaction.kwargs.get("probability")
            
            start_time = time.time()
            learner.learn(interaction.context, interaction.action, reveal, prob, info)
            learn_time = time.time()-start_time

            learner_info  = { k:v for item in LearnerContext.logger.read() for k,v in item.items() }
            time_info     = {"predict_time": predict_time, "learn_time": learn_time} if self._time else {}

            yield {**interaction_info, **learner_info, **time_info}

class OnlineWarmStartEvalTask(EvaluationTask):
    """Evaluate a Learner on a WarmStartEnvironment."""
    
    def __init__(self, time:bool = True, seed:int = 1) -> None:
        """Instantiate an OnlineOnPolicyEvalTask.

        Args:
            seed: A random seed which determines which action is taken given predictions.        
        """        
        self._seed = seed
        self._time = time

    def process(self, learner: Learner, interactions: Iterable[Interaction]) -> Iterable[Dict[Any,Any]]:

        if not isinstance(learner, SafeLearner): learner = SafeLearner(learner)
        if not interactions: return

        separable_interactions = iter(self._repeat_first_simulated_interaction(interactions))

        logged_interactions    = takewhile(lambda i: isinstance(i,LoggedInteraction), separable_interactions)
        simulated_interactions = separable_interactions

        for row in OnlineOffPolicyEvalTask(self._time).process(learner, logged_interactions):
            yield row

        for row in OnlineOnPolicyEvalTask(self._time, self._seed).process(learner, simulated_interactions):
            yield row

    def _repeat_first_simulated_interaction(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        any_simulated_found = False

        for interaction in interactions:
            yield interaction

            if isinstance(interaction, SimulatedInteraction) and not any_simulated_found:
                yield interaction

            any_simulated_found = any_simulated_found or isinstance(interaction, SimulatedInteraction)

class SimpleLearnerTask(LearnerTask):
    """Describe a Learner using its name and hyperparameters."""

    def process(self, item: Learner) -> Dict[Any,Any]:
        item = SafeLearner(item)
        return {"full_name": item.full_name, **item.params}

class SimpleEnvironmentTask(EnvironmentTask):
    """Describe an Environment using its Environment and Filter parameters."""

    def process(self, environment:Environment, interactions: Iterable[Interaction]) -> Dict[Any,Any]:
            return {"type": self._source_name(environment), **environment.params }

    def _source_name(self, env) -> str:
        base_env = env._source if isinstance(env, FilteredEnvironment) else env
        return type(base_env).__name__

class ClassEnvironmentTask(EnvironmentTask):
    """Describe an Environment made from a Classification dataset.
    
    In addition to the Environment's parameters this task also calculates a number
    of classification statistics which can be used to analyze the performance of learners
    after an Experiment has finished. To make the most of this Task sklearn should be installed.
    """

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
