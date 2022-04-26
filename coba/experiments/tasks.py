import time
import collections
import collections.abc
import math

from abc import ABC, abstractmethod
from statistics import median, mean
from itertools import takewhile, chain, compress
from typing import Iterable, Any, Dict, Sequence, Hashable

from coba.exceptions import CobaExit
from coba.random import CobaRandom
from coba.learners import Learner, SafeLearner
from coba.encodings import InteractionsEncoder
from coba.utilities import PackageChecker
from coba.contexts import InteractionContext
from coba.environments import Environment, Interaction, SimulatedInteraction, LoggedInteraction, SafeEnvironment

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

class OnlineOnPolicyEvalTask(EvaluationTask):
    """Evaluate a Learner on a SimulatedEnvironment."""

    def __init__(self, time:bool = True, seed:int = 1) -> None:
        """Instantiate an OnlineOnPolicyEvalTask.

        Args:
            seed: A random seed which determines which action is taken given predictions.

        """
        self._time = time
        self._seed = seed

    def process(self, learner: Learner, interactions: Iterable[SimulatedInteraction]) -> Iterable[Dict[Any,Any]]:

        random = CobaRandom(self._seed)

        if not isinstance(learner, SafeLearner): learner = SafeLearner(learner)
        if not interactions: return

        for interaction in interactions:

            InteractionContext.learner_info.clear()

            context = interaction.context
            actions = interaction.actions

            start_time         = time.time()
            probabilities,info = learner.predict(context, actions)
            predict_time       = time.time() - start_time

            action       = random.choice(actions, probabilities)
            action_index = actions.index(action)
            reward       = interaction.rewards[action_index]
            probability  = probabilities[action_index]

            start_time = time.time()
            learner.learn(context, action, reward, probability, info)
            learn_time = time.time() - start_time

            learner_info     = InteractionContext.learner_info
            interaction_info = { k:v[action_index] if isinstance(v,(list,tuple)) else v for k,v in interaction.kwargs.items() }

            evaluation_info  = {
                'reward'      : reward,
                'max_reward'  : max(interaction.rewards),
                'min_reward'  : min(interaction.rewards),
                'rank'        : 1+sorted(interaction.rewards,reverse=True).index(reward),
                'n_actions'   : len(interaction.actions),
            }

            if self._time:
                evaluation_info["predict_time"] = predict_time
                evaluation_info["learn_time"  ] = learn_time

            yield {**interaction_info, **learner_info, **evaluation_info}

class OnlineOffPolicyEvalTask(EvaluationTask):
    """Evaluate a Learner on a LoggedEnvironment."""

    def __init__(self, time:bool = True) -> None:
        self._time = time

    def process(self, learner: Learner, interactions: Iterable[LoggedInteraction]) -> Iterable[Dict[Any,Any]]:

        learn_time = 0
        predict_time = 0

        if not isinstance(learner, SafeLearner): learner = SafeLearner(learner)
        if not interactions: return

        for interaction in interactions:

            InteractionContext.learner_info.clear()

            if not interaction.actions:
                info             = None
                interaction_info = {}

            if interaction.actions:
                actions          = list(interaction.actions)
                start_time       = time.time()
                probs,info       = learner.predict(interaction.context, actions)
                predict_time     = time.time()-start_time
                interaction_info = {}

            if interaction.probability and interaction.actions:
                ratio            = probs[actions.index(interaction.action)] / interaction.probability
                interaction_info = {'reward': ratio*interaction.reward}

            for k,v in interaction.kwargs.items():
                interaction_info[k] = v

            reveal = interaction.reward
            prob   = interaction.probability

            start_time = time.time()
            learner.learn(interaction.context, interaction.action, reveal, prob, info)
            learn_time = time.time()-start_time

            learner_info  = InteractionContext.learner_info
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
        if not isinstance(environment, SafeEnvironment): environment = SafeEnvironment(environment)
        return { k:v for k,v in environment.params.items() if v is not None }

class ClassEnvironmentTask(EnvironmentTask):
    """Describe an Environment made from a Classification dataset.

    In addition to the Environment's parameters this task calculates a number of classification 
    statistics which can be used to analyze the performance of learners after an Experiment has 
    finished. To make the most of this Task sklearn should be installed.
    """

    def process(self, environment: Environment, interactions: Iterable[SimulatedInteraction]) -> Dict[Any,Any]:

        #sources:
        #[1]: https://arxiv.org/pdf/1808.03591.pdf
        #[2]: https://link.springer.com/content/pdf/10.1007/978-3-540-31883-5.pdf#page=468
        #[3]: https://link.springer.com/content/pdf/10.1007/s10044-012-0280-z.pdf
        #[4]: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.440.6255&rep=rep1&type=pdf

        #[3] found that information theoretic measures and landmarking measures are most important

        contexts,actions,rewards = zip(*[ (i.context, i.actions, i.rewards) for i in interactions ])
        env_stats = {}

        X = [ InteractionsEncoder('x').encode(x=c) for c in contexts ]
        Y = [ a[r.index(max(r))] for a,r in zip(actions,rewards)]

        is_dense = not isinstance(X[0],dict)

        feats          = list(range(len(X[0]))) if is_dense else set().union(*X)
        class_counts   = collections.Counter(Y).values()
        feature_counts = collections.Counter(chain(*[feats if is_dense else x.keys() for x in X])).values()

        n = len(X)
        k = len(class_counts)
        m = len(feature_counts)

        X_by_f = { f:[x[f] for x in X if is_dense or f in x] for f in feats}
        x_bin  = lambda x,f:int(n/5*(x[f]-min(X_by_f[f]))/((max(X_by_f[f])-min(X_by_f[f])) or 1))
        X_bin  = [ [x_bin(x,f) for f in feats] if is_dense else { f:x_bin(x,f) for f in x.keys() } for x in X]

        get = lambda x,f: x[f] if is_dense else x.get(f,0)

        #Information-Theoretic Meta-features

        env_stats["class_count"          ] = k
        env_stats["class_entropy_normed" ] = self._entropy_normed(Y)  # [1,2,3]
        env_stats["class_imbalance_ratio"] = self._imbalance_ratio(Y) # [1]
        env_stats["joint_XY_entropy_mean"] = mean([self._entropy([(get(x,f),y) for x,y in zip(X_bin,Y)]) for f in feats]) #[2,3]
        env_stats["mutual_XY_info_mean"  ] = mean([self._mutual_info([get(x,f) for x in X],Y) for f in feats]) #[2,3]
        env_stats["equivalent_num_attr"  ] = self._entropy(Y)/env_stats["mutual_XY_info_mean"] #[2,3]

        #Sparsity/Dimensionality measures [1,2,3]
        env_stats["feature_count"       ] = m
        env_stats["feature_per_instance"] = median([len(x)/m for x in X])
        env_stats["instance_per_feature"] = median([f/m for f in feature_counts])

        try:

            PackageChecker.sklearn("ClassEnvironmentTask.process")

            import numpy as np
            import scipy.sparse as sp

            from sklearn.feature_extraction import FeatureHasher
            from sklearn.decomposition import TruncatedSVD
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.naive_bayes import GaussianNB
            from sklearn.metrics import f1_score, accuracy_score
            from sklearn.model_selection import cross_validate
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.tree import DecisionTreeClassifier

            np_X = np.array(X) if is_dense else FeatureHasher(n_features=2**14, input_type="dict").fit_transform(X)
            np_Y = np.array(Y)

            try:
                #1NN OOB [3,4]
                oob = np_Y[KNeighborsClassifier(n_neighbors=1).fit(np_X,np_Y).kneighbors(np_X, n_neighbors=2, return_distance=False)[:,1]]
                env_stats["1nn_accuracy"] = accuracy_score(np_Y,oob)
                env_stats["1nn_f1_weighted"] = f1_score(np_Y,oob, average='weighted')
            except: #pragma: no cover
                pass

            try:
                #LDA [3,4]
                scr = cross_validate(LinearDiscriminantAnalysis(), np_X, np_Y, scoring=('accuracy','f1_weighted'))
                env_stats["lda_accuracy"] = scr['test_accuracy']
                env_stats["lda_f1_weighted"] = scr['test_f1_weighted']
            except: #pragma: no cover
                pass

            try:
                #Naive Bayes [3,4]
                scr = cross_validate(GaussianNB(), np_X, np_Y, scoring=('accuracy','f1_weighted'))
                env_stats["naive_bayes_accuracy"] = scr['test_accuracy']
                env_stats["naive_bayes_f1_weighted"] = scr['test_f1_weighted']
            except: #pragma: no cover
                pass

            try:
                #Average Node Learner [3,4]
                scr = cross_validate(RandomForestClassifier(n_estimators=100,criterion='entropy',max_depth=1), np_X, np_Y, scoring=('accuracy','f1_weighted'))
                env_stats["average_node_accuracy"] = scr['test_accuracy']
                env_stats["average_node_f1_weighted"] = scr['test_f1_weighted']
            except: #pragma: no cover
                pass

            try:
                #Best Node Learner [3,4]
                scr = cross_validate(DecisionTreeClassifier(criterion='entropy',max_depth=1), np_X, np_Y, scoring=('accuracy','f1_weighted'))
                env_stats["best_node_accuracy"] = scr['test_accuracy']
                env_stats["best_node_f1_weighted"] = scr['test_f1_weighted']
            except: #pragma: no cover
                pass

            try:
                #pca effective dimensions [1]
                cnt_X = np_X - np_X.mean(axis=0) if is_dense else sp.vstack([sp.csr_matrix(np_X.mean(axis=0))]*np_X.shape[0])
                pca_var = TruncatedSVD(n_components=min(cnt_X.shape[1]-1,10)).fit(cnt_X).explained_variance_ratio_
                env_stats["pca_dims_95"] = (pca_var<.95).sum()+1
            except: #pragma: no cover
                pass


            #sklearn's CCA doesn't seem to work with sparse so I'm leaving it out for now depsite [3]

        except CobaExit:
            pass

        return { **SimpleEnvironmentTask().process(environment, interactions), **env_stats }

    def _entropy(self, items: Sequence[Hashable]) -> float:
        return -sum([count/len(items)*math.log2(count/len(items)) for count in collections.Counter(items).values()])

    def _entropy_normed(self, items: Sequence[Hashable]) -> float:
        return self._entropy(items)/math.log2(len(set(items)))

    def _mutual_info(self, items1: Sequence[Hashable], items2: Sequence[Hashable]) -> float:
        return self._entropy(items1) + self._entropy(items2) - self._entropy(list(zip(items1,items2)))

    def _imbalance_ratio(self, items: list) -> float:
        counts = collections.Counter(items).values()
        n      = len(items)
        k      = len(counts)
        return (k-1)/k*sum([c/(n-c) for c in counts])