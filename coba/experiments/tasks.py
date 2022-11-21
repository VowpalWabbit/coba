import time
import collections
import collections.abc
import math
import warnings

from abc import ABC, abstractmethod
from statistics import mean
from itertools import combinations
from typing import Iterable, Any, Sequence, Mapping, Hashable, Union

from coba.statistics import percentile
from coba.exceptions import CobaExit, CobaException
from coba.learners import Learner, SafeLearner
from coba.encodings import InteractionsEncoder
from coba.utilities import PackageChecker, peek_first
from coba.contexts import CobaContext
from coba.environments import Environment, SafeEnvironment
from coba.environments import Interaction, SimulatedInteraction, LoggedInteraction, GroundedInteraction

class LearnerTask(ABC):
    """A task which describes a Learner."""

    @abstractmethod
    def process(self, learner: Learner) -> Mapping[Any,Any]:
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
    def process(self, environment: Environment, interactions: Iterable[Interaction]) -> Mapping[Any,Any]:
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
    def process(self, learner: Learner, interactions: Iterable[Interaction]) -> Iterable[Mapping[Any,Any]]:
        """Process the EvaluationTask.

        Args:
            learner: The Learner that we wish to evaluate.
            interactions: The Interactions which we wish to evaluate on.

        Returns:
            An iterable of evaluation statistics (for online evaluation there should be one dict per interaction).
        """
        ...

class SimpleEvaluation(EvaluationTask):

    def __init__(self, 
        reward_metrics: Union[str,Sequence[str]] = "reward", 
        time_metrics: bool = False,
        probability: bool = False) -> None:
        """
        Args:
            reward_metrics: The reward metrics to that should be recorded for each example. 
                The metric options include: reward, rank, regret.
            time_metrics: Indicates whether time metrics should be recorded.
            probability: Indicates whether the action's probability should be recorded.
        """

        if not reward_metrics:
            raise CobaException("At least one performance metric is required.")

        if isinstance(reward_metrics,str): 
            reward_metrics = [reward_metrics]

        self._metrics = reward_metrics
        self._time    = time_metrics
        self._prob    = probability

    def process(self, learner: Learner, interactions: Iterable[Interaction]) -> Iterable[Mapping[Any,Any]]:

        learner = SafeLearner(learner)

        first, interactions = peek_first(interactions)

        predict  = learner.predict
        learn    = learner.learn

        discrete = (first and first.is_discrete)

        learning_info = CobaContext.learning_info

        if 'rank' in self._metrics and not discrete:
            warnings.warn(f"The rank metric can only be calculated for discrete environments")

        calc_rank = 'rank' in self._metrics and discrete
        calc_reward = 'reward' in self._metrics
        calc_regret = 'regret' in self._metrics

        for interaction in interactions:
            
            learning_info.clear()
            out = {}

            if isinstance(interaction, SimulatedInteraction):

                context = interaction.context
                actions = interaction.actions
                rewards = interaction.rewards

                start_time       = time.time()
                action,prob,info = predict(context, actions)
                predict_time     = time.time()-start_time

                reward = rewards.eval(action)

                start_time = time.time()
                learn(context, actions, action, reward, prob, **info)
                learn_time = time.time() - start_time

            elif isinstance(interaction, LoggedInteraction):
                context = interaction.context
                actions = interaction.actions
                rewards = interaction.rewards

                if not actions or not rewards:
                    predict_time = None
                    reward       = None
                    prob         = None
                    info         = None
                else:
                    start_time   = time.time()
                    action,prob  = predict(context, actions)[:2]
                    reward       = rewards.eval(action)
                    predict_time = time.time()-start_time

                start_time = time.time()
                learn(context, actions, interaction.action, interaction.reward, interaction.probability,info={})
                learn_time = time.time()-start_time

            elif isinstance(interaction, GroundedInteraction):
                context   = interaction.context
                actions   = interaction.actions
                rewards   = interaction.rewards
                feedbacks = interaction.feedbacks

                start_time       = time.time()
                action,prob,info = predict(context, actions)
                reward           = rewards.eval(action)
                feedback         = feedbacks.eval(action)
                predict_time     = time.time()-start_time

                start_time = time.time()
                learn(context, actions, action, feedback, prob, **info)
                learn_time = time.time()-start_time

                out['feedback'] = feedback

            else:
                raise CobaException("An unknown interaction type was received.")

            if calc_reward and reward is not None: out['reward'] = reward
            if calc_rank  : out['rank'  ] = sorted(map(rewards.eval,actions)).index(reward)/(len(actions)-1)
            if calc_regret: out['regret'] = rewards.max()-reward

            if self._time: out.update(predict_time=predict_time, learn_time=learn_time)
            if self._prob and prob is not None: out.update(probability=prob)

            out.update(interaction.kwargs)
            out.update(learning_info)

            yield out
            learning_info.clear()

class SimpleLearnerInfo(LearnerTask):
    """Describe a Learner using its name and hyperparameters."""

    def process(self, item: Learner) -> Mapping[Any,Any]:
        item = SafeLearner(item)
        return {"full_name": item.full_name, **item.params}

class SimpleEnvironmentInfo(EnvironmentTask):
    """Describe an Environment using its Environment and Filter parameters."""

    def process(self, environment:Environment, interactions: Iterable[Interaction]) -> Mapping[Any,Any]:

        return { k:v for k,v in SafeEnvironment(environment).params.items() if v is not None }

class ClassEnvironmentInfo(EnvironmentTask):
    """Describe an Environment made from a Classification dataset.

    In addition to the Environment's parameters this task calculates a number of classification
    statistics which can be used to analyze the performance of learners after an Experiment has
    finished. To make the most of this Task sklearn should be installed.
    """

    def process(self, environment: Environment, interactions: Iterable[SimulatedInteraction]) -> Mapping[Any,Any]:

        #sources:
        #[1]: https://arxiv.org/pdf/1808.03591.pdf (lorena2019complex)
        #[2]: https://link.springer.com/content/pdf/10.1007/978-3-540-31883-5.pdf#page=468 (castiello2005meta)
        #[3]: https://link.springer.com/content/pdf/10.1007/s10044-012-0280-z.pdf (reif2014automatic)
        #[4]: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.440.6255&rep=rep1&type=pdf

        #[3] found that information theoretic measures and landmarking measures are most important

        contexts,actions,rewards = zip(*[ (i.context, i.actions, i.rewards) for i in interactions ])
        env_stats = {}

        X = [ InteractionsEncoder('x').encode(x=c) for c in contexts ]
        Y = [ r.argmax()                           for r in rewards  ]
        X = self._dense(X)

        classes = list(set(Y))
        feats   = list(range(len(X[0])))

        n = len(X)
        m = len(feats)
        k = len(classes)

        X_bin = self._bin(X,n/10)
        X_bin_by_f = list(zip(*X_bin))

        entropy_Y = self._entropy(Y)
        entropy_X = [self._entropy(x) for x in X_bin_by_f]

        mutual_XY_infos  = sorted([self._mutual_info(x,Y) for x in X_bin_by_f], reverse=True)

        #Information-Theoretic Meta-features
        env_stats["class_count"          ] = k
        env_stats["class_entropy"        ] = entropy_Y                # [1]
        env_stats["class_entropy_N"      ] = self._entropy_normed(Y)  # [1,2,3]
        env_stats["class_imbalance_ratio"] = self._imbalance_ratio(Y) # [1]
 
        env_stats["feature_numeric_count"  ] = sum([int(len(set(X_bin_by_f[f]))!=2) for f in feats])
        env_stats["feature_onehot_count"   ] = sum([int(len(set(X_bin_by_f[f]))==2) for f in feats])
        env_stats["feature_entropy_mean"   ] = mean([self._entropy(X_bin_by_f[f]) for f in feats])
        env_stats["feature_entropy_mean_N" ] = mean([self._entropy_normed(X_bin_by_f[f]) for f in feats])
        env_stats["joint_XY_entropy_mean"  ] = mean([self._entropy(list(zip(X_bin_by_f[f],Y))) for f in feats]) #[2,3]
        env_stats["joint_XY_entropy_mean_N"] = mean([self._entropy_normed(list(zip(X_bin_by_f[f],Y))) for f in feats])
        env_stats["mutual_XY_info_mean"    ] = mean(mutual_XY_infos) #[2,3]
        env_stats["mutual_XY_info_mean_N"  ] = mean(mutual_XY_infos)/entropy_Y if entropy_Y else None #[2,3]

        env_stats["mutual_XY_info_rank1"   ] = mutual_XY_infos[0]
        env_stats["mutual_XY_info_rank2"   ] = mutual_XY_infos[1] if len(mutual_XY_infos) > 1 else None
        env_stats["equivalent_num_X_attr"  ] = entropy_Y/mean(mutual_XY_infos) if mean(mutual_XY_infos) else None #[2,3]
        env_stats["noise_signal_ratio"     ] = (mean(entropy_X)-mean(mutual_XY_infos))/mean(mutual_XY_infos) if mean(mutual_XY_infos) else None #[2] 

        env_stats["max_fisher_discrim"    ] = self._max_fisher_discriminant_ratio(X, Y) #[1]
        #env_stats["max_fisher_discrim_dir"] = self._max_directional_fisher_discriminant_ratio(X, Y) #[1] (this dies on large feature)
        env_stats["volume_overlapping"    ] = self._volume_overlapping_region(X,Y) #[1]
        env_stats["max_single_feature_eff"] = self._max_individual_feature_efficiency(X,Y) #[1]

        #Sparsity/Dimensionality measures [1,2,3]
        env_stats["feature_count"       ] = m
        env_stats["percent_nonzero_feat"] = mean([len(list(filter(None,x)))/len(x) for x in X])

        try:

            PackageChecker.sklearn("ClassEnvironmentTask.process")

            import numpy as np

            from sklearn.decomposition import TruncatedSVD
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.naive_bayes import GaussianNB
            from sklearn.metrics import f1_score, accuracy_score
            from sklearn.model_selection import cross_validate
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.tree import DecisionTreeClassifier

            import sklearn.exceptions
            warnings.filterwarnings("ignore", category=sklearn.exceptions.FitFailedWarning)

            np_X = np.array(X)
            np_Y = np.array(Y)

            try:
                #1NN OOB [3,4]
                oob = np_Y[KNeighborsClassifier(n_neighbors=1).fit(np_X,np_Y).kneighbors(np_X, n_neighbors=2, return_distance=False)[:,1]]
                env_stats["1nn_accuracy"] = float(accuracy_score(np_Y,oob))
                env_stats["1nn_f1_macro"] = float(f1_score(np_Y,oob, average='macro'))
            except: #pragma: no cover
                pass

            try:
                #LDA [3,4]
                scr = cross_validate(LinearDiscriminantAnalysis(), np_X, np_Y, scoring=('accuracy','f1_macro'))
                env_stats["lda_accuracy"] = float(mean(scr['test_accuracy']))
                env_stats["lda_f1_weighted"] = float(mean(scr['test_f1_weighted']))
            except: #pragma: no cover
                pass

            try:
                #Naive Bayes [3,4]
                scr = cross_validate(GaussianNB(), np_X, np_Y, scoring=('accuracy','f1_macro'))
                env_stats["naive_bayes_accuracy"] = float(mean(scr['test_accuracy']))
                env_stats["naive_bayes_f1_weighted"] = float(mean(scr['test_f1_macro']))
            except: #pragma: no cover
                pass

            try:
                #Average Node Learner [3,4]
                scr = cross_validate(RandomForestClassifier(n_estimators=50,criterion='entropy',max_depth=1), np_X, np_Y, scoring=('accuracy','f1_macro'))
                env_stats["average_node_accuracy"] = float(mean(scr['test_accuracy']))
                env_stats["average_node_f1_macro"] = float(mean(scr['test_f1_weighted']))
            except: #pragma: no cover
                pass

            try:
                #Best Node Learner [3,4]
                scr = cross_validate(DecisionTreeClassifier(criterion='entropy',max_depth=1), np_X, np_Y, scoring=('accuracy','f1_macro'))
                env_stats["best_node_accuracy"] = float(mean(scr['test_accuracy']))
                env_stats["best_node_f1_macro"] = float(mean(scr['test_f1_macro']))
            except: #pragma: no cover
                pass

            try:
                #pca effective dimensions [1]
                centered_x= np_X - np_X.mean(axis=0)
                pca_var = TruncatedSVD(n_components=min(np_X.shape[1]-1,1000)).fit(centered_x).explained_variance_ratio_
                env_stats["pca_top_1_pct"] = pca_var[0:1].sum()
                env_stats["pca_top_2_pct"] = pca_var[0:2].sum()
                env_stats["pca_top_3_pct"] = pca_var[0:3].sum()
                env_stats["pca_dims_95"  ] = int(sum(np.cumsum(pca_var)<.95)+1)
            except: #pragma: no cover
                pass

            #sklearn's CCA doesn't seem to work with sparse so I'm leaving it out for now depsite [3]

        except CobaExit:
            pass

        return { **SimpleEnvironmentInfo().process(environment, interactions), **env_stats }

    def _entropy(self, items: Sequence[Hashable]) -> float:
        return -sum([count/len(items)*math.log2(count/len(items)) for count in collections.Counter(items).values()])

    def _entropy_normed(self, items: Sequence[Hashable]) -> float:
        return self._entropy(items)/(math.log2(len(set(items))) or 1)

    def _mutual_info(self, items1: Sequence[Hashable], items2: Sequence[Hashable]) -> float:
        return self._entropy(items1) + self._entropy(items2) - self._entropy(list(zip(items1,items2)))

    def _imbalance_ratio(self, items: list) -> float:
        #Equation (37) and (38) in [1]

        counts = collections.Counter(items).values()
        n      = len(items)
        k      = len(counts)

        if n == 0: return None
        if max(counts) == n: return 1

        IR = (k-1)/k*sum([c/(n-c) for c in counts])
        return 1 - 1/IR
    
    def _max_fisher_discriminant_ratio(self, X, Y) -> float:
        #Equation (3) in [1]
        #needs more testing

        feats    = list(range(len(X[0])))
        Y_set    = set(Y)
        Y_counts = collections.Counter(Y)

        X_by_f   = collections.defaultdict(list)
        X_by_fy  = collections.defaultdict(lambda: collections.defaultdict(list))

        for x,y in zip(X,Y):
            for f in feats:
                xf = x[f]
                X_by_fy[f][y].append(xf)
                X_by_f[f].append(xf)

        mean_f  = { f:      mean(X_by_f[f])                      for f in feats}
        mean_fy = { f: { y: mean(X_by_fy[f][y]) for y in Y_set } for f in feats} 

        max_ratio = 0

        for f in feats:
            ratio_numer = sum([Y_counts[y]*(mean_fy[f][y]-mean_f[f])**2 for y in Y_set])
            ratio_denom = sum([(X_by_fy[f][y][i]-mean_fy[f][y])**2 for y in Y_set for i in range(Y_counts[y]) ])
            if ratio_denom != 0:
                max_ratio   = max(max_ratio,ratio_numer/ratio_denom)

        return 1/(1+max_ratio)

    def _max_directional_fisher_discriminant_ratio(self, X, Y) -> float:
        #equation (4) in [1]
        #this code is currently not used because it can take an incredibly 
        #long time to calculate due to np.linalg.inv so we "no cover" it

        try:
            PackageChecker.sklearn('')

            import numpy as np
            from sklearn.covariance import shrunk_covariance

            Y_set = set(Y)
            X     = np.array(X).T #transpose so that equations align with paper
            Y     = np.array(Y)

            X_by_y = { y: X[:,Y==y]  for y in Y_set }
            OVO    = []

            for y1,y2 in combinations(Y_set,2):
                mu_c1 = X_by_y[y1].mean(axis=1).reshape(-1,1)
                mu_c2 = X_by_y[y2].mean(axis=1).reshape(-1,1)

                p1 = X_by_y[y1].shape[1]/(X_by_y[y1].shape[1]+X_by_y[y2].shape[1])
                p2 = X_by_y[y2].shape[1]/(X_by_y[y1].shape[1]+X_by_y[y2].shape[1])

                s1 = (X_by_y[y1]-mu_c1) @ (X_by_y[y1]-mu_c1).T
                s2 = (X_by_y[y2]-mu_c2) @ (X_by_y[y2]-mu_c2).T
                B  = (mu_c1-mu_c2) @ (mu_c1-mu_c2).T 

                s1 = shrunk_covariance(s1)
                s2 = shrunk_covariance(s2)
                B  = shrunk_covariance(B)

                W = p1*s1+p2*s2
                d = np.linalg.inv(W) @ (mu_c1-mu_c2)

                OVO.append( ((d.T@B@d) / (d.T@W@d))[0,0] )

            return 1/(1+mean(OVO)) if OVO else None

        except np.linalg.LinAlgError:#pragma: no cover
            return None
        except CobaExit:#pragma: no cover
            return None

    def _volume_overlapping_region(self, X, Y) -> float:
        #equation (9) in [1]

        X_by_y = collections.defaultdict(list)
        for x,y in zip(X,Y): X_by_y[y].append(x)
        F_by_y = { y: list(zip(*x)) for y,x in X_by_y.items()}

        minmax = lambda f,y1,y2: min(max(F_by_y[y1][f]), max(F_by_y[y2][f]))
        maxmin = lambda f,y1,y2: max(min(F_by_y[y1][f]), min(F_by_y[y2][f]))
        maxmax = lambda f,y1,y2: max(max(F_by_y[y1][f]), max(F_by_y[y2][f]))
        minmin = lambda f,y1,y2: min(min(F_by_y[y1][f]), min(F_by_y[y2][f]))

        OVO = []

        for y1,y2 in combinations(set(Y),2):
            F2 = 1
            for f in range(len(X[0])):
                if maxmax(f,y1,y2)-minmin(f,y1,y2) != 0:
                    F2 *= max(0, (minmax(f,y1,y2)-maxmin(f,y1,y2)))/(maxmax(f,y1,y2)-minmin(f,y1,y2))

            OVO.append(F2)

        return mean(OVO) if OVO else None

    def _max_individual_feature_efficiency(self, X, Y) -> float:
        #equation (11) in [1]

        try:
            PackageChecker.numpy('ClassEnvironmentTask')
            import numpy as np

            X_by_f = np.array(list(zip(*X)))
            X_by_y = collections.defaultdict(list)
            for x,y in zip(X,Y): X_by_y[y].append(x)
            F_by_y  = { y: list(zip(*x)) for y,x in X_by_y.items()}
            L_by_fy = { f: { y:percentile(F_by_y[y][f],[0.0,1.0])  for y in F_by_y } for f in range(len(X[0])) }

            minmax = lambda f,y1,y2: min(L_by_fy[f][y1][1], L_by_fy[f][y2][1])
            maxmin = lambda f,y1,y2: max(L_by_fy[f][y1][0], L_by_fy[f][y2][0])

            OVO    = []

            for y1,y2 in combinations(set(Y),2):

                n_o = []
                for f in range(len(X[0])):
                    n_o.append(int(((X_by_f[f] <= minmax(f,y1,y2)) & (maxmin(f,y1,y2) <= X_by_f[f])).sum()))

                OVO.append(min(n_o)/len(X))

            return mean(OVO) if OVO else None

        except CobaExit:
            return None

    def _dense(self, X) -> Sequence[Sequence[float]]:

        if not isinstance(X[0],dict):
            return X
        else:
            #Convert the sparse dicts into a compact dense array.
            #This is required by for a number of the analysis in this task.
            feats = sorted(set().union(*X)) #sort to make unit tests repeatable
            dense_X = [ [0]*len(feats) for _ in range(len(X)) ]

            for i,x in enumerate(X):
                for j,f in enumerate(feats):
                        dense_X[i][j] = x.get(f,0)

            return dense_X

    def _bin(self, X: Sequence[Sequence[float]], n_bins:int, lower:float=0.05, upper:float=0.95) -> Sequence[Sequence[float]]:
        X_by_f = list(zip(*X))
        lim_f  = { f: percentile(X_by_f[f],[lower,upper]) for f in range(len(X_by_f)) }
        X_bin  = [ [ round((n_bins-1)*(x[f]-lim_f[f][0])/((lim_f[f][1]-lim_f[f][0]) or 1)) for f in range(len(X_by_f)) ] for x in X]
        return X_bin
