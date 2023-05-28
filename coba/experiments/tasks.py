import collections
import collections.abc
import time
import warnings

from abc import ABC, abstractmethod
from itertools import combinations
from operator import __mul__
from statistics import mean
from typing import Iterable, Any, Sequence, Mapping, Hashable, Callable
from bisect import insort

import math

from coba.exceptions import CobaException
from coba.random import CobaRandom
from coba.backports import Literal
from coba.contexts import CobaContext
from coba.encodings import InteractionsEncoder
from coba.environments import Environment, SafeEnvironment, Interaction
from coba.exceptions import CobaExit
from coba.learners import Learner, SafeLearner
from coba.primitives import Batch
from coba.statistics import percentile, weighted_percentile
from coba.utilities import PackageChecker, peek_first

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
    def process(self, learner: Learner, interactions: Iterable[Interaction], seed:float = None) -> Iterable[Mapping[Any,Any]]:
        """Process the EvaluationTask.

        Args:
            learner: The Learner that we wish to evaluate.
            interactions: The Interactions which we wish to evaluate on.
            seed: An optional argument to seed randomness during evaluation.

        Returns:
            An iterable of evaluation results (e.g., for online evaluation there should be one dict per interaction).
        """
        ...

class SimpleEvaluation(EvaluationTask):

    def __init__(self, 
        record: Sequence[Literal['reward','rank','regret','time','probability','action','actions', 'context', 'ope_loss', 'rewards']] = ['reward'],
        learn: bool = True,
        predict: bool = True,
        seed: float = None) -> None:
        """
        Args:
            record: The datapoints to record for each interaction.
            learn: Indicates whether learning should occur as part of the evaluation task.
            evals: Indicates whether evaluation should occur as part of the evaluation task.
            seed: Provide an explicit seed to use during evaluation. If not provided a default is used.
        """

        self._record = [record] if isinstance(record,str) else record
        self._learn  = learn
        self._evals  = predict
        self._seed   = seed

    def process(self, learner: Learner, interactions: Iterable[Interaction]) -> Iterable[Mapping[Any,Any]]:

        first, interactions = peek_first(interactions)

        if 'reward' in first and 'action' in first:
            yield from OffPolicyEvaluation(self._record, self._learn, self._evals, self._seed).process(learner, interactions)
        else:
            yield from OnPolicyEvaluation(self._record, self._learn, self._seed).process(learner, interactions)

class OnPolicyEvaluation(EvaluationTask):

    IMPLICIT_EXCLUDE = {"actions", "rewards", "action", "reward", "probability"}
    ONLY_DISCRETE    = {'rank', 'rewards'}

    def __init__(self, 
        record: Sequence[Literal['reward','rank','regret','time','probability','action','context','actions','rewards']] = ['reward'],
        learn: bool = True,
        seed: float = None) -> None:
        """
        Args:
            record: The datapoints to record for each interaction.
            learn: Indicates if learning should occur during the evaluation process.
            seed: Provide an explicit seed to use during evaluation. If not provided a default is used.
        """

        self._record = [record] if isinstance(record,str) else record
        self._learn  = learn
        self._seed   = seed

    def process(self, learner: Learner, interactions: Iterable[Interaction]) -> Iterable[Mapping[Any,Any]]:

        learner = SafeLearner(learner, self._seed if self._seed is not None else CobaContext.store.get("experiment_seed"))

        first, interactions = peek_first(interactions)        
        batched  = first and isinstance(first['context'], Batch)

        for key in ['rewards','context','actions']:
            if key not in first:
                raise CobaException(f"OnPolicyEvaluation requires every interaction to have '{key}'")

        for key in ['rewards','actions']:
            if (first[key][0] if batched else first[key]) is None:
                raise CobaException(f"OnPolicyEvaluation requires every interaction to have a not None '{key}'")

        discrete = len(first['actions'][0] if batched else first['actions']) > 0
        if not discrete:
            for metric in set(self._record).intersection(OnPolicyEvaluation.ONLY_DISCRETE):
                warnings.warn(f"The {metric} metric can only be calculated for discrete environments")

        record_prob     = 'probability' in self._record
        record_time     = 'time'        in self._record
        record_action   = 'action'      in self._record
        record_context  = 'context'     in self._record
        record_actions  = 'actions'     in self._record
        record_rewards  = 'rewards'     in self._record and discrete
        record_rank     = 'rank'        in self._record and discrete
        record_reward   = 'reward'      in self._record
        record_regret   = 'regret'      in self._record
        record_ope_loss = 'ope_loss'    in self._record

        get_reward = lambda reward                  : reward
        get_regret = lambda reward, rewards         : rewards.max()-reward
        get_rank   = lambda reward, rewards, actions: sorted(map(rewards.eval,actions)).index(reward)/(len(actions)-1)

        get_reward_list  = lambda rewards,actions: list(map(rewards.eval, actions))

        predict = learner.predict
        learn   = learner.learn
        info    = CobaContext.learning_info

        info.clear()

        for interaction in interactions:

            out = {}
            interaction = interaction.copy()

            context   = interaction.pop('context')
            actions   = interaction.pop('actions')
            rewards   = interaction.pop('rewards')
            feedbacks = interaction.pop('feedbacks',None)

            batched  = isinstance(context, Batch)
            discrete = len(actions[0] if batched else actions) > 0

            if record_context: out['context'] = context
            if record_actions: out['actions'] = actions
            if record_rewards: out['rewards'] = list(map(get_reward_list,rewards,actions)) if batched else get_reward_list(rewards,actions)

            start_time         = time.time()
            action,prob,kwargs = predict(context, actions)
            predict_time       = time.time()-start_time

            reward   = rewards.eval(action)
            feedback = feedbacks.eval(action) if feedbacks else None

            start_time = time.time()
            if self._learn: learn(context, actions, action, feedback if feedbacks else reward, prob, **kwargs)
            learn_time = time.time() - start_time

            if record_time    : out['predict_time'] = predict_time
            if record_time    : out['learn_time']   = learn_time
            if record_prob    : out['probability']  = prob
            if record_action  : out['action']       = action
            if feedbacks      : out['feedback']     = feedback
            if record_reward  : out['reward']       = list(map(get_reward,reward)) if batched else get_reward(reward)
            if record_regret  : out['regret']       = list(map(get_regret,reward,rewards)) if batched else get_regret(reward, rewards)
            if record_rank    : out['rank'  ]       = list(map(get_rank,reward,rewards,actions)) if batched else get_rank(reward, rewards, actions)
            if record_ope_loss: out['ope_loss']     = _get_ope_loss(learner)

            if interaction.keys()-OnPolicyEvaluation.IMPLICIT_EXCLUDE:
                out.update({k: interaction[k] for k in interaction.keys()-OnPolicyEvaluation.IMPLICIT_EXCLUDE})

            if info:
                out.update(info)
                info.clear()

            if out:
                if batched:
                    #we flatten batched items so output works seamlessly with Result
                    yield from ({k:v[i] for k,v in out.items()} for i in range(len(context)))
                else:
                    yield out

class OffPolicyEvaluation(EvaluationTask):

    IMPLICIT_EXCLUDE = {"actions", "rewards", "action", "reward", "probability", "ope_loss"}

    def __init__(self, 
        record: Sequence[Literal['reward','time','ope_loss']] = ['reward'],
        learn: bool = True,
        predict: bool = True,
        seed: float = None) -> None:
        """
        Args:
            record: The datapoints to record for each interaction.
            learn: Indicates that off-policy learning should occur as parrt of the off-policy task.
            evals: Indicates that off-policy evaluation should occur as part of the off-policy task.
            seed: Provide an explicit seed to use during evaluation. If not provided a default is used.
        """

        self._record  = [record] if isinstance(record,str) else record
        self._learn   = learn
        self._predict = predict
        self._seed    = seed

        if 'ope_loss' in self._record:
            # OPE loss metric is only available for VW models
            # Divide by the number of samples for the average loss metric and see this article for more info
            # https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/off_policy_evaluation.html
            PackageChecker.vowpalwabbit('OffPolicyEvaluation.__init__')

    def process(self, learner: Learner, interactions: Iterable[Interaction]) -> Iterable[Mapping[Any,Any]]:

        learner = SafeLearner(learner, self._seed if self._seed is not None else CobaContext.store.get("experiment_seed"))

        first, interactions = peek_first(interactions)        
        batched  = first and isinstance(first['context'], Batch)
        discrete = 'actions' in first and len(first['actions'][0] if batched else first['actions']) > 0

        first_rewards = first.get('rewards',[None])[0] if batched else first.get('rewards',None)
        first_actions = first.get('actions',[None])[0] if batched else first.get('actions',None)

        if self._predict and first_actions is None:
            raise CobaException("Interactions need to have 'actions' defined for OPE.")

        if self._predict and (first_rewards is None or first_actions is None):
            raise CobaException("Interactions need to have 'rewards' defined for OPE. This can be done using `Environments.ope_rewards`.")

        try:
            learner.request(first['context'],first['actions'],first['actions'])
        except Exception as ex:
            implements_request = '`request`' not in str(ex)
        else:
            implements_request = True

        record_time     = 'time'     in self._record
        record_reward   = 'reward'   in self._record and self._predict and 'actions' in first
        record_ope_loss = 'ope_loss' in self._record and self._predict and 'actions' in first

        request = learner.request
        predict = learner.predict
        learn   = learner.learn
        info    = CobaContext.learning_info

        info.clear()

        for interaction in interactions:

            out = {}
            interaction = interaction.copy()

            log_context = interaction.pop('context')
            log_action  = interaction.pop('action')
            log_reward  = interaction.pop('reward') 
            log_prob    = interaction.pop('probability',None)
            log_rewards = interaction.pop('rewards',None)
            log_actions = interaction.pop('actions',None)

            batched  = isinstance(log_context, Batch)
            discrete = log_actions and len(log_actions[0] if batched else log_actions) > 0

            if record_time:
                predict_time = 0
                learn_time   = 0

            if self._predict and log_actions is not None and log_rewards is not None:
                if implements_request:
                    if discrete:
                        start_time   = time.time()
                        on_probs     = request(log_context,log_actions,log_actions)
                        predict_time = time.time()-start_time
                        if not batched:
                            ope_reward = sum(p*log_rewards.eval(a) for p,a in zip(on_probs,log_actions))
                        else:
                            ope_reward = [ sum(p*R.eval(a) for p,a in zip(P,A)) for P,A,R in zip(on_probs,log_actions,log_rewards) ]
                    else:
                        start_time   = time.time()
                        if not batched:
                            on_prob = request(log_context,log_actions,[log_action])
                        else:
                            on_prob = request(log_context,log_actions,log_action)
                        predict_time = time.time()-start_time
                        if not batched:
                            ope_reward   = on_prob*log_rewards.eval(log_action)
                        else:
                            ope_reward   = [p*r for p,r in zip(on_prob,log_rewards.eval(log_action))]
                else:
                    start_time        = time.time()
                    on_action,on_prob = predict(log_context, log_actions)[:2]
                    predict_time      = time.time()-start_time

                    #in theory we could use a ratio average in this case
                    #this would give us a lower variance estimate but we'd have
                    #to add a "weight" column to our output and handle this in Result
                    ope_reward = log_rewards.eval(on_action)

            if self._learn:
                start_time = time.time()
                if self._learn: learn(log_context, log_actions, log_action, log_reward, log_prob)
                learn_time = time.time() - start_time

            if record_time  : out['predict_time'] = predict_time
            if record_time  : out['learn_time']   = learn_time
            if record_reward: out['reward']       = ope_reward

            if interaction.keys()-OnPolicyEvaluation.IMPLICIT_EXCLUDE:
                out.update({k: interaction[k] for k in interaction.keys()-OnPolicyEvaluation.IMPLICIT_EXCLUDE})

            if record_ope_loss: out['ope_loss'] = _get_ope_loss(learner)

            if info:
                out.update(info)
                info.clear()

            if out:
                if batched:
                    #we flatten batched items so output works seamlessly with Result
                    yield from ({k:v[i] for k,v in out.items()} for i in range(len(log_context)))
                else:
                    yield out

class ExplorationEvaluation(EvaluationTask):

    def __init__(self,
        record: Sequence[Literal['context','actions','action','reward','probability','time']] = ['reward'],
        ope: bool = True,
        qpct: float = .005,
        cmax: float = 1.0,
        cinit: float = None,
        seed: float = None) -> None:
        """
        Args:
            record: The datapoints to record for each interaction.
            qpct: The unbiased case is q = 0. Smaller values give better estimates but reject more data.
            cmax: The maximum value that the evaluator is allowed to use for `c` (the rejection sampling multiplier).
                To get an unbiased estimate we need a `c` value such that c*on_prob/log_prob <= 1 for all 
                on_prob/log_prob. The value `cmax` determines the maximum value `c` can be in order to guarantee `c`
                will be an unbiased estimate. In practice, it is often better to not modify this value and instead
                change `qpct` to control the biasedness of the estimate.
            cinit: The initial value to use for `c` (the rejection sampling multiplier). If left as None then a very
                conservative, data-adaptive estimate is used to initialize `c`. Without prior knowledge of the data
                leaving this as `None` is likely the best course of action.
            seed: Provide an explicit seed to use during evaluation. If not provided a default is used.
        """

        #An implementation of https://arxiv.org/ftp/arxiv/papers/1210/1210.4862.pdf

        self._record = [record] if isinstance(record,str) else record
        self._ope    = ope
        self._qpct   = qpct
        self._cmax   = cmax
        self._cinit  = cinit
        self._seed   = seed

    def process(self, learner: Learner, interactions: Iterable[Interaction]) -> Iterable[Mapping[Any,Any]]:

        rng     = CobaRandom(self._seed if self._seed is not None else CobaContext.store.get("experiment_seed"))
        learner = SafeLearner(learner, self._seed if self._seed is not None else CobaContext.store.get("experiment_seed"))

        first_100, interactions = peek_first(interactions,n=100)
        if first_100 is None: return []

        first = first_100[0]
        batched = first and isinstance(first['context'], Batch)
        discrete = 'actions' in first and len(first['actions'][0] if batched else first['actions']) > 0

        if not all(k in first.keys() for k in ['context', 'action', 'reward', 'actions', 'probability']):
            raise CobaException("ExplorationEvaluation requires interactions with `['context', 'action', 'reward', 'actions', 'probability']`")

        if not discrete:
            raise CobaException("ExplorationEvaluation does not currently support continuous simulations")

        if batched:
            raise CobaException("ExplorationEvaluation does not currently support batching")

        try:
            learner.request(first['context'],first['actions'],first['actions'])
        except Exception as ex:
            if '`request`' in str(ex):
                raise CobaException("ExplorationEvaluation requires Learners to implement a `request` method")

        record_time     = 'time'        in self._record
        record_reward   = 'reward'      in self._record
        record_action   = 'action'      in self._record
        record_actions  = 'actions'     in self._record
        record_context  = 'context'     in self._record
        record_prob     = 'probability' in self._record
        record_ope_loss = 'ope_loss'    in self._record

        request = learner.request
        learn   = learner.learn
        info    = CobaContext.learning_info

        info.clear()

        first_probs = [i['probability'] for i in first_100] + [(1-i['probability'])/(len(i['actions'])-1) for i in first_100]
        ope_rewards = []
        Q           = []
        c           = self._cinit or min(first_probs+[self._cmax])
        t           = 0

        for interaction in interactions:

            t += 1

            info.clear()
            interaction = interaction.copy()

            log_context      = interaction.pop('context')
            log_actions      = interaction.pop('actions')
            log_action       = interaction.pop('action')
            log_reward       = interaction.pop('reward') 
            log_prob         = interaction.pop('probability')
            log_rewards      = interaction.pop('rewards',None)
            log_action_index = log_actions.index(log_action)

            if record_time:
                predict_time = 0
                learn_time   = 0

            start_time = time.time()
            on_probs = request(log_context,log_actions,log_actions)
            on_prob = on_probs[log_action_index]
            predict_time = time.time()-start_time

            if self._ope and log_rewards: ope_rewards.append(sum(map(__mul__, on_probs, map(log_rewards.eval,log_actions))))

            #I tested many techniques for managing Q and estimating its qpct percentile... 
            #Implemented here is the insort method because it provided the best runtime by far.
            #The danger of this method is that the computational complexity is T (due to inserts).
            #Even so, T has to become very large (many millions?) before it is slower than alternatives.
            #The most obvious alternatively I also played with was dequeue with a fixed size/complexity.
            #However, this requires sorting the dequeue for every accepted action which is incredibly slow.
            if on_prob != 0:
                insort(Q,log_prob/on_prob)

            #we want c*on_prob/log_prob <= 1 approximately (1 - qpct) of the time.
            #We know that if c <= min(log_prob) this condition will be met.
            #This might be too conservative though because it doesn't consider
            #the value of on_prob. For example if on_prob:=log_prob then the
            #above condition will be met if c = 1 which will be >= min(log_prob).
            if rng.random() <= c*on_prob/log_prob:

                out = {}
                if ope_rewards:
                    ope_rewards[-1] = log_reward
                else:
                    ope_rewards.append(log_reward)

                start_time = time.time()
                learn(log_context, log_actions, log_action, log_reward, on_prob)
                learn_time = time.time() - start_time

                if record_time    : out['predict_time'] = predict_time
                if record_time    : out['learn_time']   = learn_time
                if record_context : out['context']      = log_context
                if record_actions : out['actions']      = log_actions
                if record_action  : out['action']       = log_action
                if record_reward  : out['reward']       = mean(ope_rewards)
                if record_prob    : out['probability']  = on_prob
                if record_ope_loss: out['ope_loss']     = _get_ope_loss(learner)

                if info: out.update(info)
                if out : yield out

                ope_rewards.clear()
                c = min(percentile(Q,self._qpct,sort=False), self._cmax)

        if ope_rewards:
            pass
            #If we hit this it means that there was rejected data at the end of the
            #simulation. I haven't found a good way of dealing with this case. In
            #general I don't think it should be a problem unless we are working with
            #very small datasets. Commented out below is the previous functionality:
            #out = {}
            #if record_time   : out['predict_time'] = predict_time
            #if record_reward : out['reward']       = mean(ope_rewards)
            #if out: yield out

class LambdaEvaluation(EvaluationTask):

    def __init__(self, evaluator: Callable[[Learner,Iterable[Interaction]],Iterable[Mapping[Any,Any]]]):
        self._evaluator = evaluator

    def process(self, learner: Learner, interactions: Iterable[Interaction], seed:float = None) -> Iterable[Mapping[Any,Any]]:
        yield from self._evaluator(learner,interactions)

class SimpleLearnerInfo(LearnerTask):
    """Describe a Learner using its name and hyperparameters."""

    def process(self, item: Learner) -> Mapping[Any,Any]:
        return SafeLearner(item).params

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

    def process(self, environment: Environment, interactions: Iterable[Interaction]) -> Mapping[Any,Any]:

        #sources:
        #[1]: https://arxiv.org/pdf/1808.03591.pdf (lorena2019complex)
        #[2]: https://link.springer.com/content/pdf/10.1007/978-3-540-31883-5.pdf#page=468 (castiello2005meta)
        #[3]: https://link.springer.com/content/pdf/10.1007/s10044-012-0280-z.pdf (reif2014automatic)
        #[4]: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.440.6255&rep=rep1&type=pdf

        #[3] found that information theoretic measures and landmarking measures are most important

        interactions = peek_first(interactions)[1]

        if not interactions: return {}

        contexts, _ ,rewards = zip(*[ (i['context'], i['actions'], i['rewards']) for i in interactions ])
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

def _get_ope_loss(learner: SafeLearner) -> float:
    # OPE loss metric is only available for VW models
    try:
        return learner._learner._vw._vw.get_sum_loss()
    except AttributeError:
        return float("nan")