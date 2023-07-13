import time
import warnings

from operator import __mul__
from statistics import mean
from bisect import insort
from typing import Any, Iterable, Sequence, Mapping, Optional
from coba.backports import Literal

from coba.exceptions import CobaException
from coba.random import CobaRandom
from coba.contexts import CobaContext
from coba.environments import Environment
from coba.learners import Learner, SafeLearner
from coba.primitives import Batch, argmax
from coba.statistics import percentile
from coba.utilities import PackageChecker, peek_first

from coba.evaluators.primitives import Evaluator, get_ope_loss

class OnPolicyEvaluator(Evaluator):

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

    def evaluate(self, environment: Optional[Environment], learner: Optional[Learner]) -> Iterable[Mapping[Any,Any]]:

        interactions = environment.read()
        learner = SafeLearner(learner, self._seed if self._seed is not None else CobaContext.store.get("experiment_seed"))

        first, interactions = peek_first(interactions)

        if not interactions: return []

        batched  = first and isinstance(first['context'], Batch)

        for key in ['rewards','context','actions']:
            if key not in first:
                raise CobaException(f"OnPolicyEvaluator requires every interaction to have '{key}'")

        for key in ['rewards','actions']:
            if (first[key][0] if batched else first[key]) is None:
                raise CobaException(f"OnPolicyEvaluator requires every interaction to have a not None '{key}'")

        discrete = len(first['actions'][0] if batched else first['actions']) > 0
        if not discrete:
            for metric in set(self._record).intersection(OnPolicyEvaluator.ONLY_DISCRETE):
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
        get_regret = lambda reward, rewards, actions: rewards.eval(argmax(actions,rewards))-reward
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
            if record_regret  : out['regret']       = list(map(get_regret,reward,rewards)) if batched else get_regret(reward, rewards, actions)
            if record_rank    : out['rank'  ]       = list(map(get_rank,reward,rewards,actions)) if batched else get_rank(reward, rewards, actions)
            if record_ope_loss: out['ope_loss']     = get_ope_loss(learner)

            out.update({k: interaction[k] for k in interaction.keys()-OnPolicyEvaluator.IMPLICIT_EXCLUDE})

            if info:
                out.update(info)
                info.clear()

            if out:
                if batched:
                    #we flatten batched items so output works seamlessly with Result
                    yield from ({k:v[i] for k,v in out.items()} for i in range(len(context)))
                else:
                    yield out

class OffPolicyEvaluator(Evaluator):

    IMPLICIT_EXCLUDE = {"actions", "rewards", "action", "reward", "probability", "ope_loss"}

    def __init__(self,
        record: Sequence[Literal['reward','time','ope_loss','probability','action','context','actions','rewards']] = ['reward'],
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
            PackageChecker.vowpalwabbit('OffPolicyEvaluator.__init__')

    def evaluate(self, environment: Optional[Environment], learner: Optional[Learner]) -> Iterable[Mapping[Any,Any]]:

        interactions = environment.read()
        learner = SafeLearner(learner, self._seed if self._seed is not None else CobaContext.store.get("experiment_seed"))

        first, interactions = peek_first(interactions)

        if not interactions:return []

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

        record_time     = 'time'        in self._record
        record_reward   = 'reward'      in self._record and self._predict and 'actions' in first
        record_ope_loss = 'ope_loss'    in self._record and self._predict and 'actions' in first
        record_action   = 'action'      in self._record
        record_prob     = 'probability' in self._record
        record_context  = 'context'     in self._record
        record_actions  = 'actions'     in self._record
        record_rewards  = 'rewards'     in self._record and discrete

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

            if record_time  :  out['predict_time'] = predict_time
            if record_time  :  out['learn_time']   = learn_time
            if record_reward:  out['reward']       = ope_reward
            if record_action:  out['action']       = log_action
            if record_prob:    out['probability']  = log_prob
            if record_context: out['context']      = log_context
            if record_actions: out['actions']      = log_actions
            if record_rewards: out['rewards']      = log_rewards

            out.update({k: interaction[k] for k in interaction.keys()-OffPolicyEvaluator.IMPLICIT_EXCLUDE})

            if record_ope_loss: out['ope_loss'] = get_ope_loss(learner)

            if info:
                out.update(info)
                info.clear()

            if out:
                if batched:
                    #we flatten batched items so output works seamlessly with Result
                    yield from ({k:v[i] for k,v in out.items()} for i in range(len(log_context)))
                else:
                    yield out

class ExplorationEvaluator(Evaluator):

    def __init__(self,
        record: Sequence[Literal['context','actions','action','reward','probability','time']] = ['reward'],
        ope: bool = None,
        qpct: float = .005,
        cmax: float = 1.0,
        cinit: float = None,
        seed: float = None) -> None:
        """
        Args:
            record: The datapoints to record for each interaction.
            ope: Indicates whether off-policy estimates should be included from rejected training examples.
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

    def evaluate(self, environment: Optional[Environment], learner: Optional[Learner]) -> Iterable[Mapping[Any,Any]]:

        interactions = environment.read()
        learner      = SafeLearner(learner, self._seed if self._seed is not None else CobaContext.store.get("experiment_seed"))
        rng          = CobaRandom(self._seed if self._seed is not None else CobaContext.store.get("experiment_seed"))

        first_100, interactions = peek_first(interactions,n=100)

        if not interactions: return []

        first = first_100[0]
        batched = first and isinstance(first['context'], Batch)
        discrete = 'actions' in first and len(first['actions'][0] if batched else first['actions']) > 0

        if self._ope is None: self._ope = ('rewards' in first)

        if not all(k in first.keys() for k in ['context', 'action', 'reward', 'actions', 'probability']):
            raise CobaException("ExplorationEvaluator requires interactions with `['context', 'action', 'reward', 'actions', 'probability']`")

        if not discrete:
            raise CobaException("ExplorationEvaluator does not currently support continuous actions")

        if batched:
            raise CobaException("ExplorationEvaluator does not currently support batching")

        try:
            learner.request(first['context'],first['actions'],first['actions'])
        except Exception as ex:
            if '`request`' in str(ex):
                raise CobaException("ExplorationEvaluator requires Learners to implement a `request` method")

        if self._ope and 'rewards' not in first:
            raise CobaException((
                "ExplorationEvaluator was called with ope=True but the given interactions "
                "do not have an ope rewards. To assign ope rewards to environments call "
                "Environments.ope_rewards() with desired parameters before running the "
                "experiment."
            ))

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
        c           = self._cinit or min(list(filter(None,first_probs))+[self._cmax])
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

            #I tested many techniques for both maintaining Q and estimating its qpct percentile...
            #Implemented here is the insort method because it provided the best runtime by far.
            #The danger of this method is that the computational complexity is T (due to inserts).
            #Even so, T has to become very large (many millions?) before it is slower than alternatives.
            #The most obvious alternative I played with was using dequeue with a fixed size/complexity.
            #However, this requires sorting the dequeue for every accepted action which is incredibly slow.
            if on_prob != 0:
                insort(Q,log_prob/on_prob)

            #we want c*on_prob/log_prob <= 1 approximately (1 - qpct) of the time.
            #We know that if c <= min(log_prob) this condition will be met 100% of the time.
            #This might be too conservative though because it doesn't consider
            #the value of on_prob. For example if on_prob:=log_prob then the
            #above condition will be met if c = 1 which will be >= min(log_prob).
            if rng.random() <= c*(on_prob/log_prob):

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
                if record_ope_loss: out['ope_loss']     = get_ope_loss(learner)

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
