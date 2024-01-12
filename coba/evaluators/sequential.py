import time
import warnings

from operator import mul
from statistics import mean
from bisect import insort

from typing import Any, Iterable, Sequence, Mapping, Optional, Literal

from coba.exceptions  import CobaException
from coba.random      import CobaRandom
from coba.context     import CobaContext
from coba.safety      import SafeLearner
from coba.primitives  import is_batch, Dense, Sparse, Learner, Environment, Evaluator
from coba.statistics  import percentile
from coba.utilities   import PackageChecker, peek_first

def get_ope_loss(learner) -> float:
    # OPE loss metric is only available for VW models
    try:
        return learner.learner._vw._vw.get_sum_loss()
    except AttributeError:
        return float("nan")

class SequentialCB(Evaluator):
    """Sequential evaluation for CB learners."""

    _ONLY_DISCRETE    = set()
    _IMPLICIT_EXCLUDE = {"context", "actions", "rewards", "action", "reward", "probability", "eval_rewards", "learn_rewards"}

    def __init__(self,
        record: Sequence[Literal['reward','time','probability','action','context','actions','rewards']] = ['reward'],
        learn : Optional[Literal['on','off','ips','dr','dm']] = 'on',
        eval  : Optional[Literal['on','ips','dr','dm']] ='on',
        seed  : float = None) -> None:
        """Instantiate a SequentialCB evaluator.

        Args:
            record: Variables to record for each learner interaction.
            learn: action and reward revealed to learner on `learn` call:
                *on* --- on-policy action/reward (requires 'actions' and 'rewards'),
                *off* --- off-policy action/reward (requires 'action' and 'reward'),
                *ips* --- on-policy action/ips-reward (requires 'actions', 'action', 'reward', and 'probability'),
                *dr* --- on-policy action/dr-reward (requires 'actions', 'action', and 'reward'),
                *dm* --- on-policy action/dm-reward (requires 'actions', 'action', and 'reward'),
                *None* --- `learn` is not called.
            eval: reward recorded given learner prediction:
                *on* --- actual reward (requires 'actions' and 'rewards'),
                *ips* --- ips-reward (requires 'actions', 'action', 'reward', and 'probability'),
                *dr* --- dr-reward (requires 'actions', 'action', and 'reward'),
                *dm* --- dm-reward (requires 'actions', 'action', and 'reward'),
                *None* --- no reward is recorded.
            seed: Determine which action is played when learners return an action PMF.
        """

        self._record = [record] if isinstance(record,str) else record
        self._learn  = learn or ''
        self._eval   = eval or ''
        self._seed   = seed

        if 'ope_loss' in self._record:
            # OPE loss metric is only available for VW models
            # Divide by the number of samples for the average loss metric and see this article for more info
            # https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/off_policy_evaluation.html
            PackageChecker.vowpalwabbit('OffPolicyEvaluator')

    @property
    def params(self) -> Mapping[str,Any]:
        return {'learn': self._learn, 'eval': self._eval, 'seed': self._seed }

    def _required(self, has_score:bool) -> set:
        learn,eval = self._learn,self._eval

        pred = (learn and learn != 'off') or (eval and (eval != 'ips' or not has_score))
        off  = (learn and learn != 'on')  or (eval and eval != 'on')
        rwds = (learn == 'on')            or (eval == 'on')

        required_keys = set()
        if pred: required_keys.update(['actions'])
        if off : required_keys.update(['action','reward'])
        if rwds: required_keys.update(['rewards'])

        return required_keys

    def _validate(self, first: Mapping, has_score:bool) -> None:
        required_keys = self._required(has_score)
        missing_keys  = required_keys-first.keys()

        if missing_keys:
            raise CobaException(f"{self.__class__.__name__}(learn={self._learn or None},eval={self._eval or None}) requires {missing_keys}.")

        if not self._discrete(first):
            for metric in set(self._record).intersection(SequentialCB._ONLY_DISCRETE):
                warnings.warn(f"The {metric} metric can only be calculated for discrete environments")#pragma: no cover

    def _discrete(self,interaction):
        if 'actions' not in interaction:
            return False
        else:
            batched   = interaction and (is_batch(interaction.get('context')) or is_batch(interaction.get('actions')))
            actions   = interaction['actions'][0] if batched else interaction['actions']
            discrete  = len(actions or []) > 0

        return discrete

    def _results(self,learner:SafeLearner,first:Mapping, interactions:Iterable[Mapping]) -> Iterable[Mapping[Any,Any]]:
        #import here to avoid circular dependencies...
        from coba.environments.filters import OpeRewards, Batch, BatchSafe

        batched   = first and (is_batch(first.get('context')) or is_batch(first.get('action')) or is_batch(first.get('actions')))
        discrete  = self._discrete(first)
        has_score = learner.has_score

        learn,eval = self._learn,self._eval

        N          = 0
        n_zero_lrn = 0
        n_zero_val = 0
        n_zeroes   = lambda R: int(R==0) if not batched else sum([r==0 for r in R])
        lrn_warned   = False
        val_warned   = False

        lrn_on  = learn == 'on'
        lrn_off = learn == 'off'
        lrn_ips = learn == 'ips'
        lrn_dr  = learn == 'dr'
        lrn_dm  = learn == 'dm'
        val_on  = eval  == 'on'
        val_ips = eval  == 'ips'
        val_dr  = eval  == 'dr'
        val_dm  = eval  == 'dm'

        has_context = 'context'     in first
        has_actions = 'actions'     in first
        has_rewards = 'rewards'     in first
        has_reward  = 'reward'      in first
        has_action  = 'action'      in first
        has_prob    = 'probability' in first

        out_prob     = 'probability' in self._record and eval
        out_time     = 'time'        in self._record
        out_action   = 'action'      in self._record and eval
        out_context  = 'context'     in self._record
        out_actions  = 'actions'     in self._record and has_actions
        out_rewards  = 'rewards'     in self._record and has_rewards
        out_reward   = 'reward'      in self._record and eval
        out_ope_loss = 'ope_loss'    in self._record and eval

        should_pred = (learn and not lrn_off) or val_on or val_dm or val_dr or (val_ips and not has_score) or out_action

        if out_rewards and discrete:
            get_rewards = (lambda Rs,As:[[R(a) for a in A] for R,A in zip(Rs,As)]) if batched else (lambda R,A: [R(a) for a in A])
        if out_rewards and not discrete:
            get_rewards = lambda R,_: R

        info = CobaContext.learning_info
        info.clear()

        learn_type = 'IPS' if lrn_ips else 'DR' if lrn_dr else 'DM' if lrn_dm else None
        eval_type  = 'IPS' if val_ips else 'DR' if val_dr else 'DM' if val_dm else None

        if learn_type:
            interactions = BatchSafe(OpeRewards(learn_type,target='learn_rewards',features=[1,'a','xa'])).filter(interactions)

        if eval_type and eval_type != learn_type:
            interactions = BatchSafe(OpeRewards(eval_type,target='eval_rewards',features=[1,'a','xa'])).filter(interactions)

        learn_target = 'learn_rewards'
        eval_target  = 'eval_rewards' if eval_type and eval_type != learn_type else 'learn_rewards'

        for interaction in interactions:

            context = interaction['context'     ] if has_context else None
            actions = interaction['actions'     ] if has_actions else None
            rewards = interaction['rewards'     ] if has_rewards else None
            off_rwd = interaction['reward'      ] if has_reward  else None
            off_act = interaction['action'      ] if has_action  else None
            off_pr  = interaction['probability' ] if has_prob    else None

            lrn_rwds = interaction[learn_target] if learn_type else rewards if lrn_on else None
            val_rwds = interaction[eval_target ] if eval_type  else rewards if val_on else None

            N += 1 if not batched else len(lrn_rwds) if lrn_rwds else len(val_rwds)

            start = time.time()
            if should_pred: on_act,on_pr,on_kw=learner.predict(context,actions)
            pred_time = time.time()-start

            if eval:
                if not val_ips or not has_score:
                    eval_reward = val_rwds(on_act)
                else:
                    SR = learner.score(context,actions,off_act),val_rwds(off_act)
                    eval_reward = mul(*SR) if not batched else Batch.List(map(mul,*SR))
                n_zero_val += n_zeroes(eval_reward)

            if learn:
                learn_reward = off_rwd if lrn_off else lrn_rwds(on_act)
                n_zero_lrn += n_zeroes(learn_reward)
                start = time.time()
                if lrn_off: learner.learn(context, off_act, learn_reward, off_pr         )
                else      : learner.learn(context, on_act , learn_reward, on_pr , **on_kw)
                learn_time = time.time()-start

            if 200 > N and N > 20 and not lrn_warned and learn_type == 'IPS' and n_zero_lrn <= (N*.01):
                lrn_warned = True
                #The warning will display even in quiet mode due to CobaException
                CobaContext.logger.log(CobaException(
                    "WARNING: the learner's predicted actions are highly correlated"
                    " with the logging policy and we are learning from an IPS reward"
                    " estimate. This can cause biased learning due to IPS assuming"
                    " independence between logging and learning policies. To solve"
                    " this consider using DM or DR instead of IPS, make sure the"
                    " rng seed is not the same in the logging and learning policy,"
                    " or shuffle the order of interactions in the Environment."
                ))

            if 200 > N and N > 20 and not val_warned and eval_type == 'IPS' and not has_score and n_zero_val <= (N*.01):
                val_warned = True
                #The warning will display even in quiet mode due to CobaException
                CobaContext.logger.log(CobaException(
                    "WARNING: the learner's predicted actions are highly correlated"
                    " with the logging policy and we are evaluating from an IPS reward"
                    " estimate. This can cause biased evaluation due to IPS assuming"
                    " independence between logging and learning policies. To solve"
                    " this consider using DM or DR instead of IPS or make sure the"
                    " rng seed is not the same in the logging and learning policy,"
                    " or shuffle the order of interactions in the Environment."
                ))

            out = {}

            if out_time    : out['predict_time'] = pred_time
            if out_time    : out['learn_time']   = learn_time
            if out_context : out['context']      = context
            if out_actions : out['actions']      = actions
            if out_action  : out['action']       = on_act
            if out_prob    : out['probability']  = on_pr
            if out_reward  : out['reward']       = eval_reward
            if out_rewards : out['rewards']      = get_rewards(rewards,actions)
            if out_ope_loss: out['ope_loss']     = get_ope_loss(learner)

            out.update({k: interaction[k] for k in interaction.keys()-SequentialCB._IMPLICIT_EXCLUDE})

            if info:
                out.update(info)
                info.clear()

            if out:
                yield out

    def evaluate(self, environment: Optional[Environment], learner: Optional[Learner]) -> Iterable[Mapping[Any,Any]]:

        first, interactions = peek_first(environment.read())
        seed = self._seed if self._seed is not None else CobaContext.store.get("experiment_seed")

        if not interactions: return []

        from coba.environments import Finalize, Unbatch, BatchSafe

        learner = SafeLearner(learner, seed)
        self._validate(first,learner.has_score)
        results = self._results(learner, first, BatchSafe(Finalize()).filter(interactions))

        #We Unbatch to work with Result
        yield from Unbatch().filter(results)

class SequentialIGL(Evaluator):
    """Sequential evaluation for IGL learners."""

    def __init__(self,
        record: Sequence[Literal['reward','feedback','time','prob','action','context','actions','rewards','feedbacks']] = ['reward','feedback'],
        seed: int = None) -> None:
        """Instantiate SequentialIGL learners.

        Args:
            record: Variables to record for each learner interaction.
            seed: Determine which action is played when learners return an action PMF.
        """

        self._record = list(record)
        self._seed   = seed

    @property
    def params(self) -> Mapping[str,Any]:
        return { 'seed': self._seed }

    def evaluate(self, environment: Optional[Environment], learner: Optional[Learner]) -> Iterable[Mapping[Any,Any]]:

        class IglEnvironment:
            def __init__(self,env: Environment) -> None:
                self._env = env

            def read(self):
                first,interactions = peek_first(iter(self._env.read()))

                if not interactions:
                    return

                is_missing = 'context' not in first
                is_sparse  = not is_missing and isinstance(first['context'],Sparse)
                is_dense   = not is_missing and isinstance(first['context'],Dense)

                for interaction in interactions:

                    new = interaction.copy()

                    f,r = new['feedbacks'],new['rewards']
                    new['feedbacks'],new['rewards'] = r,f

                    userid = new['userid']

                    if is_missing:
                        new['context'] = userid
                    elif is_sparse:
                        new['context'] = dict(userid=userid,**new['context'])
                    elif is_dense:
                        new['context'] = (userid,)+tuple(new['context'])
                    else:
                        new['context'] = (userid, new['context'])

                    yield new

        envIGL = IglEnvironment(environment)
        record = self._record+['action']
        seed   = self._seed

        out_action   = 'action'   in self._record
        out_feedback = 'feedback' in self._record
        out_reward   = 'reward'   in self._record

        for out in SequentialCB(record,seed=seed).evaluate(envIGL,learner):
            if out_feedback  : out['feedback'] = out['reward']
            if out_reward    : out['reward']   = out.pop('feedbacks')(out['action'])
            if not out_action: del out['action']

            yield out

class RejectionCB(Evaluator):
    """Rejective evaluation for CB learners.

    This evaluator uses rejection sampling to simulate on-policy learner performance
    using only off-policy logged data. For this to work the evaluator requires each
    interaction to have 'actions', 'action', 'reward', and 'probability'.

    This gives an unbiased estimation of on-policy performance assuming two conditions

        1. The reward distribution of each interaction is stationary.
        2. The `cpct` parameter of the evaluator is set to 0.

    Remarks:
        This is an implementation of Dudík et al. (2012). The `cpct` parameter of our
        implementation is what Dudík calls *q* and `cinit` is Dudík calls `c1`. To use
        double-robust off-policy estimation as Dudík does also set `ope` to 'dr'.

    References:
        * Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li. 2012. Sample-efficient
          nonstationary policy evaluation for contextual bandits. In Proceedings of the
          Twenty-Eighth Conference on Uncertainty in Artificial Intelligence (UAI'12). AUAI
          Press, Arlington, Virginia, USA, 247-254.
    """

    def __init__(self,
        record: Sequence[Literal['context','actions','action','reward','probability','time']] = ['reward'],
        ope   : Optional[Literal['ips','dr','dm']] = None,
        cpct  : float = .005,
        cmax  : float = 1.0,
        cinit : float = None,
        seed  : float = None) -> None:
        """Instantiate a RejectionCB evaluator.

        Args:
            record: The datapoints to record for each interaction.
            ope: Indicates whether off-policy estimates should be included from rejected training examples.
            cpct: The unbiased case is q = 0. Smaller values give better estimates but rejects more data.
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

        self._record = [record] if isinstance(record,str) else record
        self._ope    = ope
        self._cpct   = cpct
        self._cmax   = cmax
        self._cinit  = cinit
        self._seed   = seed

    @property
    def params(self) -> Mapping[str,Any]:
        return {'ope': self._ope, 'cpct': self._cpct, 'cmax': self._cmax, 'cinit': self._cinit, 'seed': self._seed }

    def evaluate(self, environment: Optional[Environment], learner: Optional[Learner]) -> Iterable[Mapping[Any,Any]]:

        #import here to avoid circular dependencies...
        from coba.environments import OpeRewards, Finalize, BatchSafe

        interactions = environment.read()
        learner      = SafeLearner(learner, self._seed if self._seed is not None else CobaContext.store.get("experiment_seed"))
        rng          = CobaRandom(self._seed if self._seed is not None else CobaContext.store.get("experiment_seed"))

        first_100, interactions = peek_first(interactions,n=100)

        if not interactions: return []

        first    = first_100[0]
        batched  = first and (is_batch(first.get('context')) or is_batch(first.get('actions')))
        discrete = 'actions' in first and len(first['actions'][0] if batched else first['actions']) > 0

        if not all(k in first.keys() for k in ['context', 'action', 'reward', 'actions', 'probability']):
            raise CobaException("ExplorationEvaluator requires interactions with `['context', 'action', 'reward', 'actions', 'probability']`")

        if not discrete:
            raise CobaException("ExplorationEvaluator does not currently support continuous actions")

        if batched:
            raise CobaException("ExplorationEvaluator does not currently support batching")

        if not learner.has_score:
            raise CobaException("ExplorationEvaluator requires Learners to implement a `score` method")

        record_time     = 'time'        in self._record
        record_reward   = 'reward'      in self._record
        record_action   = 'action'      in self._record
        record_actions  = 'actions'     in self._record
        record_context  = 'context'     in self._record
        record_prob     = 'probability' in self._record
        record_ope_loss = 'ope_loss'    in self._record

        score = learner.score
        learn = learner.learn
        pred  = learner.predict
        info  = CobaContext.learning_info

        info.clear()

        first_probs = [i['probability'] for i in first_100] + [(1-i['probability'])/(len(i['actions'])-1) for i in first_100]
        ope_rewards = []
        Q           = []
        c           = self._cinit or min(list(filter(None,first_probs))+[self._cmax])
        t           = 0

        ope_type = self._ope.upper() if self._ope else None

        interactions = BatchSafe(Finalize()).filter(interactions)

        if ope_type:
            interactions = BatchSafe(OpeRewards(ope_type,features=[1,'a','xa'])).filter(interactions)

        ope_ips = ope_type == 'IPS'

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

            start_time = time.time()
            on_prob  = score(log_context,log_actions,log_action)
            predict_time = time.time()-start_time

            if ope_type:
                ope_rewards.append(on_prob*log_rewards(log_action) if ope_ips else log_rewards(pred(log_context,log_actions)[0]))

            #I tested many techniques for both maintaining Q and estimating its qpct percentile...
            #Implemented here is the insort method because it provided the best runtime by far.
            #The danger of this method is that the computational complexity is T (due to inserts).
            #Even so, T has to become very large (many millions?) before it is slower than alternatives.
            #The most obvious alternative I played with was using dequeue with a fixed size/complexity.
            #However, this requires sorting the dequeue for every accepted action which is incredibly slow.
            if on_prob != 0:
                insort(Q,log_prob/on_prob)

            #c \in log_prob/on_prob
            #so c is small when log_prob is small and on_prob is large

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
                learn(log_context, log_action, log_reward, on_prob)
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
                c = min(percentile(Q,self._cpct,sort=False), self._cmax)

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
