from collections import abc
from math import isclose
from abc import ABC, abstractmethod
from itertools import repeat
from typing import Any, Sequence, Union, Tuple, Callable, Mapping, Optional, Type

from coba.exceptions import CobaException
from coba.random import CobaRandom
from coba.primitives import Context, Action, Actions, AIndex
from coba.primitives import Batch, Dense, Sparse, HashableDense, HashableSparse

kwargs = Mapping[str,Any]
Score  = float
PMF    = Sequence[float]
PDF    = Callable[[Union[Action,AIndex]],float]

class Probs(list):
    pass

class ActionScore(tuple):
    def __new__(self, action: Union[Action,AIndex], score: Score):
        return tuple.__new__(ActionScore, (action, score))

Prediction = Union[
    PDF,
    Probs,
    ActionScore,
    Tuple[PDF                       , kwargs],
    Tuple[Union[Action,AIndex],Score, kwargs],
    Tuple[ActionScore               , kwargs],
]

class Learner(ABC):
    """The Learner interface for contextual bandit learning."""

    @property
    def params(self) -> Mapping[str,Any]: # pragma: no cover
        """Parameters describing the learner (used for descriptive purposes only).

        Remarks:
            These will become columns in the learners table of experiment results.
        """
        return {}

    @abstractmethod
    def predict(self, context: Context, actions: Sequence[Action]) -> Prediction:
        """Predict which action to take in the context.

        Args:
            context: The current context. It will either be None (multi-armed bandit),
                a value (a single feature) a hashable tuple (dense context), or a
                hashable dictionary (sparse context).
            actions: The current set of actions to choose from in the given context.
                Each action will either be a value (a single feature), a hashable tuple
                (dense context), or a hashable dictionary (sparse context)..

        Returns:
            A Prediction. Several prediction formats are supported. See the type-hint for these.
        """
        ...

    @abstractmethod
    def learn(self,
        context: Context,
        actions: Actions,
        action: Union[Action,AIndex],
        feedback: Union[float,Any],
        score: float,
        **kwargs) -> None:
        """Learn about the action taken in the context.

        Args:
            context: The context in which the action was taken.
            actions: The set of actions chosen from.
            action: Either the action that was chosen or the index of the action that was chosen.
            feedback: This will be reward for contextual bandit problems and feedback for IGL problems.
            score: This will be the probability for the action taken if a PMF/PDF is returned by predict.
                 It will be the score if an action-score pair is returned by predict. And it will be the
                 probability if off-policy learning is being performed on LoggedInteractions.
            **kwargs: Optional information returned with the prediction.
        """
        ...

class SafeLearner(Learner):
    """A wrapper for learner-likes that guarantees interface consistency."""

    def __init__(self, learner: Learner, seed:int=1) -> None:
        """Instantiate a SafeLearner.

        Args:
            learner: The learner we wish to make sure has the expected interface
        """

        self._learner     = learner if not isinstance(learner, SafeLearner) else learner._learner
        self._rng         = CobaRandom(seed)
        self._batched     = None
        self._batched_lrn = None
        self._batched_mjr = None
        self._pred_type   = None #1==PDF,2==PMF,3==Action/Score
        self._with_info   = None
        self._learn_type  = None #3==current,2==old with info,1==old without info

    @property
    def full_name(self) -> str:
        """A user-friendly name created from a learner's params for reporting purposes."""

        params = dict(self.params)
        family = params.pop("family")

        if len(params) > 0:
            return f"{family}({','.join(f'{k}={v}' for k,v in params.items())})"
        else:
            return family

    @property
    def params(self) -> Mapping[str, Any]:
        try:
            params = self._learner.params
            params = params if isinstance(params,dict) else params()
        except AttributeError:
            params = {}

        if "family" not in params:
            params["family"] = self._learner.__class__.__name__

        return params

    def predict(self, context: Context, actions: Actions) -> Tuple[Union[AIndex,Action],Score,kwargs]:

        pred      = self._safe_predict(context,actions)
        pred_type = self._pred_type
        batched   = self._batched

        #if not batched everything just works

        #if batched and what we are given are pmfs
            #we need the prediction orderd by batch first

        #if batched and what we are given are actions cores
            #we need the prediction ordered by batch second

        if pred_type is None and not batched: # first call only
            self._determine_pred_format(pred,actions)
            pred_type = self._pred_type

        elif pred_type is None and batched:
            if self._batched_mjr:
                test_pred = pred[0] 
            else:
                test_pred = [p[0] if not isinstance(p,dict) else {k:v[0] for k,v in p.items()} for p in pred]
            self._determine_pred_format(test_pred,actions[0])
            pred_type = self._pred_type

        if not self._with_info:
            pred_info = {}
        elif self._info_dict:
            pred_info = {k:[p[-1][k] for p in pred] for k in pred[0][-1]} if self._batched_mjr else pred[-1]
        else:
            pred_info = {'_': [p[-1] for p in pred]} if batched else {'_': pred[-1]}

        if self._pred_type == 2:
            if not batched:
                pmf = pred[0] if pred_info else pred
                action,score = self._get_pmf_action_score(self._rng,pmf,actions)
            else:
                pmf = [p[0] if pred_info else p for p in pred] if self._batched_mjr else pred[0]
                action,score = list(zip(*map(self._get_pmf_action_score, repeat(self._rng), pmf, actions)))
        else:
            action,score = list(zip(*[p[:2] for p in pred])) if self._batched_mjr else pred[:2]

        if batched:
            return Batch(action), Batch(score), pred_info
        else:
            return action, score, pred_info

    def learn(self, context, actions, action, reward, probability, **kwargs) -> None:

        if self._batched is None:
            self._batched = isinstance(context,Batch)

        if self._batched and self._batched_lrn == False:
            for i in range(len(context)):
                self._safe_learn(context[i],actions[i],action[i],reward[i],probability[i], {k:v[i] for k,v in kwargs.items()})
        else:
            try:
                self._safe_learn(context,actions,action,reward,probability,kwargs)
            except:
                all_failed = False
                try:
                    for i in range(len(context)):
                        self._safe_learn(context[i],actions[i],action[i],reward[i],probability[i], {k:v[i] for k,v in kwargs.items()})
                    self._batched_lrn = False
                except:
                    all_failed = True
                if all_failed: raise

    def _determine_pred_type(self,pred,is_discrete) -> Optional[int]:
        if self._is_type_1(pred): raise CobaException("PDF predictions are currently not supported.")
        if self._is_type_2(pred,is_discrete): return 2
        if self._is_type_3(pred,is_discrete): return 3
        return None

    def _determine_has_info(self,pred,pred_type) -> Mapping[Any,Any]:
        if pred_type == 2:
            return not isinstance(pred[0],(int,float))
        else: #pred_type==3
            return len(pred) == 3

    def _determine_pred_format(self, pred, actions):

        is_discrete = 0 < len(actions) and len(actions) < float('inf')
        pred_type = self._determine_pred_type(pred, is_discrete) or self.get_inferred_type(pred, actions)
        with_info = self._determine_has_info(pred,pred_type)

        self._pred_type = pred_type
        self._with_info = with_info
        self._info_dict = with_info and isinstance(pred[-1],dict) 

    def _safe_predict(self, context, actions):

        batched = self._batched

        if batched is None: 
            batched = isinstance(context,Batch)
            self._batched = batched

        if not batched:
            return self._learner.predict(context,actions)

        if batched:
            batched_lrn = self._batched_lrn
            if batched_lrn == True:
                return self._learner.predict(context,actions)
            elif batched_lrn == False:
                return list(map(self._learner.predict,context,actions))
            elif batched_lrn is None:
                try:
                    batch_size = len(context)
                    pred = self._learner.predict(context,actions)
                    n_rows = len(pred)
                    n_cols = len(pred[0])

                    self._batched_lrn = True

                    if n_rows != n_cols:
                        self._batched_mjr = len(pred) == batch_size
                    else:
                        #The major order of pred is not determinable. So we
                        #now do a small "test" to determine the major order.
                        test_pred = self._learner.predict(Batch([context[0]]),Batch([actions[0]]))
                        self._batched_mjr = len(test_pred) == 1
                    return pred

                except Exception as e:
                    self._batched_lrn = False
                    self._batched_mjr = True
                    try:
                        return list(map(self._learner.predict,context,actions))
                    except:
                        pass
                    raise

    def _safe_learn(self,context,actions,action,reward,probability,kwargs):
        if self._learn_type==3:
            self._learner.learn(context, actions, action, reward, probability, **kwargs)
        elif self._learn_type==2:
            self._learner.learn(context, action, reward, probability, kwargs.get('_',kwargs))
        elif self._learn_type==1:
            self._learner.learn(context, action, reward, probability)
        else:
            all_failed = False
            try:
                self._learner.learn(context, actions, action, reward, probability, **kwargs)
                self._learn_type = 3
            except:
                try:
                    self._learner.learn(context, action, reward, probability, kwargs.get('_',kwargs))
                    self._learn_type = 2
                except:
                    try:
                        self._learner.learn(context, action, reward, probability)
                        self._learn_type = 1
                    except:
                        all_failed = True

                if all_failed: raise

    def _is_type_1(self, pred):
        #PDF
        return callable(pred) or callable(pred[0])

    def _is_type_2(self, pred, is_discrete:bool):
        #PMF

        explicit = isinstance(pred,Probs) or isinstance(pred[0],Probs)
        pmf_sans_info = is_discrete and isinstance(pred,abc.Sequence) and len(pred) > 3 or len(pred) == 3 and not isinstance(pred[2],dict)
        pmf_with_info = is_discrete and isinstance(pred,abc.Sequence) and len(pred) == 2 and isinstance(pred[0],abc.Sequence)
        
        return explicit or pmf_sans_info or pmf_with_info

    def _is_type_3(self, pred, is_discrete:bool):
        #Action Score
        explicit = isinstance(pred,ActionScore) or isinstance(pred[0],ActionScore)
        with_info = is_discrete and len(pred) == 3 and not isinstance(pred[2],(float,int))
        return explicit or with_info or not is_discrete

    def get_inferred_type(self,pred,actions) -> int:
        possible_types = []

        if self._possible_type_2(pred,actions): possible_types.append(2)
        if self._possible_type_3(pred,actions): possible_types.append(3)

        if len(possible_types) == 1:
            return possible_types[0]
        else:
            raise CobaException("We were unable to parse the given prediction format." 
                " This is likely because action features were returned for a discrete"
                " problem. When the action space is discrete, and you wish to directly"
                " return the selected action rather than a PMF, please provide the"
                " action index (i.e., actions.index(action)). Alternatively, this can"
                " also happen for two action problems. In this case we suggest using"
                " coba.learners.Probs or coba.learners.ActionScore to provide explicit"
                " type information (e.g., return coba.learners.Probs([1,0])).")

    def _pred_0_possible_pmf(self,pred,actions):
        possible_pmf = pred[0] if isinstance(pred[0],abc.Sequence) else pred
        return isclose(sum(possible_pmf), 1, abs_tol=.001) and len(possible_pmf) == len(actions)

    def _pred_0_possible_action(self,pred,actions):
        is_discrete = 0 < len(actions) and len(actions) < float('inf')
        return not is_discrete or pred[0] in list(range(len(actions)))

    def _possible_type_2(self,pred,actions):
        #PMF
        return self._pred_0_possible_pmf(pred,actions)

    def _possible_type_3(self,pred,actions):
        #action,score
        correct_shape = len(pred) in [2,3]        
        return self._pred_0_possible_action(pred,actions) and correct_shape

    def _get_pmf_action_score(self,rng,pmf,actions):
        assert len(pmf) == len(actions), "The learner returned an invalid number of probabilities for the actions"
        assert isclose(sum(pmf), 1, abs_tol=.001), "The learner returned a pmf which does not sum to one."
        return rng.choice(list(enumerate(pmf)), pmf)

    def __str__(self) -> str:
        return self.full_name

def requires_hashables(cls:Type[Learner]):

    def make_hashable(item):
        if isinstance(item,Dense): return HashableDense(item)
        if isinstance(item,Sparse): return HashableSparse(item)
        return item

    old_predict = cls.predict
    old_learn   = cls.learn

    def new_predict(self,c,A):
        return old_predict(self,make_hashable(c),list(map(make_hashable,A)))

    def new_learn(self,c,A,a,r,p,**kwargs):
        old_learn(self,make_hashable(c),list(map(make_hashable,A)), make_hashable(a),r,p,**kwargs)

    cls.predict = new_predict
    cls.learn   = new_learn

    return cls
