from collections import abc
from math import isclose
from itertools import repeat
from typing import Any, Sequence, Tuple, Mapping
from coba.backports import Literal

from coba.exceptions import CobaException
from coba.random import CobaRandom
from coba.primitives import Batch, Context, Action, Actions
from coba.learners.primitives import Learner, PMF, ActionProb, Actions, Prob, kwargs, Prediction

def pred_format(pred:Prediction, batch_order:Literal['not','row','col'], has_kwarg:bool, actions:Actions):

    #allowed:
        #PMF,
        #ActionProb,
        #Tuple[PMF        , kwargs]
        #Tuple[ActionProb , kwargs]
        #Tuple[Action,Prob, kwargs]

    if batch_order == 'row':
        pred = pred[0]

    if batch_order == 'col' and not has_kwarg:
        pred = [p[0] for p in pred]

    if batch_order == 'col' and has_kwarg:
        pred = [p[0] for p in pred[:-1]] + [{k:v[0] for k,v in pred[-1].items()}]

    if batch_order !='not':
        actions = actions[0]

    if has_kwarg:
        pred = pred[0] if len(pred)==2 else pred[:-1]

    #at this point pred should be flattened, unbatched and not have kwargs so it is
        #PMF
        #ActionProb
        #Tuple[Prob,...]
        #Tuple[Action,Prob]

    if isinstance(pred,PMF):
        return 'PM'

    if isinstance(pred,ActionProb):
        return 'AP'

    if len(pred) == len(actions) and len(actions) != 2:
        return 'PM'

    _possible_pmf = possible_pmf(pred,actions)
    _possible_act = possible_action(pred[0],actions)

    if _possible_pmf and not _possible_act and len(pred) == len(actions):
        return 'PM'

    if _possible_act and not _possible_pmf and len(pred) == 2:
        return 'AP'

    raise CobaException("We were unable to parse the given prediction format."
        " This is likely because action features were returned for a discrete"
        " problem. When the action space is discrete, and you wish to directly"
        " return the selected action rather than a PMF, please provide the"
        " action index (i.e., actions.index(action)). Alternatively, this can"
        " also happen for two action problems. In this case we suggest using"
        " coba.learners.PMF or coba.learners.ActionProb to provide explicit"
        " type information (e.g., return coba.learners.PMF([1,0])).")

def batch_order(predictor, pred: Prediction, context, actions) -> Literal['not','col','row']:

    if not isinstance(actions,Batch) or not isinstance(context,Batch): return 'not'

    n_rows = len(actions)
    n_dim1 = len(pred)
    n_dim2 = len(pred[0])

    if n_dim1 == n_dim2:
        #The major order of pred is not determinable. So we
        #now do a small "test" to determine the major order.
        #n_cols will always be >= 2 so we know we can distinguish
        pred   = predictor(Batch([context[0]]),Batch([actions[0]]))
        n_rows = 1

    return 'row' if len(pred) == n_rows else 'col'

def possible_pmf(item, actions):
    try:
        return len(item)==len(actions) and isclose(sum(item), 1, abs_tol=.001) and all(i >=0 for i in item)
    except:
        return False

def possible_action(item, actions):
    try:
        return item in actions or len(actions) == 0
    except:
        return False

class SafeLearner(Learner):
    """A wrapper for learner-likes that guarantees interface consistency."""

    def __init__(self, learner: Learner, seed:int=1) -> None:
        """Instantiate a SafeLearner.

        Args:
            learner: The learner we wish to make sure has the expected interface
        """

        self.learner = learner if not isinstance(learner, SafeLearner) else learner.learner
        self._rng    = CobaRandom(seed)
        self._method = {}

        self._pred_kwargs = None
        self._pred_batch  = None
        self._pred_format = None

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
            params = self.learner.params
            params = params if not callable(params) else params()               
            params = params if isinstance(params,dict) else {'params':str(params)}
        except AttributeError:
            params = {}

        if "family" not in params:
            params["family"] = self.learner.__class__.__name__

        return params

    def _safe_call(self, key, method, args, kwargs = {}):

        if key in self._method:
            prev_method = self._method[key]

            if prev_method==1:
                return method(*args,**(kwargs or {}))

            if prev_method==2:
                return [ method(*a,**{k:v[i] for k,v in kwargs.items()}) for i,a in enumerate(zip(*args)) ]

        try:
            self._method[key] = 1
            return self._safe_call(key, method, args, kwargs)
        except:
            try:
                self._method[key] = 2
                return self._safe_call(key, method, args, kwargs)
            except:
                del self._method[key]
            raise

    def request(self, context: Context, actions: Actions, request: Actions) -> Sequence[Prob]:

        try:
            return self._safe_call('request', self.learner.request, (context,actions,request))
        except AttributeError as ex:
            if "'request'" in str(ex):
                raise CobaException(("The `request` method is not implemented for this learner."))
            raise

    def predict(self, context: Context, actions: Actions) -> Tuple[Action,Prob,kwargs]:

        pred = self._safe_call('predict', self.learner.predict, (context,actions))

        if self._pred_batch is None: # first call only
            predictor = lambda X,A: self._safe_call('predict', self.learner.predict, (X,A))

            self._pred_batch  = batch_order(predictor,pred,context,actions)
            self._pred_kwargs = isinstance(pred[-1] if self._pred_batch != 'row' else pred[0][-1],abc.Mapping)
            self._pred_format = pred_format(pred, self._pred_batch, self._pred_kwargs, actions)

        if self._pred_batch == 'not':
            kwargs = pred[-1] if self._pred_kwargs else {}
            pred   = pred[ 0] if self._pred_kwargs and len(pred)==2 else pred

            if self._pred_format == 'PM':
                i = self._get_pmf_index(pred)
                a,p = actions[i], pred[i]

            if self._pred_format == 'AP':
                a,p = pred[0],pred[1]

            return a,p,kwargs

        if self._pred_batch == 'row':
            kwargs = [ p[-1] if self._pred_kwargs else {} for p in pred ]
            kwargs = {k: [kw[k] for kw in kwargs] for k in kwargs[0] }
            pred   = [p[0] if len(p)==2 else p[:-1] for p in pred] if self._pred_kwargs else pred

            A,P = [],[]
            if self._pred_format == 'PM':
                I = [self._get_pmf_index(p) for p in pred]
                A = [ a[i] for a,i in zip(actions,I) ]
                P = [ p[i] for p,i in zip(pred,I) ]

            if self._pred_format == 'AP':
                A = [p[0] for p in pred]
                P = [p[1] for p in pred]

            return A,P,kwargs

        if self._pred_batch == 'col':
            kwargs = pred[-1] if self._pred_kwargs else {}
            pred   = pred[:-1] if self._pred_kwargs else pred

            if self._pred_format == 'PM':
                I = [self._get_pmf_index(p) for p in zip(*pred)]
                A = [ a[i] for a,i in zip(actions,I) ]
                P = [ p[i] for p,i in zip(pred,I) ]

            if self._pred_format == 'AP':
                A = pred[0]
                P = pred[1]

            return A,P,kwargs

    def learn(self, context, actions, action, reward, probability, **kwargs) -> None:

        try:
            self._safe_call('learn', self.learner.learn, (context,actions,action,reward,probability), kwargs)
        except TypeError as ex:
            if 'got an unexpected' in str(ex):
                raise CobaException("It appears that learner.predict returned kwargs but learner.learn did not accept them.") from ex
            if 'learn() missing' in str(ex):
                raise CobaException("It appears that learner.predict expected kwargs but learner.predict did not provide any.") from ex
            raise

    def _get_pmf_index(self,pmf):
        return self._rng.choice(range(len(pmf)), pmf)

    def __str__(self) -> str:
        return self.full_name
