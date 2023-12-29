from math import isclose
from collections import abc
from typing import Any, Tuple, Mapping, Literal

from coba.utilities import sample_actions, try_else
from coba.exceptions import CobaException
from coba.random import CobaRandom
from coba.primitives import is_batch, Learner, Context, Action, Actions, Prob, PMF, kwargs, Prediction

def first_row(pred: Prediction, batch_order:Literal['not','row','col'], has_kwargs:bool) -> Tuple[bool,Prediction]:
    if batch_order == 'col':
        if has_kwargs and isinstance(pred[0],dict):
            pred = pred[0]
        if isinstance(pred,dict):
            return {k:v[0] for k,v in pred.items()}
        else:
            row = [p[0] for p in (pred[:-1] if has_kwargs else pred) ]
            return row[0] if len(row) == 1 else row
    else:
        if batch_order == 'row': pred = pred[0]
        return (pred[0] if len(pred)==2 else pred[:-1]) if has_kwargs else pred

def has_kwargs(pred: Prediction, batch_order:Literal['not','row','col']) -> bool:
    try:
        return isinstance(pred[-1] if batch_order != 'row' else pred[0][-1], abc.Mapping)
    except:
        return False

def batch_order(predictor, pred: Prediction, context, actions, method=None) -> Literal['not','col','row']:
    if method == 2: return 'row'
    if not is_batch(actions) and not is_batch(context): return 'not'

    no_len         = lambda item: not hasattr(item,'__len__')
    is_all_dicts   = all(isinstance(p,dict) for p in pred)
    is_dict_col    = isinstance(pred,dict)
    is_dict_col_kw = is_all_dicts and pred[0].keys() != pred[-1].keys() and len(pred)==2
    is_dict_row    = is_all_dicts and pred[0].keys() == pred[-1].keys()

    if is_dict_col or is_dict_col_kw : return 'col'
    if is_dict_row or no_len(pred[0]): return 'row'
    #if isinstance(pred[0],dict)

    n_rows = len(actions)
    n_dim1 = len(pred)
    n_dim2 = len(pred[0])

    if n_dim1 == n_dim2:
        #The major order of pred is not determinable. So we
        #now do a small "test" to determine the major order.
        #n_cols will always be >= 2 so we know we can distinguish
        class Batch(list): is_batch=True
        pred   = predictor(Batch([context[0]]),Batch([actions[0]]))
        n_rows = 1

    return 'row' if len(pred) == n_rows else 'col'

def pred_format(std_pred:Prediction, actions:Actions, og_pred:Prediction = None):
    unclear_format = CobaException("We were unable to determine the prediction format from the "
    f"given value: {og_pred}. To work around this you can provide explicit format information by "
    "returnning a dict wrapper: {'pmf':<pred>}, {'action':<pred>}, or {'action_prob':<pred>}.")
    no_act_two  = CobaException("We were given a two item pred without actions. We cannot tell "
    "if this is an action or action_prob. Please use explicit hints to let us know by returning "
    "either {'action':<pred>} or {'action_prob':<pred>}.")
    no_act_pmf  = CobaException("We were given a PMF but there are no actions to choose from.")
    bad_len_pmf = CobaException("We were given a PMF whose length did match len(actions).")
    bad_len_ap  = lambda ap: CobaException("An explicit action_prob was passed but it is not "
    "a two piece tuple. A valid format has the form (<action>,<prob>). We were given {ap}.")

    no_len = lambda item: not hasattr(item,'__len__')

    #possible std_pred:
        #pmf, action, [action,prob], {'pmf':...}, {'action':...}, {'action_prob':...}

    if isinstance(std_pred, dict):
        if 'pmf' in std_pred:
            pmf = std_pred['pmf']
            if not actions: raise no_act_pmf
            if no_len(pmf) or len(pmf) != len(actions): raise bad_len_pmf
            return 'PM*'
        if 'action' in std_pred:
            return 'AX*'
        if 'action_prob' in std_pred:
            ap = std_pred['action_prob']
            if no_len(ap) or len(ap) != 2:
                raise bad_len_ap(ap)
            return 'AP*'

    if no_len(std_pred) or isinstance(std_pred,str):
        #action
        std_pred = [std_pred]
    elif len(std_pred) > 2:
        #pmf or action
        std_pred = [std_pred]
    elif len(std_pred) == 2:
        #pmf, action or [action,prob]
        if not actions:
            raise no_act_two
        elif any(std_pred[0] is a for a in actions):
            #[action,prob] (this should always be correct due to _safe_actions)
            #when could pred[0] be identified as action but it isn't?
            #see, https://stackoverflow.com/a/306353/1066291
            pass
        else:
            #pmf or action
            std_pred = [std_pred]

    #at this point pred should be
        #[pmf], [action], [action,prob]

    if len(std_pred) == 2:
        #only [action,prob] has two items
        return 'AP'

    #now it is [pmf] or [action]

    if actions == [] or actions is None:
        #a continuous action problem
        #so they had to return action
        return 'AX'

    if any(std_pred[0] is action for action in actions):
        return 'AX'

    if possible_pmf(std_pred[0],actions):
        return 'PM'

    if possible_action(std_pred[0],actions):
        return 'AX'

    raise unclear_format

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

def batch_size(args):
    for arg in args:
        if is_batch(arg):
            return len(arg)

def raise_if_not_valid_out(out,expected_len):
    def len_or_0(obj):
        return try_else(lambda: len(obj), 0)
    if out is None:
        raise CobaException("The given prediction was none and did not match the batch_size.")
    if all(isinstance(p,dict) for p in out) and out[0].keys() != out[-1].keys(): #pragma: no cover
        out = out[0]
    if isinstance(out,dict):
        is_valid = expected_len == len_or_0(next(iter(out.values())))
    else:
        is_valid = expected_len in [len_or_0(out),len_or_0(out[0])]
    if not is_valid:
        raise CobaException("The given prediction did not appear to match the batch_size.")

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

        self._prev_actions = None
        self._safe_actions = None

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

    @property
    def has_score(self) -> bool:
        try:
            self.learner.score(None,None,None)
        except Exception as ex:
            return "score" not in str(ex)
        else:
            return True

    def _method1(self,method,args,kwargs):
        return method(*args,**kwargs)

    def _method2(self,method,args,kwargs):
        pred = [ method(*a,**{k:v[i] for k,v in kwargs.items()}) for i,a in enumerate(zip(*args)) ]
        if not pred:
            raise CobaException(
                f"Something went wrong. No prediction was returned when using batch fallback methods "
                f"on the args: {args}. See the exception above for the reason why fallback methods "
                "were attempted."
            )
        return pred

    def _safe_call(self, key, method, args, kwargs = {}, has_out = True):
        if self._method.get(key) == 1:
            return self._method1(method, args, kwargs)
        elif self._method.get(key) == 2:
            return self._method2(method, args, kwargs)
        elif not any(map(is_batch,args)):
            self._method[key] = 1
            return self._method1(method,args,kwargs)
        else:
            expected_size = batch_size(args)
            try:
                out = self._method1(method,args,kwargs)
                if has_out: raise_if_not_valid_out(out,expected_size)
                self._method[key] = 1
                return out
            except Exception as outer_e:
                try:
                    out = self._method2(method,args,kwargs)
                    if has_out: raise_if_not_valid_out(out,expected_size)
                    self._method[key] = 2
                    return out
                except Exception as inner_e:
                    raise inner_e from outer_e

    def _parse_pred(self, context, actions, pred):
        if self._pred_batch is None: # first call only
            predictor         = lambda X,A: self._safe_call('predict', self.learner.predict, (X,A))
            self._pred_batch  = batch_order(predictor,pred,context,actions,self._method['predict'])
            self._pred_kwargs = has_kwargs(pred,self._pred_batch)
            first_pred_row    = first_row(pred,self._pred_batch,self._pred_kwargs)
            first_actions     = actions[0] if self._pred_batch != 'not' else actions
            self._pred_format = pred_format(first_pred_row, first_actions, pred)

        if self._pred_batch == 'not':
            kwargs = pred[-1] if self._pred_kwargs else {}
            pred   = pred[ 0] if self._pred_kwargs and len(pred)==2 else pred

            if self._pred_format.endswith('*'):
                pred = list(pred.values())[0]

            if self._pred_format[:2] == 'PM':
                a,p = sample_actions(actions, pred, self._rng)

            if self._pred_format[:2] == 'AP':
                a,p = pred[:2]

            if self._pred_format[:2] == 'AX':
                a,p = pred,None

            return a,p,kwargs

        if self._pred_batch == 'row':
            kwargs = [ p[-1] if self._pred_kwargs else {} for p in pred ]
            kwargs = {k: [kw[k] for kw in kwargs] for k in kwargs[0] }
            pred   = [p[0] if len(p)==2 else p[:-1] for p in pred] if self._pred_kwargs else pred

            if self._pred_format.endswith('*'):
                pred = [list(p.values())[0] for p in pred]

            A,P = [],[]
            if self._pred_format[:2] == 'PM':
                A, P = list(map(list, zip(*[sample_actions(a, p, self._rng) for a, p in zip(actions, pred)])))

            if self._pred_format[:2] == 'AX':
                A = pred
                P = [None]*len(pred)

            if self._pred_format[:2] == 'AP':
                A,P = zip(*pred)

            return A,P,kwargs

        if self._pred_batch == 'col':
            kwargs = pred[-1] if self._pred_kwargs else {}
            pred   = pred[:-1] if self._pred_kwargs else pred

            if self._pred_format.endswith('*'):
                pred = list(pred.values())[0]

            if self._pred_format[:2] == 'PM':
                A, P = list(map(list, zip(*[sample_actions(a, p, self._rng) for a, p in zip(actions, pred)])))

            if self._pred_format[:2] == 'AX':
                A = pred
                P = [None]*len(pred)

            if self._pred_format[:2] == 'AP':
                if self._pred_format.endswith('*'):
                    A,P = zip(*pred)
                else:
                    A,P = pred

            return A,P,kwargs

    def score(self, context: Context, actions: Actions, action: Action) -> Prob:
        try:
            return self._safe_call('score', self.learner.score,(context,actions,action))
        except AttributeError as ex:
            if "'score'" in str(ex):
                raise CobaException(("The `score` method is not implemented for this learner."))
            raise

    def predict(self, context: Context, actions: Actions) -> Tuple[Action,Prob,kwargs]:
        #this logic should guarantee that we can differentiate prediction formats
        #it allows us to "is" checks to see if a returned value "is" one of the actions
        if self._prev_actions != actions:
            self._prev_actions = actions
            all_safe = 0 not in actions and 1 not in actions
            make_safe = lambda a: float(a) if a in [0,1] else a
            self._safe_actions = actions if all_safe else [ make_safe(a) for a in actions]

        pred = self._safe_call('predict', self.learner.predict, (context,self._safe_actions))
        return self._parse_pred(context, self._safe_actions, pred)

    def learn(self, context, action, reward, probability, **kwargs) -> None:
        try:
            self._safe_call('learn', self.learner.learn, (context,action,reward,probability), kwargs, has_out=False)
        except TypeError as ex:
            if 'got an unexpected' in str(ex):
                raise CobaException("It appears that learner.predict returned kwargs but learner.learn did not accept them.") from ex
            if 'learn() missing' in str(ex):
                raise CobaException("It appears that learner.learn expected kwargs but learner.predict did not provide any.") from ex
            raise

    def __str__(self) -> str:
        return self.full_name
