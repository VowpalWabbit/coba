from collections import abc
from math import isclose
from abc import ABC, abstractmethod
from typing import Any, Sequence, Union, Tuple, Callable, Mapping, Optional

from coba.exceptions import CobaException
from coba.random import CobaRandom
from coba.environments import Context, Action, Actions

kwargs = Mapping[str,Any]
Score  = float
PMF    = Sequence[float]
PDF    = Callable[[Action],float]

class Probs(list):
    pass

class ActionScore(tuple):
    def __new__(self, action: Action, score: Score):
        return tuple.__new__(ActionScore, (action, score))

Prediction = Union[
    PMF,
    PDF,
    Probs,
    ActionScore,
    Tuple[PMF         , kwargs],
    Tuple[PDF         , kwargs],
    Tuple[Action,Score, kwargs],
    Tuple[ActionScore , kwargs],
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
        action: Action, 
        feedback: Union[float,Any], 
        score: float, 
        **kwargs) -> None:
        """Learn about the action taken in the context.

        Args:
            context: The context in which the action was taken.
            actions: The set of actions chosen from.
            action: The action that was chosen.
            feedback: This will be reward for contextual bandit problems and feedback for IGL problems.
            score: This will be the probability for the action taken if a PMF/PDF is returned by predict.
                 It will be the score if an action-score pair is returned by predict. And it will be the
                 probability if off-policy learning is being performed on LoggedInteractions.
            **kwargs: Optional information returned with the prediction.
        """
        ...

class SafeLearner(Learner):
    """A wrapper for learner-likes that guarantees interface consistency."""

    def __init__(self, learner: Learner) -> None:
        """Instantiate a SafeLearner.

        Args:
            learner: The learner we wish to make sure has the expected interface
        """

        self._learner    = learner if not isinstance(learner, SafeLearner) else learner._learner
        self._rng        = CobaRandom(1)
        self._pred_type  = None #1==PDF,2==PMF,3==Action/Score
        self._with_info  = None
        self._learn_type = None #3==current,2==old with info,1==old without info

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

    def predict(self, context: Context, actions: Actions) -> Tuple[Action,Score,kwargs]:

        pred = self._learner.predict(context, actions)

        if self._pred_type is None: # first call
            pred_type = self.get_definite_type(pred) or self.get_inferred_type(pred,actions)
            pred_info = self.get_info(pred,pred_type)
            self._pred_type = pred_type
            self._with_info = bool(pred_info)
        else:
            pred_type = self._pred_type
            pred_info = pred[-1] if self._with_info else {}

        if self._pred_type == 1 or self._pred_type == 2:
            pmf_or_pdf = pred[0] if pred_info else pred
            pmf        = list(map(pred,actions)) if pred_type == 1 else pmf_or_pdf
            assert len(pmf) == len(actions), "The learner returned an invalid number of probabilities for the actions"
            assert isclose(sum(pmf), 1, abs_tol=.001), "The learner returned a pmf which didn't sum to one."
            
            action,score = self._rng.choice(list(zip(actions,pmf)), pmf)
        else:
            action,score = pred[:2]

        return action, score, {'info':pred_info}

    def learn(self, context, actions, action, reward, probability, info) -> None:
        if self._learn_type==3:
            self._learner.learn(context, actions, action, reward, probability, **(info or {}))
        elif self._learn_type==2:
            self._learner.learn(context, action, reward, probability, info)
        elif self._learn_type==1:
            self._learner.learn(context, action, reward, probability)
        else:
            all_failed = False
            try:
                self._learner.learn(context, actions, action, reward, probability, **info)
                self._learn_type = 3
            except:
                try:
                    self._learner.learn(context, action, reward, probability, info)
                    self._learn_type = 2
                except:
                    try:
                        self._learner.learn(context, action, reward, probability)
                        self._learn_type = 1
                    except:
                        all_failed = True
                
                if all_failed: raise

    def get_definite_type(self,pred) -> Optional[int]:
        if self._is_type_1(pred): return 1
        if self._is_type_2(pred): return 2
        if self._is_type_3(pred): return 3
        return None

    def get_inferred_type(self,pred,actions) -> int:
        possible_types = []

        if self._possible_type_2(pred,actions): possible_types.append(2)
        if self._possible_type_3(pred,actions): possible_types.append(3)

        if len(possible_types) == 1:
            return possible_types[0]
        else:
            raise CobaException("We were unable to parse the given prediction format." 
                " If prediction format returned by your learner is definitely correct"
                " then we suggest using coba.learners.Probs or coba.learners.ActionScore"
                " to provide explicit type information.")

    def get_info(self,pred,pred_type) -> Mapping[Any,Any]:
        if pred_type == 1:
            return pred[1] if isinstance(pred,abc.Sequence) and len(pred) == 2 else {}        
        elif pred_type == 2:
            return pred[1] if not isinstance(pred[0],(int,float)) else {}
        else: #pred_type==3
            return pred[2] if len(pred) == 3 else {}

    def _is_type_1(self, pred):
        #PDF
        return callable(pred) or callable(pred[0])

    def _is_type_2(self, pred):
        #PMF
        explicit = isinstance(pred,Probs) or isinstance(pred[0],Probs)
        too_long_for_anything_else = len(pred) > 3
        return explicit or too_long_for_anything_else

    def _is_type_3(self, pred):
        #Action Score
        explicit = isinstance(pred,ActionScore) or isinstance(pred[0],ActionScore)
        return explicit

    def _pred_0_possible_pmf(self,pred,actions):
        try:
            possible_pmf = pred[0] if isinstance(pred[0],abc.Sequence) else pred
            return isclose(sum(possible_pmf), 1, abs_tol=.001) and len(possible_pmf) == len(actions)
        except:
            return False

    def _pred_0_possible_action(self,pred,actions):
        return not actions or pred in actions or pred[0] in actions

    def _possible_type_2(self,pred,actions):
        #PMF
        return self._pred_0_possible_pmf(pred,actions)

    def _possible_type_3(self,pred,actions):
        #action,score
        correct_shape = len(pred)==2 or (len(pred)==3 and isinstance(pred[-1],dict))
        return self._pred_0_possible_action(pred,actions) and correct_shape

    def __str__(self) -> str:
        return self.full_name
