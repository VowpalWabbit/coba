"""The expected interface for all learner implementations."""

from numbers import Number
from typing import Any, Sequence, Dict, Tuple, Optional

from coba.environments import Context, Action
from coba.learners.primitives import Learner

Info = Any
Probs = Sequence[float]

class SafeLearner(Learner):

        @property
        def params(self) -> Dict[str, Any]:
            try:
                params = self._learner.params
            except AttributeError:
                params = {}

            if "family" not in params:
                params["family"] = self._learner.__class__.__name__

            return params

        @property
        def full_name(self) -> str:
            params = dict(self.params)
            family = params.pop("family")

            if len(params) > 0:
                return f"{family}({','.join(f'{k}={v}' for k,v in params.items())})"
            else:
                return family

        def __init__(self, learner: Learner) -> None:
            
            self._learner = learner if not isinstance(learner, SafeLearner) else learner._learner

        def predict(self, context: Context, actions: Sequence[Action]) -> Tuple[Probs, Info]:
            predict = self._learner.predict(context, actions)

            predict_has_no_info = len(predict) != 2 or isinstance(predict[0],Number)

            if predict_has_no_info:
                info    = None
                predict = predict
            else:
                info    = predict[1]
                predict = predict[0]

            assert len(predict) == len(actions), "The learner returned an invalid number of probabilities for the actions"
            assert round(sum(predict),2) == 1 , "The learner returned a pmf which didn't sum to one."

            return (predict,info)

        def learn(self, context: Context, action: Action, reward: float, probability:float, info: Info) -> Optional[Dict[str,Any]]:
            return self._learner.learn(context, action, reward, probability, info) or {}

        def __repr__(self) -> str:
            return self.full_name