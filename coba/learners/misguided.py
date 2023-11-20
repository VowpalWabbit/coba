from coba.learners.primitives import Learner

class MisguidedLearner(Learner):
    """A contextual bandit learner that is given incorrect reward information. It is
    useful when generating difficult logged data in off-policy experiments.
    """
    def __init__(self, learner: Learner, shifter:float, scaler:float) -> None:
        self._learner = learner
        self._shifter = shifter
        self._scaler  = scaler

    @property
    def params(self):
        return {**self._learner.params, 'misguided': [self._shifter,self._scaler]}

    def score(self, context, actions, action = None):
        return self._learner.score(context, actions, action)

    def predict(self, context, actions):
        return self._learner.predict(context, actions)

    def learn(self, context, action, reward, probability,**kwargs):
        self._learner.learn(context, action, self._shifter + self._scaler*reward, probability,**kwargs)
