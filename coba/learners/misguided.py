from coba.learners.primitives import Learner

class MisguidedLearner(Learner):
    """A contextual bandit learner that is given incorrect reward information. It is
    useful when generating logged data that is difficult in off-policy experiments.
    """
    def __init__(self, learner: Learner, shifter:float, scaler:float) -> None:
        self._learner = learner
        self._shifter = shifter
        self._scaler  = scaler

    @property
    def params(self):
        return {**self._learner.params, 'misguided': [self._shifter,self._scaler]}

    def request(self, context, actions, request):
        return self._learner.request(context, actions, request)

    def predict(self, context, actions):
        return self._learner.predict(context, actions)

    def learn(self, context, actions, action, reward, probability):
        self._learner.learn(context, actions, action, self._shifter + self._scaler*reward, probability)
