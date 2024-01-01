from coba.primitives import Learner, Context, Actions, Action, Prob

class MisguidedLearner(Learner):
    """A learner trained on incorrect reward information.

    Remarks:
        This is useful to generate difficult logged data in off-policy experiments.
    """
    def __init__(self, learner: Learner, shifter:float, scaler:float) -> None:
        """Instantiate a MisguidedLearner.

        Args:
            learner: The learner that will be trained using incorrect rewards.
            shifter: A constant that will be added to all reward values during learning.
            scaler: A constant that will be multipled by the reward value during learning.

        Remarks:
            The final reward value that learner will see is shifter+scaler*reward.
        """
        self._learner = learner
        self._shifter = shifter
        self._scaler  = scaler

    @property
    def params(self):
        return {**self._learner.params, 'misguided': [self._shifter,self._scaler]}

    def score(self, context: Context, actions: Actions, action: Action) -> Prob:
        return self._learner.score(context, actions, action)

    def predict(self, context: Context, actions: Actions):
        return self._learner.predict(context, actions)

    def learn(self, context: Context, action: Action, reward: float, probability: float,**kwargs):
        self._learner.learn(context, action, self._shifter + self._scaler*reward, probability,**kwargs)
