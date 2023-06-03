import coba as cb
import datetime
import matplotlib.pyplot as plt

class CycledLearner:

    def __init__(self, learner: cb.Learner) -> None:
        self._learner = learner

    @property
    def params(self):
        return self._learner.params

    def request(self, context, actions, request):
        return self._learner.request(context, actions, request)

    def predict(self, context, actions):
        return self._learner.predict(context, actions)

    def learn(self, context, actions, action, reward, probability):
        action = actions[(actions.index(action)+1)%len(actions)]
        self._learner.learn(context, actions, action, reward, probability)

if __name__ == "__main__":

    n_processes = 8
    n_take      = 5_000

    envs      = cb.Environments.cache_dir('.coba_cache').from_template('./examples/templates/208_multiclass.json',n_take=100_000).where(n_interactions=100_000)
    envs_take = envs.take(n_take)

    #Our goal is to use logged data to estimate online exploration performance for a learner
    #For this experiment we will use cb.VowpalEpsilonLearner() as our learner though any could be used...

    # True online exploration performance across our environments
    # Our goal is to determine how well the below methods estimate these values
    print(f"{datetime.datetime.now()} on-policy")
    r = cb.Experiment(envs_take, cb.VowpalEpsilonLearner(), evaluation_task=cb.OnPolicyEvaluation()).run(processes=n_processes,quiet=True)
    r.filter_fin(n_take).plot_learners(labels='on-policy', colors=0, out=None)

    #We create logged data to use for estimating online performance
    #We use a "CycledLearner" so that the logged data doesn't make it too easy to estimate online performance
    logs      = envs.logged(CycledLearner(cb.VowpalEpsilonLearner())).ope_rewards("DM")
    logs_take = envs_take.logged(CycledLearner(cb.VowpalEpsilonLearner())).ope_rewards("DM")

    # Next we want to know how well traditional off-policy learning and evaluation methods match
    # To make the test more fair we intentionally handicap the logging policy by cycling the actions
    # If we didn't do this and the logging policy equaled the off-policy learner off-policy looks perfect
    print(f"{datetime.datetime.now()} off-policy DM")
    r = cb.Experiment(logs_take, cb.VowpalEpsilonLearner(), evaluation_task=cb.OffPolicyEvaluation()).run(processes=n_processes,quiet=True)
    r.filter_fin(n_take).plot_learners(labels='off-policy DM', colors=1, out=None)

    # An alternative method to off-policy learning is to instead use the off policy estimate of a reward as the true reward.
    # That is, perform on-policy analysis using the off-policy estimated reward function.
    print(f"{datetime.datetime.now()} on-policy DM")
    r = cb.Experiment(logs_take, cb.VowpalEpsilonLearner(), evaluation_task=cb.OnPolicyEvaluation()).run(processes=n_processes,quiet=True)
    r.filter_fin(n_take).plot_learners(labels='on-policy DM', colors=2, out=None)

    # Finally we want to see how well the new Exploration Eval method approximates true online performance
    print(f"{datetime.datetime.now()} explore-eval")
    r = cb.Experiment(logs, cb.VowpalEpsilonLearner(), evaluation_task=cb.ExplorationEvaluation()).run(processes=n_processes, quiet=True)
    r.plot_learners(labels='explore-eval OPE',colors=3,out=None)

    print(f"{datetime.datetime.now()} explore-eval")
    r = cb.Experiment(logs, cb.VowpalEpsilonLearner(), evaluation_task=cb.ExplorationEvaluation(ope=False)).run(processes=n_processes, quiet=True)
    r.plot_learners(labels='explore-eval no OPE',colors=4,out=None)

    plt.show()
