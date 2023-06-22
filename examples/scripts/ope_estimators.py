"""
This is an example script that creates and execuates an Experiment.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

import coba as cb
import datetime
import matplotlib.pyplot as plt

if __name__ == "__main__":

    n_processes = 8
    n_take      = 100_000

    #envs = cb.Environments.cache_dir('.coba_cache').from_template('./examples/templates/208_multiclass.json',n_take=n_take).where(n_interactions=n_take)
    envs = cb.Environments.cache_dir('.coba_cache').from_openml(150,take=n_take).where(n_interactions=n_take)
    logs = envs.logged(cb.VowpalEpsilonLearner())

    print(f"{datetime.datetime.now()} true")
    r = cb.Experiment(logs, cb.VowpalOffPolicyLearner(), evaluation_task=cb.OffPolicyEvaluation()).run(processes=n_processes,quiet=True)
    r.filter_fin(n_take).plot_learners(labels='true OPE', colors=0, out=None)

    print(f"{datetime.datetime.now()} IPS")
    r = cb.Experiment(logs.ope_rewards('IPS'), cb.VowpalOffPolicyLearner(), evaluation_task=cb.OffPolicyEvaluation()).run(processes=n_processes,quiet=True)
    r.filter_fin(n_take).plot_learners(labels='IPS OPE' , colors=1, out=None)

    print(f"{datetime.datetime.now()} DM")
    r = cb.Experiment(logs.ope_rewards('DM'), cb.VowpalOffPolicyLearner(), evaluation_task=cb.OffPolicyEvaluation()).run(processes=n_processes,quiet=True)
    r.filter_fin(n_take).plot_learners(labels='DM OPE'  , colors=2, out=None)

    print(f"{datetime.datetime.now()} DR")
    r = cb.Experiment(logs.ope_rewards('DR'), cb.VowpalOffPolicyLearner(), evaluation_task=cb.OffPolicyEvaluation()).run(processes=n_processes,quiet=True)
    r.filter_fin(n_take).plot_learners(labels='DR OPE'  , colors=3, out=None)

    plt.show()
