import coba as cb

env0 = cb.Environments.from_linear_synthetic(100,n_actions=5,seed=1).binary().shuffle(n=10)
lrns = [cb.RandomLearner(seed=1),cb.RandomLearner(seed=2)]

result = cb.Experiment(env0,lrns).run(quiet=True)

#result.plot_learners(err='bs')
result.plot_learners(err=cb.BootstrapCI(.01,cb.mean))