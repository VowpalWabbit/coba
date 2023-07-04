
import coba as cb
import itertools as it

import timeit
import time

envs2 = cb.Environments.cache_dir('.coba_cache').from_template('examples/templates/class208.json',n_take=100_000,strict=True)[20:21].chunk()
logs2 = envs2.logged(cb.MisguidedLearner(cb.VowpalEpsilonLearner(),1,-1)).ope_rewards("DR").cache()

ground_truth  = it.product(envs2,[cb.VowpalEpsilonLearner()],[cb.OnPolicyEvaluator()])
first_option  = it.product(logs2,[cb.VowpalEpsilonLearner()],[cb.OnPolicyEvaluator()])
second_option = it.product(logs2,[cb.VowpalEpsilonLearner()],[cb.OffPolicyEvaluator()])
third_option  = it.product(logs2,[cb.VowpalEpsilonLearner()],[cb.ExplorationEvaluator()])

experiment2 = cb.Experiment(it.chain(ground_truth,first_option,second_option,third_option))

experiment2.run()

#result = cb.Result.from_file('./examples/notebooks/out2.log.gz')

#result.plot_learners(l=['ope_reward','eval_type'],p='openml_task',out=None)

#start =time.time()
#result.filter_val(eval_type={'!=':'ExplorationEvaluator'}).plot_learners(l=['ope_reward','eval_type'],p='openml_task',colors=[1,2,3],out=None)
#print(time.time()-start)

# class CycledLearner:

#     def __init__(self, learner: cb.Learner) -> None:
#         self._learner = learner

#     @property
#     def params(self):
#         return self._learner.params

#     def request(self, context, actions, request):
#         return self._learner.request(context, actions, request)

#     def predict(self, context, actions):
#         return self._learner.predict(context, actions)

#     def learn(self, context, actions, action, reward, probability):
#         action = actions[(actions.index(action)+1)%len(actions)]
#         self._learner.learn(context, actions, action, reward, probability)

# if __name__ == "__main__":

#     n_processes = 8
#     n_take      = 4_000

#     envs = cb.Environments.cache_dir('.coba_cache').from_template('./examples/templates/208_multiclass.json',n_take=n_take)
#     logs = envs.logged(cb.VowpalEpsilonLearner()).chunk().ope_rewards([None,'IPS','DM','DR'])

#     result = cb.Experiment(logs, cb.VowpalEpsilonLearner()).run(processes=n_processes)

#     result.filter_fin(n_take).plot_learners(l='ope_reward')
#     result.plot_contrast('None',['IPS','DM','DR'],labels=None,l='ope_reward',x='ope_reward',err='sd',boundary=False,legend=False)
