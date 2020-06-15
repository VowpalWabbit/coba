from random import uniform

from bbench.games import LambdaGame
from bbench.solvers import RandomSolver, EpsilonAverageSolver
from bbench.benchmarks import UniversalBenchmark

#define a game
game = LambdaGame(lambda i: None, lambda s: [0,1,2,3,4], lambda s,a: uniform(a-2, a+2))

#create three different solvers
randomsolver_factory = lambda: RandomSolver()
averagesolver_factory1 = lambda: EpsilonAverageSolver(1/1000, lambda a: 0)
averagesolver_factory2 = lambda: EpsilonAverageSolver(1/1000, lambda a: 10)

#define a benchmark
benchmark = UniversalBenchmark([game], lambda i: 10, 10)

#benchmark all three solvers
random_result   = benchmark.evaluate(randomsolver_factory)
average_result1 = benchmark.evaluate(averagesolver_factory1)
average_result2 = benchmark.evaluate(averagesolver_factory2)

#print their mean rewards by iteration
print("random means by iteration")
print([ i.mean for i in random_result.iteration_stats])

print("average non-optimistic epsilon means by iteration with epsilon = 1/1000")
print([ i.mean for i in average_result1.iteration_stats])

print("average optimistic epsilon means by iteration with epsilon = 1/1000")
print([ i.mean for i in average_result2.iteration_stats])