"""

"""

from abc import ABC, abstractmethod
from typing import Iterable, Sequence, List, Callable, Optional, Tuple, cast
from itertools import islice

from bbench.games import Game, Round
from bbench.solvers import Solver


class Result:

    @staticmethod
    def avg(vals: Sequence[float]) -> float:
        return sum(vals)/len(vals)

    @staticmethod
    def var(vals: Sequence[float]) -> Optional[float]:
        if(len(vals) == 1):
            return None
        
        avg = Result.avg(vals)
        return sum([(val-avg)**2 for val in vals])/len(vals)

    @staticmethod
    def sem(vals: Sequence[float]) -> Optional[float]:
        if(len(vals) == 1):
            return None

        var = cast(float,Result.var(vals))
        return (var/len(vals))**(1/2)

    def __init__(self, samples_by_x: Sequence[Sequence[float]]) -> None:
        averages = list(map(Result.avg, samples_by_x))

        self._points   = list(enumerate(averages,1))
        self._errors   = list(map(Result.sem,samples_by_x))

    @property
    def points(self) -> Sequence[Tuple[float,float]]:
        return self._points

    def print(self) -> 'Result':
        return self
    
    def plot(self) -> 'Result':
        return self

class Benchmark(ABC):

    @abstractmethod
    def evaluate(self, solver: Solver) -> Result:
        pass

class ProgressiveBenchmark(Benchmark):
    def __init__(self, games: Iterable[Game], n_rounds: int) -> None:
        self._games = games
        self._n_rounds = n_rounds

    def evalute(self, solver_factory: Callable[[],Solver]):
        round_pvrs: List[List[float]] = [ [] for i in range(self._n_rounds) ]

        for game in self._games:
            solver = solver_factory()
            progressive_reward = 0.0

            for n,r in enumerate(islice(game.rounds, self._n_rounds)):

                choice = solver.choose(r.state, r.actions)                
                state  = r.state
                action = r.actions[choice]
                reward = r.rewards[choice]

                solver.learn(state, action, reward)

                progressive_reward = 1/(n+1) * reward + (n/n+1) * reward

                round_pvrs[n].append(progressive_reward)

        return Result(round_pvrs)


class IterationGroupBenchmark(Benchmark):
    def __init__(self, games: Iterable[Game],  n_iterations: int, n_rounds: int) -> None:
        self._games = games
        self._n_iterations = n_iterations
        self._n_rounds     = n_rounds

    def evalute(self, solver_factory: Callable[[],Solver]):
        iteration_rwds: List[List[float]] = [ [] for i in range(self._n_iterations) ]

        for game in self._games:
            solver = solver_factory()

            for i in range(self._n_iterations):

                rounds  = islice(game.rounds, self._n_rounds)
                
                choices = [ solver.choose(r.state, r.actions) for r in rounds ]
                states  = [ r.state for r in rounds ]
                actions = [ r.actions[c] for r,c in zip(rounds,choices)]
                rewards = [ r.rewards[c] for r,c in zip(rounds,choices)]

                iteration_rwds[i].extend(rewards)

                for s,a,r in zip(states,actions,rewards):
                    solver.learn(s,a,r)

        return Result(iteration_rwds)