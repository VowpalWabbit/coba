"""

"""

from abc import ABC, abstractmethod
from typing import Iterable, Callable
from itertools import islice

from bbench.games import Game, Round
from bbench.solvers import Solver


class Result(ABC):

    @abstractmethod
    def print(self) -> 'Result':
        return self
    
    @abstractmethod
    def plot(self) -> 'Result':
        return self

class Benchmark(ABC):

    @abstractmethod
    def evaluate(self, solver: Solver) -> Result:
        pass

class IterationBenchmark(Benchmark):
    def __init__(self, games: Iterable[Game],  n_iterations: int, n_rounds: int) -> None:
        self._games = games
        self._n_iterations = n_iterations
        self._n_rounds     = n_rounds

    def evalute(self, solver_factory: Callable[Solver]):
        r_by_i = [ [] for i in range(self._n_iterations) ]

        for game in self._games:
            solver = solver_factory()

            for i in range(self._n_iterations):

                rounds  = islice(game.rounds(), self._n_rounds)
                states  = [ r.state for r in rounds ]
                choices = list(self.play_rounds(rounds, solver))
                rewards = [ r.rewards[c] for r,c in zip(rounds,choices)]

                r_by_i[i].append(rewards)

                for s,a,r in zip(states,choices,rewards):
                    solver.Learn(s,a,r)

    def play_rounds(self, rounds: Iterable[Round], solver: Solver) -> int:
        for _round in rounds:
            yield solver.Choose(_round.state, _round.actions)