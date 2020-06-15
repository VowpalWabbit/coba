"""The benchmarks module contains core benchmark functionality and protocols.

This module contains the abstract interface expected for Benchmark implementations. This 
module also contains several Benchmark implementations and Result data transfer class.

Todo:
    * Consider refactoring Result so that samples are the core of the class rather 
        than values and errors. This would allow for separate properties for many 
        statistics of interest (e.g., means, medians, percentiles, SE), supporting 
        more flexible reporting capabilities.
"""

from abc import ABC, abstractmethod
from typing import Union, Iterable, Sequence, Collection, List, Callable, Optional, Tuple, cast
from itertools import islice

from bbench.games import Game, Round
from bbench.solvers import Solver

class Stats:
    def __init__(self, values: Collection[float]):
        self._values = values
        self._mean = None if len(self._values) == 0 else sum(self._values)/len(self._values)

    @property
    def mean(self) -> Optional[float]:
        return self._mean

class Result:

    def __init__(self, observations: Collection[Tuple[int,int,float]]):
        self._observations      = observations
        self._iteration_ids     = list(set( o[1] for o in self._observations ))
        self._iteration_stats   = [ self.predicate_stats(lambda o: o[1] == i) for i in self._iteration_ids ]
        self._progressive_stats = [ self.predicate_stats(lambda o: o[1] <= i) for i in self._iteration_ids ]

    @property
    def iteration_stats(self) -> Sequence[Stats]:
        return self._iteration_stats
    
    @property
    def progressive_stats(self) -> Sequence[Stats]:
        return self._progressive_stats

    @property
    def observations(self):
        return self._observations

    def predicate_stats(self, predicate: Callable[[Tuple[int,int,float]],bool]):
        return Stats([o[2] for o in filter(predicate, self._observations)])

class Benchmark(ABC):
    """The interface for Benchmark implementations."""
    
    @abstractmethod
    def evaluate(self, solver_factory: Callable[[],Solver]) -> Result:
        """Calculate the performance for a provided bandit Solver.

        Args:
            solver_factory: A function to create Solver instances. The function should 
                always create the same Solver in order to get an unbiased performance 
                Result. This method can be as simple as `lambda: My_Solver(...)`.

        Returns:
            A Result containing the performance statistics of the benchmark.


        Remarks:
            The solver factory is necessary because a Result can be calculated using
            observed performance over several games. In these cases the easiest way to 
            reset a Solver's learned state is to create a new one.
        """
        ...

class UniversalBenchmark(Benchmark):
    """An on-policy Benchmark using unbiased samples to estimate performance statistics.

        Remarks:
            Samples are unbiased only if the sequence of rounds in a given game are unbiased.
            This doesn't mean that a game can't be static, it simply means that there should
            be a uniform random shuffling of all rounds performed at least once on a game. Once
            such a shuffling has been done then a game can be fixed for all benchmarks after that.
    """

    def __init__(self, games: Collection[Game], n_rounds: Callable[[int],int], n_iterations: int):
        self._games = games
        self._n_rounds = n_rounds
        self._n_iterations = n_iterations

    def evaluate(self, solver_factory: Callable[[],Solver]) -> Result:
        """Collect observations of a Solver playing the benchmark's games to create Results.

        Args:
            solver_factory: See the base class for more information.
        
        Returns:
            See the base class for more information.
        """

        results:List[Tuple[int,int,float]] = []

        for game_idx, game in enumerate(self._games):
            solver = solver_factory()
            
            # make sure game rounds iterator stays constant
            # so that rounds don't restart on each iteration
            game_rounds = game.rounds

            for iteration_idx in range(self._n_iterations):

                rounds  = list(islice(game_rounds, self._n_rounds(iteration_idx)))

                choices = [ solver.choose(r.state, r.actions) for r in rounds ]
                states  = [ r.state for r in rounds ]
                actions = [ r.actions[c] for r,c in zip(rounds,choices)]
                rewards = [ r.rewards[c] for r,c in zip(rounds,choices)]

                results.extend((game_idx, iteration_idx, reward) for reward in rewards)

                for s,a,r in zip(states,actions,rewards):
                    solver.learn(s,a,r)

        return Result(results)