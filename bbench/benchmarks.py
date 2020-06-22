"""The benchmarks module contains core benchmark functionality and protocols.

This module contains the abstract interface expected for Benchmark implementations. This 
module also contains several Benchmark implementations and Result data transfer class.

Todo:
    * finish docstring for Stats and Results
    * Add more statistics to the Stats class
    * Finish incorporating built in plotting somehow
"""

from abc import ABC, abstractmethod
from typing import Union, Iterable, Sequence, List, Callable, Optional, Tuple, cast
from itertools import islice

from bbench.games import Game, Round
from bbench.solvers import Solver
from bbench.utilities import check_matplotlib_support

class Stats:

    @staticmethod
    def from_values(vals: Sequence[float]) -> "Stats":
        mean = None if len(vals) == 0 else sum(vals)/len(vals)

        return Stats(mean)

    def __init__(self, mean: Optional[float]):
        self._mean = mean

    @property
    def mean(self) -> Optional[float]:
        return self._mean

class Result:

    def __init__(self, observations: Sequence[Tuple[int,int,float]]):
        self._observations = observations
        self._batch_stats  = []
        self._sweep_stats  = []

        iter_curr  = self._observations[0][1]
        iter_count = 0
        iter_mean  = 0.
        prog_count = 0
        prog_mean  = 0.

        # we manually calculate the statistics the first time
        # using online methods so that we only have to pass
        # through all the observations once.

        # first we have to sort to make sure that batch 
        # indexes spread across games are grouped together.

        for observation in sorted(self._observations, key=lambda o: o[1]):

            if(iter_curr != observation[1]):
                self._batch_stats.append(Stats(iter_mean))
                self._sweep_stats.append(Stats(prog_mean))

                iter_curr  = observation[1]
                iter_count = 0
                iter_mean  = 0

            iter_count += 1
            prog_count += 1

            iter_mean = (1/iter_count) * observation[2] + (1-1/iter_count) * iter_mean
            prog_mean = (1/prog_count) * observation[2] + (1-1/prog_count) * prog_mean

        self._batch_stats.append(Stats(iter_mean))
        self._sweep_stats.append(Stats(prog_mean))

    @property
    def batch_stats(self) -> Sequence[Stats]:
        return self._batch_stats

    @property
    def sweep_stats(self) -> Sequence[Stats]:
        return self._sweep_stats

    def predicate_stats(self, predicate: Callable[[Tuple[int,int,float]],bool]) -> Stats:
        return Stats.from_values([o[2] for o in filter(predicate, self._observations)])

    @property
    def observations(self) -> Sequence[Tuple[int,int,float]]:
        return self._observations

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
        Results are unbiased only if the sequence of rounds in a given game are stationary.
        This doesn't mean that a game can't be static, it simply means that there should
        be a uniform random shuffling of all rounds performed at least once on a game. Once
        such a shuffling has been done then a game can be fixed for all benchmarks after that.
    """

    def __init__(self, 
        games: Sequence[Game], 
        n_game_rounds: Optional[int], 
        n_batch_rounds: Union[int, Callable[[int],int]]) -> None:
        
        self._games          = games
        self._n_game_rounds  = n_game_rounds
        
        if isinstance(n_batch_rounds, int):
            self._n_batch_rounds = lambda i: n_batch_rounds
        else:
            self._n_batch_rounds = n_batch_rounds

    def evaluate(self, solver_factory: Callable[[],Solver]) -> Result:
        """Collect observations of a Solver playing the benchmark's games to create Results.

        Args:
            solver_factory: See the base class for more information.
        
        Returns:
            See the base class for more information.
        """

        results:List[Tuple[int,int,float]] = []

        for game_index, game in enumerate(self._games):

            game_solver   = solver_factory()
            batch_index   = 0
            batch_samples = []

            for r in islice(game.rounds, self._n_game_rounds):

                choice = game_solver.choose(r.state, r.actions) 
                state  = r.state
                action = r.actions[choice]
                reward = r.rewards[choice]

                batch_samples.append((state, action, reward))

                if len(batch_samples) == self._n_batch_rounds(batch_index):

                    for (state,action,reward) in batch_samples:
                        game_solver.learn(state,action,reward)
                        results.append((game_index, batch_index, reward))

                    batch_samples = []
                    batch_index += 1
                    
            for (state,action,reward) in batch_samples:
                game_solver.learn(state,action,reward)
                results.append((game_index, batch_index, reward))

        return Result(results)