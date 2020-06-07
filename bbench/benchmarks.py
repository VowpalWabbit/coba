"""The benchmarks module containing all core benchmark functionality and protocols.

This module contains an abstract base class representing the expected interface for
all Benchmark implementations. Several common Benchmarch protocols and data transfer 
objects to represent and pass the result of any Benchmark.

Todo:
    * Consider refactoring Result so that samples are the core of the class rather 
        than values and errors. This would allow for separate properties for many 
        statistics of interest (e.g., means, medians, percentiles, SE), supporting 
        more flexible reporting capabilities.
"""

from abc import ABC, abstractmethod
from typing import Union, Iterable, Sequence, List, Callable, Optional, Tuple, cast
from itertools import islice

from bbench.games import Game, Round
from bbench.solvers import Solver

class Result:
    """A class to contain a sequence of values along with their optional error amount."""

    @staticmethod
    def from_samples(samples: Sequence[Union[float,Sequence[float]]]) -> 'Result':
        """Create Result from samples using mean and standard error.

        Args:
            samples: A collection of samples used to estimate mean and standard errors.
        
        Returns:
          A Result with values equal to sample mean and errors equal to standard error.
        """

        safe_samples = list(map(Result._ensure_list, samples))

        values = list(map(Result._avg, safe_samples))
        errors = list(map(Result._sem, safe_samples))

        return Result(values, errors)

    def __init__(self, values:Sequence[float], errors: Sequence[Optional[float]]) -> None:
        """Initialize Result.

        Args:
            values: The values of the result (e.g., mean performance).
            errors: Estimated errors in the values (e.g., standard error).

        Remarks:
            The length of `values` and `errors` are assumed to always be equal.
        """
        self._values = values
        self._errors = errors

    @property
    def values(self) -> Sequence[float]:
        """A sequence of values returned by a Benchmark (often mean performance)."""
        return self._values
    
    @property
    def errors(self) -> Sequence[Optional[float]]:
        """An optional sequence of errors returned by a Benchmark (often standard error)."""
        return self._errors

    @staticmethod
    def _avg(vals: Sequence[float]) -> float:
        return sum(vals)/len(vals)

    @staticmethod
    def _var(vals: Sequence[float]) -> Optional[float]:
        if(len(vals) == 1):
            return None

        avg = Result._avg(vals)
        return sum([(val-avg)**2 for val in vals])/len(vals)

    @staticmethod
    def _sem(vals: Sequence[float]) -> Optional[float]:
        if(len(vals) == 1):
            return None

        var = cast(float,Result._var(vals))
        return (var/len(vals))**(1/2)

    @staticmethod
    def _ensure_list(vals:Union[float,Sequence[float]]) -> Sequence[float]:
        if isinstance(vals, (float,int)):
            return [vals]
        else:
            return vals


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

class ProgressiveBenchmark(Benchmark):
    """An on-policy Benchmark measuring progressive validation gain (Blum et al., 1999).

        Remarks:
            To receive errors from this benchmark there must be more than one game.

        References:
            A. Blum, A. Kalai, and J. Langford. Beating the hold-out: Bounds for k-fold and 
            progressive cross-validation. In Conference on Learning Theory (COLT), 1999.
    """

    def __init__(self, games: Sequence[Game], n_rounds: int = 30) -> None:
        """Initialize ProgressiveBenchmark.

        Args:
            games: The games used for online, on-policy evaluation.
            n_rounds: The number of rounds to be played on each game.
        """
        self._games = games
        self._n_rounds = n_rounds

    def evaluate(self, solver_factory: Callable[[],Solver]) -> Result:
        """Calculate the progressive validation gain on a per-round basis.
            
            Args:
                solver_factory: See the base class for more information.
            
            Returns:
                See the base class for more information.
        """
        round_pvg: List[List[float]] = [ [] for i in range(self._n_rounds) ]

        for game in self._games:
            solver = solver_factory()
            progressive_reward = 0.0

            for n,r in enumerate(islice(game.rounds, self._n_rounds)):

                choice = solver.choose(r.state, r.actions)                
                state  = r.state
                action = r.actions[choice]
                reward = r.rewards[choice]

                solver.learn(state, action, reward)

                progressive_reward = 1/(n+1) * reward + n/(n+1) * progressive_reward

                round_pvg[n].append(progressive_reward)

        return Result.from_samples(round_pvg)

class TraditionalBenchmark(Benchmark):
    """An on-policy Benchmark using unbiased samples to estimate E[reward|learning-iteration].

        Remarks:
            Samples are unbiased only if the sequence of rounds in a game are unbiased.
            This doesn't mean that a game can't be static, it simply means that there should
            be a uniform random shuffling of all rounds performed at least once on a game. Once
            such a shuffling has been done then a game can be fixed for all benchmarks after that.
    """

    def __init__(self, games: Sequence[Game], n_rounds: int, n_iterations: int) -> None:
        """Initialize TraditionalBenchmark.

        Args:
            games: The games used for sampling and learning.
            n_rounds: The number of rounds in each learning-iteration.
            n_iterations: The number of learning learning-iterations to sample.
        """
        self._games        = games
        self._n_rounds     = n_rounds
        self._n_iterations = n_iterations
        

    def evaluate(self, solver_factory: Callable[[],Solver]) -> Result:
        """Calculate the E[reward|learning-iteration] for the given Solver factory.

            Args:
                solver_factory: See the base class for more information.
            
            Returns:
                See the base class for more information.
        """

        iteration_rwds: List[List[float]] = [ [] for i in range(self._n_iterations) ]

        for game in self._games:
            solver = solver_factory()

            for i in range(self._n_iterations):

                rounds  = list(islice(game.rounds, self._n_rounds))
                
                choices = [ solver.choose(r.state, r.actions) for r in rounds ]
                states  = [ r.state for r in rounds ]
                actions = [ r.actions[c] for r,c in zip(rounds,choices)]
                rewards = [ r.rewards[c] for r,c in zip(rounds,choices)]

                iteration_rwds[i].extend(rewards)

                for s,a,r in zip(states,actions,rewards):
                    solver.learn(s,a,r)

        return Result.from_samples(iteration_rwds)