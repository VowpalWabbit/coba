"""The benchmarks module contains core benchmark functionality and protocols.

This module contains the abstract interface expected for Benchmark implementations. This 
module also contains several Benchmark implementations and Result data transfer class.

Todo:
    * finish docstring for Stats and Results
    * Add more statistics to the Stats class
    * Finish incorporating built in plotting somehow
"""

from abc import ABC, abstractmethod
from typing import Union, Sequence, List, Callable, Optional, Tuple, Generic, TypeVar
from itertools import islice

from bbench.simulations import Simulation, State, Action
from bbench.learners import Learner

T_S = TypeVar('T_S', bound=State)
T_A = TypeVar('T_A', bound=Action)

class Stats:

    @staticmethod
    def from_values(vals: Sequence[float]) -> "Stats":
        mean = float('nan') if len(vals) == 0 else sum(vals)/len(vals)

        return Stats(mean)

    def __init__(self, mean: float):
        self._mean = mean

    @property
    def mean(self) -> float:
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
        # indexes spread across simulations are grouped together.

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

class Benchmark(Generic[T_S,T_A], ABC):
    """The interface for Benchmark implementations."""
    
    @abstractmethod
    def evaluate(self, learner_factory: Callable[[],Learner[T_S,T_A]]) -> Result:
        """Calculate the performance for a provided bandit Learner.

        Args:
            learner_factory: A function to create Learner instances. The function should 
                always create the same Learner in order to get an unbiased performance 
                Result. This method can be as simple as `lambda: MyLearner(...)`.

        Returns:
            A Result containing the performance statistics of the benchmark.

        Remarks:
            The learner factory is necessary because a Result can be calculated using
            observed performance over several simulations. In these cases the easiest 
            way to reset a learner's learned policy is to create a new learner.
        """
        ...

class UniversalBenchmark(Benchmark[T_S,T_A]):
    """An on-policy Benchmark using unbiased samples to estimate performance statistics.

    Remarks:
        Results are unbiased only if the sequence of rounds in a given sim are stationary.
        This doesn't mean that a sim can't be static, it simply means that there should
        be a uniform random shuffling of all rounds performed at least once on a sim. Once
        such a shuffling has been done then a sim can be fixed for all benchmarks after that.
    """

    def __init__(self, 
        simulations   : Sequence[Simulation[T_S,T_A]],
        n_sim_rounds  : Optional[int],
        n_batch_rounds: Union[int, Callable[[int],int]]) -> None:
        
        self._simulations: Sequence[Simulation[T_S,T_A]]   = simulations
        self._n_sim_rounds  = n_sim_rounds

        if isinstance(n_batch_rounds, int):
            self._n_batch_rounds = lambda i: n_batch_rounds
        else:
            self._n_batch_rounds = n_batch_rounds

    def evaluate(self, learner_factory: Callable[[],Learner[T_S,T_A]]) -> Result:
        """Collect observations of a Learner playing the benchmark's simulations to create Results.

        Args:
            learner_factory: See the base class for more information.
        
        Returns:
            See the base class for more information.
        """

        results:List[Tuple[int,int,float]] = []

        for sim_index, sim in enumerate(self._simulations):

            sim_learner   = learner_factory()
            batch_index   = 0
            batch_samples = []

            for r in islice(sim.rounds, self._n_sim_rounds):

                choice = sim_learner.choose(r.state, r.actions) 
                state  = r.state
                action = r.actions[choice]
                reward = r.rewards[choice]

                batch_samples.append((state, action, reward))

                if len(batch_samples) == self._n_batch_rounds(batch_index):

                    for (state,action,reward) in batch_samples:
                        sim_learner.learn(state,action,reward)
                        results.append((sim_index, batch_index, reward))

                    batch_samples = []
                    batch_index += 1
                    
            for (state,action,reward) in batch_samples:
                sim_learner.learn(state,action,reward)
                results.append((sim_index, batch_index, reward))

        return Result(results)