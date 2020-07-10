"""The benchmarks module contains core benchmark functionality and protocols.

This module contains the abstract interface expected for Benchmark implementations. This 
module also contains several Benchmark implementations and Result data transfer class.

Todo:
    * Add more statistics to the Stats class
    * Incorporate out of the box plots
"""

import json

from abc import ABC, abstractmethod
from typing import Union, Sequence, List, Callable, Optional, Tuple, Generic, TypeVar, TextIO
from itertools import islice

from bbench.simulations import Simulation, State, Action
from bbench.learners import Learner

ST_in = TypeVar('ST_in', bound=State, contravariant=True)
AT_in = TypeVar('AT_in', bound=Action, contravariant=True)

class Stats:
    """A class to store summary statistics calculated from some sample."""

    @staticmethod
    def from_values(vals: Sequence[float]) -> "Stats":
        """Create a Stats class for some given sequence of values.

        Args:
            vals: A sample of values to calculate statistics for.
        """
        mean = float('nan') if len(vals) == 0 else sum(vals)/len(vals)

        return Stats(mean)

    def __init__(self, mean: float):
        """Instantiate a Stats class.

        Args:
            mean: The mean for some sample of interest.
        """
        self._mean = mean

    @property
    def mean(self) -> float:
        """The mean for some sample."""
        return self._mean

class Result:
    """A class to contain the results of a benchmark evaluation."""

    def __init__(self, observations: Sequence[Tuple[int,int,float]]):
        """Instantiate a Result.

        Args:
            observations: A sequence of three valued tuples where each tuple represents the result of 
                a single round in a benchmark evaluation. The first value in each tuple is a zero-based 
                game index. The second value in the tuple is a zero-based batch index. The final value
                in the tuple is the amount of reward received after taking an action in a round.
        """
        self._observations = observations
        self._batch_stats  = []
        self._cumulative_batch_stats  = []

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
                self._cumulative_batch_stats.append(Stats(prog_mean))

                iter_curr  = observation[1]
                iter_count = 0
                iter_mean  = 0

            iter_count += 1
            prog_count += 1

            iter_mean = (1/iter_count) * observation[2] + (1-1/iter_count) * iter_mean
            prog_mean = (1/prog_count) * observation[2] + (1-1/prog_count) * prog_mean

        self._batch_stats.append(Stats(iter_mean))
        self._cumulative_batch_stats.append(Stats(prog_mean))

    @property
    def batch_stats(self) -> Sequence[Stats]:
        """Pre-calculated statistics for each batch index."""
        return self._batch_stats

    @property
    def cumulative_batch_stats(self) -> Sequence[Stats]:
        """Pre-calculated statistics where batches accumulate all prior batches as you go.  """
        return self._cumulative_batch_stats

    def predicate_stats(self, predicate: Callable[[Tuple[int,int,float]],bool]) -> Stats:
        """Calculate the statistics for any given filter predicate.
        
        Args:
            predicate: Determine which observations to include when calculating statistics.
        """
        return Stats.from_values([o[2] for o in filter(predicate, self._observations)])

    @property
    def observations(self) -> Sequence[Tuple[int,int,float]]:
        return self._observations

class Benchmark(Generic[ST_in,AT_in], ABC):
    """The interface for Benchmark implementations."""
    
    @abstractmethod
    def evaluate(self, learner_factory: Callable[[],Learner[ST_in,AT_in]]) -> Result:
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

class UniversalBenchmark(Benchmark[ST_in,AT_in]):
    """An on-policy Benchmark using samples drawn from simulations to estimate performance statistics."""

    @staticmethod
    def from_json(json_IO: TextIO) -> None:
        """Creates a UniversalBenchmark from configuration IO.

        Args:
            json_IO: An IO stream containing json configuration settings
        """
        json_objects = json.load(json_IO)

        simulations = json_objects["simulations"]
        batches     = json_objects["batches"]
    
        raise NotImplementedError()

    def __init__(self, 
        simulations   : Sequence[Simulation[ST_in,AT_in]],
        n_sim_rounds  : Optional[int],
        n_batch_rounds: Union[int, Callable[[int],int]]) -> None:

        self._simulations: Sequence[Simulation[ST_in,AT_in]]   = simulations
        self._n_sim_rounds  = n_sim_rounds

        if isinstance(n_batch_rounds, int):
            self._n_batch_rounds = lambda i: n_batch_rounds
        else:
            self._n_batch_rounds = n_batch_rounds

    def evaluate(self, learner_factory: Callable[[],Learner[ST_in,AT_in]]) -> Result:
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