"""The benchmarks module contains core benchmark functionality and protocols.

This module contains the abstract interface expected for Benchmark implementations. This 
module also contains several Benchmark implementations and Result data transfer class.

Todo:
    * Add more statistics to the Stats class
    * Incorporate out of the box plots
"""

import json
import collections

from abc import ABC, abstractmethod
from typing import Union, Sequence, List, Callable, Tuple, Generic, TypeVar, Dict, Any, overload
from itertools import islice

from bbench.simulations import Simulation, State, Action
from bbench.learners import Learner

_S = TypeVar('_S', bound=State)
_A = TypeVar('_A', bound=Action)

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

class Benchmark(Generic[_S,_A], ABC):
    """The interface for Benchmark implementations."""
    
    @abstractmethod
    def evaluate(self, learner_factory: Callable[[],Learner[_S,_A]]) -> Result:
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

class UniversalBenchmark(Benchmark[_S,_A]):
    """An on-policy Benchmark using samples drawn from simulations to estimate performance statistics."""

    @staticmethod
    def from_json(json_val:Union[str, Dict[str,Any]]) -> 'UniversalBenchmark':
        """Create a UniversalBenchmark from configuration IO.

        Args:
            json_val: Either a json string or the decoded json object.

        Returns:
            The UniversalBenchmark representation of the given JSON string or object.
        """

        if isinstance(json_val, str):
            config = json.loads(json_val)
        else:
            config = json_val

        is_singular = isinstance(config["simulations"], dict)
        sim_configs = config["simulations"] if not is_singular else [ config["simulations"] ]

        simulations = [ Simulation.from_json(sim_config) for sim_config in sim_configs ]
        batches     = config["batches"]
    
        return UniversalBenchmark(simulations, batches)

    def __init__(self, 
        simulations: Sequence[Simulation[_S,_A]],
        batches    : Union[int, Sequence[int], Callable[[int],int]] = 1, 
        n_rounds   : int = None) -> None:
        """Instantiate a UniversalBenchmark.
        
        Args:
            simulations: A sequence of simulations to benchmark against
            batches: Indicates how to batch evaluations and learning. If batches is an integer
                then all simulations will run to completion with batch sizes of the given int.
                If batches is a sequence of integers then `sum(batches)` rounds will be pulled
                from each simulation and batched according to each int in the sequence. If
                batches is a function of batch_index then it runs until the simulation ends
                with the size of each batch_index equal to the given `func(batch_index)`.
        """

        self._simulations = simulations
        self._batches     = batches

        #if batches is a sequence if ints limit to the batch sum else don't limit the rounds
        self._n_rounds = sum(self._batches) if isinstance(self._batches,collections.Sequence) else n_rounds

    def evaluate(self, learner_factory: Callable[[],Learner[_S,_A]]) -> Result:
        """Collect observations of a Learner playing the benchmark's simulations to calculate Results.

        Args:
            learner_factory: See the base class for more information.
        
        Returns:
            See the base class for more information.
        """

        results:List[Tuple[int,int,float]] = []

        for sim_index, sim in enumerate(self._simulations):

            sim_learner   = learner_factory()
            batch_index   = 0
            batch_choices = []

            for r in islice(sim.rounds, self._n_rounds):

                index = sim_learner.choose(r.state, r.actions)

                batch_choices.append(r.choices[index])
                
                if len(batch_choices) == self._batch_size(batch_index):
                    for (state,action,reward) in sim.rewards(batch_choices):
                        sim_learner.learn(state,action,reward)
                        results.append((sim_index, batch_index, reward))

                    batch_choices = []
                    batch_index += 1
                    
            for (state,action,reward) in sim.rewards(batch_choices):
                sim_learner.learn(state,action,reward)
                results.append((sim_index, batch_index, reward))

        return Result(results)

    def _batch_size(self, i:int) -> int:
        if isinstance(self._batches, int                 ): return self._batches
        if isinstance(self._batches, collections.Sequence): return self._batches[i]
        if callable  (self._batches                      ): return self._batches(i)
        raise Exception("We were unable to determine batch size from the supplied parameters")