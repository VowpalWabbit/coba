"""The benchmarks module contains core benchmark functionality and protocols.

This module contains the abstract interface expected for Benchmark implementations. This 
module also contains several Benchmark implementations and Result data transfer class.

Todo:
    * Incorporate out of the box plots
"""


import json
import collections
import math

from abc import ABC, abstractmethod
from typing import Union, Sequence, List, Callable, Tuple, Generic, TypeVar, Dict, Any, overload, cast, Optional
from itertools import islice, count
from collections import defaultdict

from coba.simulations import LazySimulation, Simulation, State, Action
from coba.learners import Learner
from coba.utilities import JsonTemplating, OnlineMean, OnlineVariance

_S = TypeVar('_S', bound=State)
_A = TypeVar('_A', bound=Action)

class Stats:
    """A class to store summary statistics calculated from some sample."""

    @staticmethod
    def from_observations(observations: Sequence[float]) -> 'Stats':
        """Create a Stats class for some given sequence of values.

        Args:
            vals: A sample of values to calculate statistics for.
        """

        online_mean = OnlineMean()
        online_var  = OnlineVariance()

        for observation in observations:
            online_mean.update(observation)
            online_var .update(observation)

        N    = len(observations)
        mean = online_mean.mean
        var  = online_var.variance
        SEM  = math.sqrt(var/N) if N > 0 else float('nan')

        return Stats(N, mean, var, SEM)

    def __init__(self, N: int = 0, mean: float = float('nan'), variance:float = float('nan'), SEM: float = float('nan')):
        """Instantiate a Stats class.

        Args:
            N: The size of the sample of interest.
            mean: The mean for some sample of interest.
            variance: The variance for some sample of interest.
            SEM: The standard error of the mean for some sample of interest.
        """

        self._N        = N
        self._mean     = mean
        self._variance = variance
        self._SEM      = SEM

    @property
    def N(self) -> int:
        """The size of the sample."""
        return self._N

    @property
    def mean(self) -> float:
        """The mean for some sample."""
        return self._mean

    @property
    def variance(self) -> float:
        """The mean for some sample."""
        return self._variance

    @property
    def SEM(self) -> float:
        """The mean for some sample."""
        return self._SEM

    def blend(self, stats: 'Stats') -> None: #type: ignore #(this error is a bug with pylance)
        """Calculate the stats that would come from blending two samples.
        
        Args:
            stats: The previously calculated stats that we wish to merge.

        Remarks:
            In theory, if every 'Stats' object kept their entire sample then we could estimate all the
            values that we calculate below empirically by mixing all samples into a single big pot and
            then recalculating mean, variance and SEM. Unfortunately, this approach while simple has two
            drawbacks. First, if a benchmark has many simulations with many examples this would mean that 
            our package would always be constrained since our memory complexity would be O(n). Second, our 
            computation complexity would also be around O(Sn) since merging `S` stats will require relooping 
            over all samples again. The downside of not taking the simple approach is that stats with N == 1 
            become problematic since we don't know their variance. Therefore we adjust for this below by 
            adding the average variance back in for ever stat with N == 1.
        """

        total     = 0.
        total_var = 0.
        total_N   = 0
        total_N_1 = 0 

        for stat in [stats, self]: #type: ignore #(this error is a bug with pylance)
            total     += stat.mean * stat.N if stat.N > 0 else 0
            total_var += stat.variance * stat.N if stat.N > 1 else 0
            total_N   += stat.N
            total_N_1 += int(stat.N == 1)

        #when we only have a single observation there is no way for us to estimate
        #the variance of that random number therefore add in a neutral amount of
        #variance since this number is still in total and total_N and we don't want
        #to remove it.
        total_var += (total_var/total_N) * total_N_1

        #to understand why we calculate variance as we do below consider the following
        # E[Z] = (3/5)E[X] + (2/5)E[Y]
        # 5*Var[Z] = 3*Var[X] + 2*Var[Y]

        self._N        = total_N
        self._mean     = total/total_N
        self._variance = total_var/total_N
        self._SEM      = math.sqrt(total_var)/total_N

    def copy(self) -> 'Stats':
        return Stats(self._N, self._mean, self._variance, self._SEM)

class Result:
    """A class to contain the results of a benchmark evaluation."""

    @staticmethod
    def from_observations(observations: Sequence[Tuple[int,int,float]], drop_first_batch:bool=True) -> 'Result':
        """Create a Result object from the provided observations. 

        Args:
            observations: A sequence of three valued tuples where each tuple represents the result of 
                a single round in a benchmark evaluation. The first value in each tuple is a zero-based 
                sim index. The second value in the tuple is a zero-based batch index. The final value
                in the tuple is the amount of reward received after taking an action in a round.
            drop_first_batch: An indicator determining if the first batch should be excluded from Result.
        """
        result = Result(drop_first_batch = drop_first_batch)

        sim_batch_observations: List[float] = []
        sim_index = None
        batch_index = None

        for observation in sorted(observations, key=lambda o: (o[0], o[1])):

            if sim_index is None: sim_index = observation[0]
            if batch_index is None: batch_index = observation[1]

            if sim_index != observation[0] or batch_index != observation[1]:

                result.add_observations(sim_index, batch_index, sim_batch_observations)

                sim_index              = observation[0]
                batch_index            = observation[1]
                sim_batch_observations = []

            sim_batch_observations.append(observation[2])

        if sim_index is not None and batch_index is not None:
            result.add_observations(sim_index, batch_index, sim_batch_observations)

        return result

    def __init__(self, learner_name: str = None, drop_first_batch=True):
        """Instantiate a Result.

        Args:
            drop_first_batch: Indicates if the first batch in every simulation should be excluded from
                Result. The first batch represents choices made without any learning and says relatively
                little about a learner potentially biasing cumulative statistics siginificantly.
        """

        self._learner_name = learner_name if learner_name is not None else "Unknown"

        self._sim_batch_stats: Dict[Tuple[int,int], Stats] = defaultdict(Stats)
        self._batch_stats    : Dict[int           , Stats] = defaultdict(Stats)
        self._sim_stats      : Dict[int           , Stats] = defaultdict(Stats)
        self._prog_stats     : Dict[int           , Stats] = defaultdict(Stats)

        self._drop_first_batch = drop_first_batch

    @property
    def learner_name(self) -> Optional[str]:
        """A descriptive name for the learner used to Generate these results."""

        return self._learner_name

    @property
    def sim_stats(self) -> Sequence[Stats]:
        """Pre-calculated statistics for each simulation."""
        return [ self._sim_stats[key] for key in sorted(self._sim_stats)]

    @property
    def batch_stats(self) -> Sequence[Stats]:
        """Pre-calculated statistics for each batch index."""
        return [ self._batch_stats[key] for key in sorted(self._sim_stats)]

    @property
    def cumulative_batch_stats(self) -> Sequence[Stats]:
        """Pre-calculated statistics where batches accumulate all prior batches as you go."""

        cum_stats = Stats()
        stats     = []

        for key in self._batch_stats:
            cum_stats.blend(self._batch_stats[key])
            stats.append(cum_stats.copy())

        return stats

    def add_observations(self, simulation_index: int, batch_index: int, rewards: Sequence[float]):
        """Update result stats with a new set of batch observations.

        Args:
            simulation_index: A unique identifier for the simulation that rewards came from
            batch_index: A unique identifier for the batch that rewards came from
            rewards: The observed reward values for the given batch in a simulation
        """

        if (self._drop_first_batch and batch_index == 0) or len(rewards) == 0: 
            return

        stats = Stats.from_observations(rewards)

        self._sim_batch_stats[(simulation_index, batch_index)].blend(stats)
        self._batch_stats[batch_index].blend(stats)
        self._sim_stats[batch_index].blend(stats)

class Benchmark(Generic[_S,_A], ABC):
    """The interface for Benchmark implementations."""

    @abstractmethod
    def evaluate(self, learner_factories: Sequence[Callable[[],Learner[_S,_A]]]) -> Sequence[Result]:
        """Calculate the performance for a provided bandit Learner.

        Args:
            learner_factories: A sequence of functions to create Learner instances. Each function 
                should always create the same Learner in order to get an unbiased performance 
                Result. This method can be as simple as `lambda: MyLearner(...)`.

        Returns:
            The resulting performance statistics for each given learner to evaluate.

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
            config = cast(Dict[str,Any],json.loads(json_val))
        else:
            config = json_val

        config = JsonTemplating.parse(config)

        is_singular = isinstance(config["simulations"], dict)
        sim_configs = config["simulations"] if not is_singular else [ config["simulations"] ]

        #by default load simulations lazily
        for sim_config in sim_configs:
            if "lazy" not in sim_config:
                sim_config["lazy"] = True

        simulations = [ Simulation.from_json(sim_config) for sim_config in sim_configs ]

        if "count" in config["batches"]:
            return UniversalBenchmark(simulations, batch_count=config["batches"]["count"])
        else:
            return UniversalBenchmark(simulations, batch_size=config["batches"]["size"])    

    @overload
    def __init__(self, simulations: Sequence[Simulation[_S,_A]],*, batch_count: int) -> None:
        ...

    @overload
    def __init__(self, simulations: Sequence[Simulation[_S,_A]],*, batch_size: Union[int, Sequence[int], Callable[[int],int]]) -> None:
        ...

    def __init__(self,
        simulations: Sequence[Simulation[_S,_A]], batch_count: int = None, batch_size : Union[int, Sequence[int], Callable[[int],int]] = None) -> None:
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
        self._batch_count = batch_count
        self._batch_size  = batch_size

    def evaluate(self, learner_factories: Sequence[Callable[[],Learner[_S,_A]]]) -> Sequence[Result]:
        """Collect observations of a Learner playing the benchmark's simulations to calculate Results.

        Args:
            learner_factories: See the base class for more information.
        
        Returns:
            See the base class for more information.
        """

        learner_results: List[Result] = [Result(name) for name in self._safe_names(learner_factories)]

        for sim_index, sim in enumerate(self._simulations):

            if isinstance(sim, LazySimulation):
                sim.load()
                
            batch_sizes = self._batch_sizes(len(sim.rounds))
            n_rounds = sum(batch_sizes)

            for factory, result in zip(learner_factories, learner_results):

                sim_learner   = factory()
                batch_index   = 0
                batch_choices = []

                for r in islice(sim.rounds, n_rounds):

                    index = sim_learner.choose(r.state, r.actions)

                    batch_choices.append(r.choices[index])
                    
                    if len(batch_choices) == batch_sizes[batch_index]:
                        observations = sim.rewards(batch_choices)
                        self._update_learner_and_result(sim_learner, result, sim_index, batch_index, observations)

                        batch_choices = []
                        batch_index += 1

                observations = sim.rewards(batch_choices)
                self._update_learner_and_result(sim_learner, result, sim_index, batch_index, observations)

            if isinstance(sim, LazySimulation):
                sim.unload()

        return learner_results

    def _batch_sizes(self, n_rounds: int) -> Sequence[int]:

        if self._batch_count is not None:
            
            batches   = [int(float(n_rounds)/(self._batch_count))] * self._batch_count
            remainder = n_rounds % self._batch_count
            
            if remainder > 0:
                spacing = float(self._batch_count)/remainder
                for i in range(remainder): batches[int(i*spacing)] += 1

            return batches
        
        if isinstance(self._batch_size, int): 
            return [self._batch_size] * int(float(n_rounds)/self._batch_size)

        if isinstance(self._batch_size, collections.Sequence): 
            return self._batch_size

        if callable(self._batch_size):
            batch_size_iter        = (self._batch_size(i) for i in count())
            next_batch_size        = next(batch_size_iter)
            remaining_rounds       = n_rounds
            batch_sizes: List[int] = []

            while remaining_rounds > next_batch_size:
                batch_sizes.append(next_batch_size)
                remaining_rounds -= next_batch_size
                next_batch_size  = next(batch_size_iter)
            
            return batch_sizes
        
        raise Exception("We were unable to determine batch size from the supplied parameters")

    def _update_learner_and_result(self, 
        learner: Learner[_S,_A],
        result: Result,
        sim_index:int, 
        batch_index:int, 
        results:Sequence[Tuple[_S,_A,float]]):

        result.add_observations(sim_index, batch_index, [ batch_result[2] for batch_result in results ])

        for (state,action,reward) in results:
            learner.learn(state,action,reward)

    def _safe_names(self, factories: Sequence[Callable[[],Learner[_S,_A]]]) -> Sequence[Optional[str]]:

        names: List[Optional[str]] = []

        for factory in factories:
            try:
                names.append(factory().name)
            except:
                names.append(None)
        
        return names

