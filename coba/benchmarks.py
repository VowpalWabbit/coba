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
from coba.utilities import JsonTemplating
from coba.statistics import Stats

_S = TypeVar('_S', bound=State)
_A = TypeVar('_A', bound=Action)

class Result:
    """A class to contain the results of a benchmark evaluation."""

    @staticmethod
    def from_observations(observations: Sequence[Tuple[int,int,float]], drop_first_batch:bool=True) -> 'Result':
        """Create a Result object from the provided observations. 

        Args:
            observations: A sequence of three valued tuples where each tuple represents the result of 
                a single interaction in a benchmark evaluation. The first value in each tuple is a zero-based 
                sim index. The second value in the tuple is a zero-based batch index. The final value
                in the tuple is the amount of reward received after taking an action in an interaction.
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
            simulations: A sequence of simulations to benchmark against.
            batch_count: How many interaction batches per simulation (batch_size will be spread evenly).
            batch_size: An indication of how large every batch should be. If batch_size is an integer
                then simulations will run until completion with batch sizes of the given int. If 
                batch_size is a sequence of integers then `sum(batch_size)` interactions will be 
                pulled from simulations and batched according to the sequence. If batch_size is a 
                function then simulation run until completion with batch_size determined by function.
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

            try:
                if isinstance(sim, LazySimulation):
                    sim.load()
                    
                batch_sizes = self._batch_sizes(len(sim.interactions))
                n_interactions = sum(batch_sizes)

                for factory, result in zip(learner_factories, learner_results):

                    sim_learner   = factory()
                    batch_index   = 0
                    batch_choices = []

                    for r in islice(sim.interactions, n_interactions):

                        index = sim_learner.choose(r.state, r.actions)

                        batch_choices.append(r.choices[index])
                        
                        if len(batch_choices) == batch_sizes[batch_index]:
                            observations = sim.rewards(batch_choices)
                            self._update_learner_and_result(sim_learner, result, sim_index, batch_index, observations)

                            batch_choices = []
                            batch_index += 1

                    observations = sim.rewards(batch_choices)
                    self._update_learner_and_result(sim_learner, result, sim_index, batch_index, observations)
            except Exception as e:
                print(f"     * {e}")

            if isinstance(sim, LazySimulation):
                sim.unload()

        return learner_results

    def _batch_sizes(self, n_interactions: int) -> Sequence[int]:

        if self._batch_count is not None:
            
            batches   = [int(float(n_interactions)/(self._batch_count))] * self._batch_count
            remainder = n_interactions % self._batch_count
            
            if remainder > 0:
                spacing = float(self._batch_count)/remainder
                for i in range(remainder): batches[int(i*spacing)] += 1

            return batches
        
        if isinstance(self._batch_size, int): 
            return [self._batch_size] * int(float(n_interactions)/self._batch_size)

        if isinstance(self._batch_size, collections.Sequence): 
            return self._batch_size

        if callable(self._batch_size):
            batch_size_iter        = (self._batch_size(i) for i in count())
            next_batch_size        = next(batch_size_iter)
            remaining_interactions = n_interactions
            batch_sizes: List[int] = []

            while remaining_interactions > next_batch_size:
                batch_sizes.append(next_batch_size)
                remaining_interactions -= next_batch_size
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

