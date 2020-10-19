"""The benchmarks module contains core benchmark functionality and protocols.

This module contains the abstract interface expected for Benchmark implementations. This 
module also contains several Benchmark implementations and Result data transfer class.
"""

import json
import collections

from concurrent.futures import ProcessPoolExecutor
from statistics import mean
from abc import ABC, abstractmethod
from itertools import repeat, product, islice, accumulate
from statistics import median
from pathlib import Path
from typing import (
    Iterable, Tuple, Hashable, Union, Sequence, Callable, 
    Generic, TypeVar, Dict, Any, cast, Optional
)

from coba.simulations import Interaction, LazySimulation, JsonSimulation, Simulation, Context, Action, Key, Choice, Reward
from coba.preprocessing import Batcher
from coba.learners import Learner
from coba.execution import ExecutionContext
from coba.json import CobaJsonDecoder
from coba.data import ReadWrite, DiskReadWrite, MemoryReadWrite, Table
from coba.random import Random

_C  = TypeVar('_C', bound=Context)
_A  = TypeVar('_A', bound=Action)

class TransactionReadWrite:

    def __init__(self, readwrite: ReadWrite):
        self._readwrite = readwrite

    def write_learner(self, learner_id:int, **kwargs):
        """Write learner metadata row to Result.
        
        Args:
            learner_id: The primary key for the given learner.
            kwargs: The metadata to store about the learner.
        """

        key_items = [(cast(str,"learner_id"),learner_id)]
        row       = collections.OrderedDict(key_items + list(kwargs.items()))

        self._readwrite.write(["L", row])

    def write_simulation(self, simulation_id: int, **kwargs):
        """Write simulation metadata row to Result.
        
        Args:
            simulation_index: The index of the simulation in the benchmark's simulations.
            simulation_seed: The seed used to shuffle the simulation before evaluation.
            kwargs: The metadata to store about the learner.
        """
        key       = simulation_id
        key_items = [(cast(str,"simulation_id"),simulation_id)]
        row       = collections.OrderedDict(key_items + list(kwargs.items()))

        self._readwrite.write(["S", row])

    def write_batch(self, learner_id:int, simulation_id:int, seed: Optional[int], batch_index:int, **kwargs):
        """Write batch metadata row to Result.

        Args:
            learner_id: The primary key for the learner we observed the batch for.
            simulation_id: The primary key for the simulation the batch came from.
            batch_index: The index of the batch within the simulation.
            kwargs: The metadata to store about the batch.
        """
        key       = (learner_id, simulation_id, seed, batch_index)
        key_items = [("learner_id",learner_id), ("simulation_id",simulation_id), ("seed", seed), ("batch_index",batch_index)]
        row       = collections.OrderedDict(key_items + list(kwargs.items()))

        self._readwrite.write(["B", row])

    def read(self) -> Iterable[Any]:
        return self._readwrite.read()

class Result:
    """A class for creating and returning the result of a Benchmark evaluation."""

    @staticmethod
    def from_transaction_log(filename: str = None) -> 'Result':
        """Create a Result from a transaction file."""
        
        if filename is None or not Path(filename).exists(): return Result()

        decoder = CobaJsonDecoder()
        lines   = Path(filename).read_text().split("\n")
        
        return Result.from_transactions([decoder.decode(l) for l in lines if l != ''])

    @staticmethod
    def from_transactions(transactions: Iterable[Tuple[str,Dict[str,Any]]]) -> 'Result':

        result = Result()

        for transaction in transactions:
            
            if transaction[0] == "L":
                result.learners.add_row(**transaction[1])
            
            if transaction[0] == "S":
                result.simulations.add_row(**transaction[1])
            
            if transaction[0] == "B":
                result.batches.add_row(**transaction[1])

        return result

    def __init__(self) -> None:
        """Instantiate a Result class."""

        self.learners    = Table("Learners"   , ['learner_id'])
        self.simulations = Table("Simulations", ['simulation_id'])
        self.batches     = Table("Batches"    , ['learner_id', 'simulation_id', 'seed', 'batch_index'])

    def to_tuples(self) -> Tuple[Sequence[Any], Sequence[Any], Sequence[Any]]:
        return (
            self.learners.to_tuples(),
            self.simulations.to_tuples(),
            self.batches.to_tuples()
        )

    def to_indexed_tuples(self) -> Tuple[Dict[Hashable,Any], Dict[Hashable,Any], Dict[Hashable,Any]]:
        return (
            self.learners.to_indexed_tuples(),
            self.simulations.to_indexed_tuples(),
            self.batches.to_indexed_tuples()
        )

    def to_pandas(self) -> Tuple[Any,Any,Any]:
        l = self.learners.to_pandas()
        s = self.simulations.to_pandas()
        b = self.batches.to_pandas()

        return (l,s,b)

    def __str__(self) -> str:
        return str({ "Learners": len(self.learners), "Simulations": len(self.simulations), "Batches": len(self.batches) })

    def __repr__(self) -> str:
        return str(self)

class Benchmark(Generic[_C,_A], ABC):
    """The interface for Benchmark implementations."""

    @abstractmethod
    def evaluate(self, learner_factories: Sequence[Callable[[],Learner[_C,_A]]]) -> Result:
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

class UniversalBenchmark(Benchmark[_C,_A]):
    """An on-policy Benchmark using samples drawn from simulations to estimate performance statistics."""

    class _Simulation(Simulation[Context, Action]):
        
        def __init__(self, 
            index       :int, 
            seed        : Optional[int], 
            batch_sizes : Sequence[int],
            simulation  : Simulation[Context,Action]) -> None:

            self.index        = index
            self.seed         = seed
            self.batch_slices = list(accumulate([0] + list(batch_sizes)))

            self._interactions = Random(seed).shuffle(simulation.interactions) if seed else simulation.interactions
            self._interactions = self._interactions[0:sum(batch_sizes)]
            
            if isinstance(simulation, (LazySimulation, JsonSimulation)):
                self._rewards = simulation._simulation.rewards
            else:
                self._rewards = simulation.rewards

        @property
        def batches(self) -> Iterable[Iterable[Interaction[Context, Action]]]:
            for i in range(len(self.batch_slices)-1):
                yield islice(self._interactions, self.batch_slices[i], self.batch_slices[i+1])

        @property
        def interactions(self) -> Sequence[Interaction[Context, Action]]:
            return self._interactions

        def rewards(self, choices: Sequence[Tuple[Key, Choice]]) -> Sequence[Reward]:
            return self._rewards(choices)

    class _Learner(Learner[Context, Action]):
        
        def __init__(self, index:int, factory: Callable[[],Learner[Context,Action]]) -> None:
            self.index = index
            learner = factory()
            
            try:
                self._family = learner.family
            except:
                self._family = learner.__class__.__name__

            try:
                self._params = learner.params
            except:
                self._params =  {}

            if len(self.params) > 0:
                self._full_name = f"{self.family}({','.join(f'{k}={v}' for k,v in self.params.items())})"
            else:
                self._full_name = self.family

            self._choose = learner.choose
            self._learn  = learner.learn

        @property
        def family(self) -> str:
            return self._family

        @property
        def params(self) -> Dict[str,Any]:
            return self._params

        @property
        def full_name(self) -> str:
            return self._full_name

        def choose(self, key: Key, context: Context, actions: Sequence[Action]) -> Choice:
            return self._choose(key, context, actions)
        
        def learn(self, key: Key, context: Context, action: Action, reward: Reward) -> None:
            self._learn(key, context, action, reward)

    @staticmethod
    def from_file(filename:str) -> 'UniversalBenchmark[Context,Action]':
        """Instantiate a Benchmark from a config file."""

        suffix = Path(filename).suffix
        
        if suffix == ".json":
            return UniversalBenchmark.from_json(Path(filename).read_text())

        raise Exception(f"The provided file type ('{suffix}') is not a valid format for benchmark configuration")

    @staticmethod
    def from_json(json_val:Union[str, Dict[str,Any]]) -> 'UniversalBenchmark[Context,Action]':
        """Create a UniversalBenchmark from json text or object.

        Args:
            json_val: Either a json string or the decoded json object.

        Returns:
            The UniversalBenchmark representation of the given JSON string or object.
        """

        if isinstance(json_val, str):
            config = cast(Dict[str,Any],json.loads(json_val))
        else:
            config = json_val

        config = ExecutionContext.Templating.parse(config)

        if not isinstance(config["simulations"], collections.Sequence):
            config["simulations"] = [ config["simulations"] ]

        simulations  = [ Simulation.from_json(sim_config) for sim_config in config["simulations"] ]
        batcher      = Batcher.from_json(config.get("batches","{'size': 1}"))
        ignore_first = config.get("ignore_first", True)
        ignore_raise = config.get("ignore_raise", True)
        shuffle      = config.get("shuffle", [None])

        return UniversalBenchmark(simulations, batcher, ignore_first, ignore_raise, shuffle)

    def __init__(self,
        simulations : Sequence[Simulation[_C,_A]], 
        batcher: Batcher,
        ignore_first: bool = True,
        ignore_raise: bool = True,
        shuffle_seeds: Sequence[Optional[int]] = [None],
        max_processes: int = None) -> None:
        """Instantiate a UniversalBenchmark.

        Args:
            simulations: A sequence of simulations to benchmark against.
            batcher: Determines how each simulation is broken into evaluation/learning batches.
            ignore_first: Determines if the first batch should be ignored since no learning has occured yet.
            ignore_raise: Determines if exceptions during benchmark evaluation are raised or simply logged.
            shuffle_seeds: A sequence of seeds to shuffle simulations by when evaluating. None means no shuffle.
            max_processes: The maximum number of process to spawn while evaluating the benchmark. This value will
                override any value that is in a coba config file.
        
        See the overloads for more information.
        """

        self._simulations   = simulations
        self._batcher       = batcher
        self._ignore_first  = ignore_first
        self._ignore_raise  = ignore_raise
        self._seeds         = shuffle_seeds
        self._max_processes = max_processes

    def ignore_raise(self, value:bool=True) -> 'UniversalBenchmark[_C,_A]':
        return UniversalBenchmark(self._simulations, self._batcher, self._ignore_first, value, self._seeds, self._max_processes)

    def ignore_first(self, value:bool=True) -> 'UniversalBenchmark[_C,_A]':
        return UniversalBenchmark(self._simulations, self._batcher, value, self._ignore_raise, self._seeds, self._max_processes)

    def core_count(self, value:int) -> 'UniversalBenchmark[_C,_A]':
        return UniversalBenchmark(self._simulations, self._batcher, self._ignore_first, self._ignore_raise, self._seeds, value)

    def evaluate(self, learner_factories: Sequence[Callable[[],Learner[_C,_A]]], transaction_log:str = None) -> Result:
        """Collect observations of a Learner playing the benchmark's simulations to calculate Results.

        Args:
            learner_factories: See the base class for more information.

        Returns:
            See the base class for more information.
        """
        
        restored     = Result.from_transaction_log(transaction_log)
        readwrite    = DiskReadWrite(transaction_log) if transaction_log else MemoryReadWrite()
        transactions = TransactionReadWrite(readwrite)

        n_restored_learners = len(restored.learners._rows)
        n_given_learners    = len(learner_factories)

        if n_restored_learners > 0 and n_restored_learners != n_given_learners:
            raise Exception("The number of learners differs from the transaction log.")

        mp = self._max_processes if self._max_processes else ExecutionContext.Config.max_processes

        if n_restored_learners == 0:
            for learner in map(UniversalBenchmark._Learner, *zip(*enumerate(learner_factories))):
                learner_row = {"family":learner.family, "full_name": learner.full_name, **learner.params}
                transactions.write_learner(learner.index, **learner_row)

        #write number of learners, number of simulations, batcher, shuffle seeds and ignore first
        #make sure all these variables are the same. If they've changed then fail gracefully.                    

        tasks = self._make_tasks(self._simulations, learner_factories, restored, transactions)

        if mp == 1:
            for task_results in map(self._process_task, tasks, repeat(restored)):
                for key, row in filter(None,task_results):
                    transactions.write_batch(*key, **row)

        if mp > 1:
            with ProcessPoolExecutor(mp) as exe: 
                for task_results in exe.map(self._process_task, tasks, repeat(restored)):
                    for key, row in filter(None,task_results):
                        transactions.write_batch(*key, **row)

        return Result.from_transactions(transactions.read())

    def _make_simulations(self, simulations, restored, results) -> 'Iterable[UniversalBenchmark._Simulation]':
        for index, simulation in enumerate(simulations):
            try:
                if self._simulation_finished_in_restored(restored, index): continue

                with self._lazy_simulation(simulation) as loaded_simulation:

                    batch_sizes = self._batcher.batch_sizes(len(loaded_simulation.interactions))

                    if index not in restored.simulations:
                        results.write_simulation(index,
                            interaction_count = sum(batch_sizes[int(self._ignore_first):]),
                            batch_count       = len(batch_sizes[int(self._ignore_first):]),
                            seed_count        = len(self._seeds),
                            context_size      = int(median(self._context_sizes(loaded_simulation))),
                            action_count      = int(median(self._action_counts(loaded_simulation)))
                        )

                    for seed in self._seeds: yield UniversalBenchmark._Simulation(index, seed, batch_sizes, simulation)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                ExecutionContext.Logger.log_exception(e, "unhandled exception:")
                if not self._ignore_raise: raise e

    def _make_learners(self, simulation, factories, restored, results) -> 'Iterable[UniversalBenchmark._Learner]':
        for learner in map(UniversalBenchmark._Learner, *zip(*enumerate(factories))):
            if not self._simulation_learner_finished_in_restored(restored, simulation, learner.index):
                yield learner

    def _make_tasks(self, simulations, factories, restored, results) -> 'Iterable[Tuple[UniversalBenchmark._Simulation, Learner]]':
        for simulation in self._make_simulations(simulations, restored, results):
            for learner in self._make_learners(simulation, factories, restored, results):
                yield (simulation,learner)
        
    def _process_task(self, task, restored):
        result = []
        for batch in enumerate(task[0].batches):
            result.append(self._process_batch(task[0], task[1], batch, restored))
        return result

    def _process_batch(self, simulation, learner, batch, restored):
        batch_index        = batch[0]
        batch_interactions = batch[1]
        
        keys     = []
        contexts = []
        choices  = []
        actions  = []

        if self._ignore_first:
            batch_index -= 1

        for interaction in batch_interactions:

            choice = learner.choose(interaction.key, interaction.context, interaction.actions)

            assert choice in range(len(interaction.actions)), "An invalid action was chosen by the learner"

            keys    .append(interaction.key)
            contexts.append(interaction.context)
            choices .append(choice)
            actions .append(interaction.actions[choice])

        rewards = simulation.rewards(list(zip(keys, choices))) 

        for (key,context,action,reward) in zip(keys,contexts,actions,rewards):
            learner.learn(key,context,action,reward)

        batch_key = (learner.index, simulation.index, simulation.seed, batch_index)

        if batch_index >= 0 and batch_key not in restored.batches:
            key = (learner.index, simulation.index, simulation.seed, batch_index)
            row = {"N":len(rewards), "reward":mean(rewards)}
            return (key,row)
        else:
            return None

    #Begin utility classes
    def _simulation_finished_in_restored(self, restored: Result, simulation_index: int) -> bool:

        if simulation_index not in restored.simulations:
            return False #this simulation has never been processed

        n_learners = len(restored.learners._rows)
        n_batches  = restored.simulations.get_row(simulation_index)['batch_count']
        n_seeds    = len(self._seeds)

        total_batch_count    = n_learners * 1 * n_seeds * n_batches
        restored_batch_count = 0

        for learner_index, shuffle_seed, batch_index in product(range(n_learners), self._seeds, range(n_batches)):
            batch_key = (learner_index, simulation_index, shuffle_seed, batch_index)
            restored_batch_count += int(batch_key in restored.batches)

        return restored_batch_count == total_batch_count

    def _simulation_learner_finished_in_restored(self, restored, simulation, learner_index) -> bool:

        if simulation.index not in restored.simulations:
            return False #this simulation has never been processed

        restored_simulation = restored.simulations.get_row(simulation.index)

        if restored_simulation['batch_count'] == 0:
            return True #this simulation was previously processed and found to be too small to batch

        fully_evaluated_batch_count = restored_simulation['batch_count']
        restored_batch_count        = 0

        for batch_index in range(restored_simulation['batch_count']):
            batch_key = (learner_index, simulation.index, simulation.seed, batch_index)
            restored_batch_count += int(batch_key in restored.batches)

        # true if all batches were evaluated previously
        return restored_batch_count == fully_evaluated_batch_count

    def _lazy_simulation(self, simulation: Simulation) -> Union[LazySimulation,JsonSimulation]:
        return simulation if isinstance(simulation, (LazySimulation, JsonSimulation)) else LazySimulation(lambda: simulation)

    def _context_sizes(self, simulation: Simulation) -> Iterable[int]:
        for context in [i.context for i in simulation.interactions]:
            yield 0 if context is None else len(context) if isinstance(context,tuple) else 1
    
    def _action_counts(self, simulation: Simulation) -> Iterable[int]:
        for actions in [i.actions for i in simulation.interactions]:
            yield len(actions)