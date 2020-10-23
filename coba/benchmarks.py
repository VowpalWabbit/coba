"""The benchmarks module contains core benchmark functionality and protocols.

This module contains the abstract interface expected for Benchmark implementations. This 
module also contains several Benchmark implementations and Result data transfer class.
"""

import itertools
import json
import collections

from statistics import mean
from abc import ABC, abstractmethod
from itertools import product, islice, accumulate, groupby
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
from coba.data import Pipe, MemorySink, MemorySource, StopPipe, Filter, DiskSource, DiskSink, JsonEncode, JsonDecode, Table
from coba.random import Random

_C_out = TypeVar('_C_out', bound=Context, covariant=True)
_A_out = TypeVar('_A_out', bound=Action, covariant=True)
_C     = TypeVar('_C'    , bound=Context)
_A     = TypeVar('_A'    , bound=Action)

class Result:
    """A class for creating and returning the result of a Benchmark evaluation."""

    @staticmethod
    def from_transaction_log(filename: Optional[str]) -> 'Result':
        """Create a Result from a transaction file."""
        
        if filename is None or not Path(filename).exists(): return Result()

        Pipe.join(DiskSource(filename), [JsonDecode(), TransactionPromote(), JsonEncode()], DiskSink(filename, 'w')).run()
        
        return Result.from_transactions(Pipe.join(DiskSource(filename), [JsonDecode()]).read())

    @staticmethod
    def from_transactions(transactions: Iterable[Any]) -> 'Result':

        result = Result()

        for transaction in transactions:
            if transaction[0] == "version"  : result.version = transaction[1]
            if transaction[0] == "benchmark": result.benchmark = transaction[1]
            if transaction[0] == "L"        : result.learners.add_row(transaction[1], **transaction[2])
            if transaction[0] == "S"        : result.simulations.add_row(transaction[1], **transaction[2])
            if transaction[0] == "B"        : result.batches.add_row(*transaction[1], **transaction[2])

        return result

    def __init__(self) -> None:
        """Instantiate a Result class."""

        self.version     = None
        self.benchmark   = cast(Dict[str,Any],{})
        self.learners    = Table("Learners"   , ['learner_id'])
        self.simulations = Table("Simulations", ['simulation_id'])
        
        #Warning, if you change the order of the columns for batches then:
        # 1. TransactionLogPromote.current_version() will need to be bumped to version 2
        # 2. TransactionLogPromote.to_next_version() will need to promote version 1 to 2
        # 3. TransactionLog.write_batch will need to be modified to write in new order
        self.batches     = Table("Batches"    , ['learner_id', 'simulation_id', 'seed', 'batch_index'])

    def to_tuples(self) -> Tuple[Sequence[Any], Sequence[Any], Sequence[Any]]:
        return (
            self.learners.to_tuples(),
            self.simulations.to_tuples(),
            self.batches.to_tuples()
        )

    def to_indexed_tuples(self) -> Tuple[Dict[int,Any], Dict[int,Any], Dict[Tuple[int,int,Optional[int],int],Any]]:
        return (
            cast(Dict[int,Any], self.learners.to_indexed_tuples()),
            cast(Dict[int,Any], self.simulations.to_indexed_tuples()),
            cast(Dict[Tuple[int,int,Optional[int],int],Any], self.batches.to_indexed_tuples())
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

class Transaction:

    @staticmethod
    def version(version) -> Any:
        return ['version', version]
    
    @staticmethod
    def benchmark(n_learners, n_simulations, n_seeds, batcher, ignore_first) -> Any:
        data = {
            "n_learners"   :n_learners,
            "n_simulations":n_simulations,
            "n_seeds"      :n_seeds,
            "batcher"      : batcher, 
            "ignore_first" : ignore_first
        }
        
        return ['benchmark',data]

    @staticmethod
    def learner(learner_id:int, **kwargs) -> Any:
        """Write learner metadata row to Result.
        
        Args:
            learner_id: The primary key for the given learner.
            kwargs: The metadata to store about the learner.
        """
        return ["L", learner_id, kwargs]

    @staticmethod
    def simulation(simulation_id: int, **kwargs) -> Any:
        """Write simulation metadata row to Result.
        
        Args:
            simulation_index: The index of the simulation in the benchmark's simulations.
            kwargs: The metadata to store about the learner.
        """
        return ["S", simulation_id, kwargs]

    @staticmethod
    def batch(learner_id:int, simulation_id:int, seed: Optional[int], batch_index:int, **kwargs) -> Any:
        """Write batch metadata row to Result.

        Args:
            learner_id: The primary key for the learner we observed the batch for.
            simulation_id: The primary key for the simulation the batch came from.
            batch_index: The index of the batch within the simulation.
            kwargs: The metadata to store about the batch.
        """
        return ["B", (learner_id, simulation_id, seed, batch_index), kwargs]

class TaskToTransactions(Filter):
    
    def __init__(self, ignore_first: bool, ignore_raise: bool, batcher: Batcher) -> None:
        self._ignore_first = ignore_first
        self._ignore_raise = ignore_raise
        self._batcher      = batcher

    def filter(self, tasks: Iterable[Any]) -> Iterable[Any]:
        for task in tasks:
            for transaction in self._process_task(task):
                yield transaction

    def _process_task(self, task) -> Iterable[Any]:
        
        simulation_index = task[0]
        simulation       = task[1]

        try:
            with self._lazy_simulation(simulation) as simulation:

                batch_sizes = self._batcher.batch_sizes(len(simulation.interactions))

                yield Transaction.simulation(simulation_index,                     
                    interaction_count = sum(batch_sizes[int(self._ignore_first):]),
                    batch_count       = len(batch_sizes[int(self._ignore_first):]),
                    context_size      = int(median(self._context_sizes(simulation))),
                    action_count      = int(median(self._action_counts(simulation))))

                for (seed, learner_index, factory) in task[2]:
                    learner = factory.create()
                    for batch_index, batch in enumerate(self._shuffle_batch(simulation.interactions, seed, batch_sizes)):
                        for transaction in self._process_batch(simulation_index, simulation, seed, learner_index, learner, batch_index, batch):
                            yield transaction

        except KeyboardInterrupt:
            raise
        except Exception as e:
            ExecutionContext.Logger.log_exception(e, "unhandled exception:")
            if not self._ignore_raise: raise e

    def _process_batch(self, simulation_index, simulation, seed, learner_index, learner, batch_index, batch) -> Iterable[Any]:
        
        keys     = []
        contexts = []
        choices  = []
        actions  = []

        if self._ignore_first:
            batch_index -= 1

        for interaction in batch:

            choice = learner.choose(interaction.key, interaction.context, interaction.actions)

            assert choice in range(len(interaction.actions)), "An invalid action was chosen by the learner"

            keys    .append(interaction.key)
            contexts.append(interaction.context)
            choices .append(choice)
            actions .append(interaction.actions[choice])

        rewards = simulation.rewards(list(zip(keys, choices))) 

        for (key,context,action,reward) in zip(keys,contexts,actions,rewards):
            learner.learn(key,context,action,reward)

        if batch_index >= 0:
            yield Transaction.batch(learner_index, simulation_index, seed, batch_index, N=len(rewards), reward=round(mean(rewards),5))

    def _shuffle_batch(self, interactions, seed, batch_sizes):
        batch_slices = list(accumulate([0] + list(batch_sizes)))
        interactions = Random(seed).shuffle(interactions) if seed else interactions
        
        for i in range(len(batch_slices)-1):
            yield islice(interactions, batch_slices[i], batch_slices[i+1])

    def _lazy_simulation(self, simulation: Simulation) -> Union[LazySimulation,JsonSimulation]:
        return simulation if isinstance(simulation, (LazySimulation, JsonSimulation)) else LazySimulation(lambda: simulation)

    def _context_sizes(self, simulation: Simulation) -> Iterable[int]:
        for context in [i.context for i in simulation.interactions]:
            yield 0 if context is None else len(context) if isinstance(context,tuple) else 1
    
    def _action_counts(self, simulation: Simulation) -> Iterable[int]:
        for actions in [i.actions for i in simulation.interactions]:
            yield len(actions)

class TransactionPromote(Filter):

    CurrentVersion = 1

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        items_iter = iter(items)
        items_peek = next(items_iter)
        items_iter = itertools.chain([items_peek], items_iter)

        version = 0 if items_peek[0] != 'version' else items_peek[1]

        if version == TransactionPromote.CurrentVersion:
            raise StopPipe()

        while version != TransactionPromote.CurrentVersion:
            if version == 0:
                promoted_items = [["version",1]]

                for transaction in items:

                    if transaction[0] == "S":
                        
                        index  = transaction[1][1]['simulation_id']
                        values = transaction[1][1]

                        del values['simulation_id']

                        promoted_items.append([transaction[0], index, values])

                    if transaction[0] == "L":

                        index  = transaction[1][1]['learner_id']
                        values = transaction[1][1]

                        del values['learner_id']

                        promoted_items.append([transaction[0], index, values])

                    if transaction[0] == "B":
                        key_columns = ['learner_id', 'simulation_id', 'seed', 'batch_index']
                        
                        index  = [ transaction[1][1][k] for k in key_columns ]
                        values = transaction[1][1]
                        
                        for key_column in key_columns: del values[key_column]
                        
                        if 'reward' in values:
                            values['reward'] = values['reward'].estimate
                        
                        if 'mean_reward' in values:
                            values['reward'] = values['mean_reward'].estimate
                            del values['mean_reward']

                        promoted_items.append([transaction[0], index, values])

                items   = promoted_items
                version = 1

        return items

class TransactionIsNew(Filter):

    def __init__(self, restored: Result):

        self._existing = restored

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        for item in items:

            tipe  = item[0]
            index = item[1]

            if tipe == "B" and index in self._existing.batches:
                continue

            if tipe == "S" and index in self._existing.simulations:
                continue

            if tipe == "L" and index in self._existing.learners:
                continue

            yield item
 
class Factory(Generic[_C_out, _A_out]):
    def __init__(self, ctor: Callable[...,Learner[_C_out,_A_out]], *args, **kwargs) -> None:
        self._ctor   = ctor
        self._args   = args
        self._kwargs = kwargs

    def create(self) -> Learner[_C_out,_A_out]:
        return self._ctor(*self._args, **self._kwargs)

class Benchmark(Generic[_C,_A], ABC):
    """The interface for Benchmark implementations."""

    @abstractmethod
    def evaluate(self, factories: Sequence[Factory[_C,_A]]) -> Result:
        """Calculate the performance for a provided bandit Learner.

        Args:
            factories: A sequence of functions to create Learner instances. Each function 
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

    def evaluate(self, factories: Sequence[Factory[_C,_A]], transaction_log:str = None) -> Result:
        """Collect observations of a Learner playing the benchmark's simulations to calculate Results.

        Args:
            factories: See the base class for more information.

        Returns:
            See the base class for more information.
        """

        restored             = Result.from_transaction_log(transaction_log)
        task_source          = MemorySource(self._make_tasks(self._simulations, self._seeds, factories, restored))
        transaction_sink     = Pipe.join([JsonEncode()], DiskSink(transaction_log)) if transaction_log else MemorySink()
        task_to_transactions = TaskToTransactions(self._ignore_first, self._ignore_raise, self._batcher)
        transactions_are_new = TransactionIsNew(restored)

        n_given_learners    = len(factories)
        n_given_simulations = len(self._simulations)
        n_given_seeds       = len(self._seeds)
        given_batcher       = self._batcher.__class__.__name__
        ignore_first        = self._ignore_first

        preamble_transactions = []

        if restored.version is None:
            preamble_transactions.append(['version', TransactionPromote.CurrentVersion])

        if len(restored.benchmark) == 0:
            preamble_transactions.append(Transaction.benchmark(n_given_learners, n_given_simulations, n_given_seeds, given_batcher, ignore_first))

        if len(restored.learners) == 0:
            for index, factory in enumerate(factories):
                learner = factory.create()
                try:
                    family = learner.family
                except:
                    family = learner.__class__.__name__

                try:
                    params = learner.params
                except:
                    params =  {}

                if len(params) > 0:
                    full_name = f"{family}({','.join(f'{k}={v}' for k,v in params.items())})"
                else:
                    full_name = family

                preamble_transactions.append(Transaction.learner(index, family=family, full_name=full_name, **params))

        if len(restored.benchmark) != 0:
            assert n_given_learners    == restored.benchmark['n_learners'   ], "The currently evaluating benchmark doesn't match the given transaction log"
            assert n_given_simulations == restored.benchmark['n_simulations'], "The currently evaluating benchmark doesn't match the given transaction log"
            assert n_given_seeds       == restored.benchmark['n_seeds'      ], "The currently evaluating benchmark doesn't match the given transaction log"
            assert given_batcher       == restored.benchmark['batcher'      ], "The currently evaluating benchmark doesn't match the given transaction log"
            assert ignore_first        == restored.benchmark['ignore_first' ], "The currently evaluating benchmark doesn't match the given transaction log"

        mp = self._max_processes if self._max_processes else ExecutionContext.Config.max_processes

        transaction_sink.write(preamble_transactions)
        Pipe.join(task_source, [task_to_transactions, transactions_are_new], transaction_sink).run(mp)

        if isinstance(transaction_sink, Pipe.FiltersSink):
            transaction_sink = transaction_sink._sink

        if isinstance(transaction_sink, MemorySink):
            return Result.from_transactions(transaction_sink.items)
        
        if isinstance(transaction_sink, DiskSink):
            return Result.from_transaction_log(transaction_sink.filename)

        raise Exception("Transactions were written to an unrecognized sink.")

    def _make_tasks(self, simulations, seeds, factories, restored) -> Iterable[Any]:

        simulations = dict(enumerate(simulations))
        seeds       = seeds
        factories   = dict(enumerate(factories))

        expected_batch_count = cast(Dict[Tuple[Any,Any,Any], int], collections.defaultdict(lambda:  int(-1)))
        restored_batch_count = cast(Dict[Tuple[Any,Any,Any], int], collections.defaultdict(lambda:  int( 0)))

        for k,v in restored.simulations.rows.items():
            expected_batch_count[k] = v[restored.simulations._columns.index('batch_count')]

        for k in restored.batches.rows:
            restored_batch_count[(k[1],k[2],k[0])] +=1

        is_not_complete = lambda t: restored_batch_count[t] != expected_batch_count[t[0]]
        
        task_keys       = filter(is_not_complete, product(simulations, seeds, factories))
        task_key_groups = groupby(task_keys, key=lambda t: t[0])

        seed_learner = lambda t: (t[1], t[2], factories[t[2]])
        
        for simulation_key, task_key_group in task_key_groups:
            yield simulation_key, simulations[simulation_key], list(map(seed_learner, task_key_group))