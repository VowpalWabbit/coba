"""The benchmarks module contains core benchmark functionality and protocols.

This module contains the abstract interface expected for Benchmark implementations. This 
module also contains several Benchmark implementations and Result data transfer class.
"""

import math
import itertools
import json
import collections

from copy import deepcopy
from statistics import mean
from abc import ABC, abstractmethod
from itertools import product, islice, accumulate, groupby
from statistics import median
from pathlib import Path
from typing import Iterable, Tuple, Union, Sequence, Generic, TypeVar, Dict, Any, cast, Optional, overload, List

from coba.learners import Learner, Choice, Key
from coba.simulations import JsonSimulation, Simulation, Context, Action, Reward
from coba.execution import ExecutionContext
from coba.data.structures import Table
from coba.data.filters import Filter, JsonEncode, JsonDecode
from coba.data.sources import Source, MemorySource, DiskSource
from coba.data.sinks import Sink, MemorySink, DiskSink
from coba.data.pipes import Pipe, StopPipe
from coba.statistics import OnlineMean, OnlineVariance
from coba.utilities import check_matplotlib_support, check_pandas_support
from coba.random import CobaRandom

_C = TypeVar('_C', bound=Context)
_A = TypeVar('_A', bound=Action)

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

        check_pandas_support("Result.to_pandas")

        l = self.learners.to_pandas()
        s = self.simulations.to_pandas()
        b = self.batches.to_pandas()

        return (l,s,b)

    def standard_plot(self, select_learners: Sequence[int] = None,  show_err: bool = False, show_sd: bool = False) -> None:

        check_matplotlib_support('Plots.standard_plot')

        def _plot(axes, label, xs, ys, vs, ns):
            axes.plot(xs, ys, label=label)

            if show_sd:
                ls = [ y-math.sqrt(v) for y,v in zip(ys,vs) ]
                us = [ y+math.sqrt(v) for y,v in zip(ys,vs) ]
                axes.fill_between(xs, ls, us, alpha = 0.1)

            if show_err:
                # I don't really understand what this is... For each x our distribution
                # is changing so its VAR is also changing. What does it mean to calculate
                # sample variance from a deterministic collection of random variables with
                # different distributions? For example sample variance of 10 random variables
                # from dist1 and 10 random variables from dist2... This is not the same as 20
                # random variables with 50% chance drawing from dist1 and 50% chance of drawing
                # from dist2. So the distribution can only be defined over the whole space (i.e.,
                # all 20 random variables) and not for a specific random variable. Oh well, for
                # now I'm leaving this as it is since I don't have any better ideas. I think what
                # I've done is ok, but I need to more some more thought into it.
                ls = [ y-math.sqrt(v/n) for y,v,n in zip(ys,vs,ns) ]
                us = [ y+math.sqrt(v/n) for y,v,n in zip(ys,vs,ns) ]
                axes.fill_between(xs, ls, us, alpha = 0.1)

        learners, _, batches = self.to_indexed_tuples()

        learners = {key:value for key,value in learners.items() if select_learners is None or key in select_learners}
        batches  = {key:value for key,value in batches.items() if select_learners is None or value.learner_id in select_learners}

        learner_index_key = lambda batch: (batch.learner_id, batch.batch_index)
        sorted_batches    = sorted(batches.values(), key=learner_index_key)
        grouped_batches   = groupby(groupby(sorted_batches , key=learner_index_key), key=lambda x: x[0][0])

        max_batch_N = 0

        indexes     = cast(Dict[int,List[int  ]], collections.defaultdict(list))
        incounts    = cast(Dict[int,List[int  ]], collections.defaultdict(list))
        inmeans     = cast(Dict[int,List[float]], collections.defaultdict(list))
        invariances = cast(Dict[int,List[float]], collections.defaultdict(list))
        cucounts    = cast(Dict[int,List[int  ]], collections.defaultdict(list))
        cumeans     = cast(Dict[int,List[float]], collections.defaultdict(list))
        cuvariances = cast(Dict[int,List[float]], collections.defaultdict(list))

        for learner_id, learner_batches in grouped_batches:

            cucount    = 0
            cumean     = OnlineMean()
            cuvariance = OnlineVariance()

            for (_, batch_index), index_batches in learner_batches:

                incount    = 0
                inmean     = OnlineMean()
                invariance = OnlineVariance()

                for N, reward in [ (b.N, b.reward) for b in index_batches]:
                    
                    max_batch_N = max(N, max_batch_N)
                    
                    incount     = incount + 1
                    inmean      .update(reward)
                    invariance  .update(reward)
                    cucount     = cucount + 1
                    cumean      .update(reward)
                    cuvariance  .update(reward)

                #sanity check, sorting above (in theory) should take care of this...
                #if this isn't the case then the cu* values will be incorrect...
                assert indexes[learner_id] == [] or batch_index > indexes[learner_id][-1]

                incounts[learner_id].append(incount)
                indexes[learner_id].append(batch_index)
                inmeans[learner_id].append(inmean.mean)
                invariances[learner_id].append(invariance.variance)
                cucounts[learner_id].append(cucount)
                cumeans[learner_id].append(cumean.mean)
                cuvariances[learner_id].append(cuvariance.variance)

        import matplotlib.pyplot as plt #type: ignore

        fig = plt.figure(figsize=(12,4))

        index_unit = "Interaction" if max_batch_N ==1 else "Batch"
        
        ax1 = fig.add_subplot(1,2,1) #type: ignore
        ax2 = fig.add_subplot(1,2,2) #type: ignore

        for learner_id in learners:
            _plot(ax1, learners[learner_id].full_name, indexes[learner_id], inmeans[learner_id], invariances[learner_id], incounts[learner_id])

        ax1.set_title(f"Instantaneous Reward")
        ax1.set_ylabel("Reward")
        ax1.set_xlabel(f"{index_unit} Index")

        for learner_id in learners:
            _plot(ax2, learners[learner_id].full_name, indexes[learner_id], cumeans[learner_id], cuvariances[learner_id], cucounts[learner_id])

        ax2.set_title("Progressive Validation")
        #ax2.set_ylabel("Reward")
        ax2.set_xlabel(f"{index_unit} Index")

        (bot1, top1) = ax1.get_ylim()
        (bot2, top2) = ax2.get_ylim()

        ax1.set_ylim(min(bot1,bot2), max(top1,top2))
        ax2.set_ylim(min(bot1,bot2), max(top1,top2))

        scale = 0.75
        box1 = ax1.get_position()
        box2 = ax2.get_position()
        ax1.set_position([box1.x0, box1.y0 + box1.height * (1-scale), box1.width, box1.height * scale])
        ax2.set_position([box2.x0, box2.y0 + box2.height * (1-scale), box2.width, box2.height * scale])

        # Put a legend below current axis
        fig.legend(*ax1.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .175), ncol=2) #type: ignore

        plt.show()

    def __str__(self) -> str:
        return str({ "Learners": len(self.learners), "Simulations": len(self.simulations), "Batches": len(self.batches) })

    def __repr__(self) -> str:
        return str(self)

class Batcher(ABC):

    @staticmethod
    def from_json(json_val:Union[str, Dict[str, Any]]) -> 'Batcher':

        config = json.loads(json_val) if isinstance(json_val,str) else json_val

        if "count" in config:
            return CountBatcher.from_json(config)
        elif "size" in config:
            return SizeBatcher.from_json(config)
        elif "sizes" in config:
            return SizesBatcher.from_json(config)
        else:
            raise Exception("we were unable to determine an appropriate batching rule for the benchmark.")

    @abstractmethod
    def batch_sizes(self, n_interactions:int) -> Sequence[int]:
        ...

class CountBatcher(Batcher):

    @staticmethod
    def from_json(json_val:Union[str, Dict[str, Any]]) -> 'CountBatcher':
        config = json.loads(json_val) if isinstance(json_val,str) else json_val
        return CountBatcher(config["count"], config.get("min",0), config.get("max",math.inf))

    def __init__(self, batch_count:int, min_interactions:float = 1, max_interactions: float = math.inf) -> None:
        self._batch_count      = batch_count
        self._max_interactions = max_interactions
        self._min_interactions = min_interactions

    def batch_sizes(self, n_interactions: int) -> Sequence[int]:
        if n_interactions < self._min_interactions:
            return []

        n_interactions = int(min(n_interactions, self._max_interactions))

        batches   = [int(float(n_interactions)/(self._batch_count))] * self._batch_count
        remainder = n_interactions - sum(batches)
        for i in range(remainder): batches[int(i*len(batches)/remainder)] += 1

        return batches

class SizeBatcher(Batcher):

    @staticmethod
    def from_json(json_val:Union[str, Dict[str, Any]]) -> 'SizeBatcher':
        config = json.loads(json_val) if isinstance(json_val,str) else json_val
        return SizeBatcher(config["size"], config.get("min",0), config.get("max",math.inf))

    def __init__(self, batch_size:int, min_interactions:float = 1, max_interactions: float = math.inf) -> None:
        self._batch_size       = batch_size
        self._max_interactions = max_interactions
        self._min_interactions = min_interactions

    def batch_sizes(self, n_interactions: int) -> Sequence[int]:
        if n_interactions < self._min_interactions:
            return []

        n_interactions = int(min(n_interactions, self._max_interactions))

        return [self._batch_size] * int(n_interactions/self._batch_size)

class SizesBatcher(Batcher):

    @staticmethod
    def from_json(json_val:Union[str, Dict[str, Any]]) -> 'SizesBatcher':
        
        config = json.loads(json_val) if isinstance(json_val,str) else json_val
        return SizesBatcher(config["sizes"])

    def __init__(self, batch_sizes: Sequence[int]) -> None:
        self._batch_sizes = batch_sizes

    def batch_sizes(self, n_interactions: int) -> Sequence[int]:
        if sum(self._batch_sizes) > n_interactions:
            return []

        return self._batch_sizes

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
    def learners(learners: 'Sequence[BenchmarkLearner[_C, _A]]') -> Iterable[Any]:
        for index, learner in enumerate(learners):
            yield Transaction.learner(index, family=learner.family, full_name=learner.full_name, **learner.params)

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

class TaskSource(Source):
    
    def __init__(self, simulations, seeds, learners, restored) -> None:
        self._simulations = simulations
        self._seeds       = seeds
        self._learners    = learners
        self._restored    = restored

    def read(self) -> Iterable:
        simulations = dict(enumerate(self._simulations))
        seeds       = self._seeds
        learners    = dict(enumerate(self._learners))
        restored    = self._restored

        expected_batch_count = cast(Dict[Tuple[Any,Any,Any], int], collections.defaultdict(lambda:  int(-1)))
        restored_batch_count = cast(Dict[Tuple[Any,Any,Any], int], collections.defaultdict(lambda:  int( 0)))

        for k,v in restored.simulations.rows.items():
            expected_batch_count[k] = v[restored.simulations._columns.index('batch_count')]

        for k in restored.batches.rows:
            restored_batch_count[(k[1],k[2],k[0])] +=1

        is_not_complete = lambda t: restored_batch_count[t] != expected_batch_count[t[0]]
        
        task_keys       = filter(is_not_complete, product(simulations, seeds, learners))
        task_key_groups = groupby(task_keys, key=lambda t: t[0])

        seed_learner = lambda t: (t[1], t[2], learners[t[2]])
        
        for simulation_key, task_key_group in task_key_groups:
            yield simulation_key, simulations[simulation_key], list(map(seed_learner, task_key_group))
    
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
            with WithSimulation(simulation) as simulation:

                batch_sizes = self._batcher.batch_sizes(len(simulation.interactions))

                yield Transaction.simulation(simulation_index,
                    interaction_count = sum(batch_sizes[int(self._ignore_first):]),
                    batch_count       = len(batch_sizes[int(self._ignore_first):]),
                    context_size      = int(median(self._context_sizes(simulation))),
                    action_count      = int(median(self._action_counts(simulation))))

                for (seed, learner_index, learner_template) in task[2]:

                    learner = deepcopy(learner_template)
                    learner.init()

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

    def _shuffle_batch(self, interactions, seed, batch_sizes: Sequence[int]):
        batch_slices = list(accumulate([0] + list(batch_sizes))) #type: ignore
        interactions = CobaRandom(seed).shuffle(interactions) if seed else interactions
        
        for i in range(len(batch_slices)-1):
            yield islice(interactions, batch_slices[i], batch_slices[i+1])

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

                        values['reward'] = round(values['reward', 5])

                        promoted_items.append([transaction[0], index, values])

                items   = promoted_items
                version = 1

        return items

class TransactionIsNew(Filter):

    def __init__(self, existing: Result):

        self._existing = existing

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        for item in items:

            tipe  = item[0]

            if tipe == "version" and self._existing.version is not None:
                continue

            if tipe == "benchmark" and len(self._existing.benchmark) != 0:
                continue

            if tipe == "B" and item[1] in self._existing.batches:
                continue

            if tipe == "S" and item[1] in self._existing.simulations:
                continue

            if tipe == "L" and item[1] in self._existing.learners:
                continue

            yield item

class TransactionSink(Sink):
    
    def __init__(self, transaction_log: Optional[str], restored: Result) -> None:
        self._sink = Pipe.join([JsonEncode()], DiskSink(transaction_log)) if transaction_log else MemorySink()
        self._sink = Pipe.join([TransactionIsNew(restored)], self._sink)

    def write(self, items: Sequence[Any]) -> None:
        self._sink.write(items)

    @property
    def result(self) -> Result:
        if isinstance(self._sink, Pipe.FiltersSink):
            final_sink = self._sink.final_sink()
        else:
            final_sink = self._sink

        if isinstance(final_sink, MemorySink):
            return Result.from_transactions(cast(Iterable[Any], final_sink.items))

        if isinstance(final_sink, DiskSink):
            return Result.from_transaction_log(final_sink.filename)

        raise Exception("Transactions were written to an unrecognized sink.")

class WithSimulation:
    def __init__(self, simulation: Simulation) -> None:
        self._simulation = simulation

    def __enter__(self) -> Simulation:
        with ExecutionContext.Logger.log(f"loading simulation..."):
            try:
                return self._simulation.__enter__() #type: ignore
            except AttributeError:
                return self._simulation

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        try:
            self._simulation.__exit__(exception_type, exception_value, traceback) #type: ignore
        except AttributeError:
            pass

class BenchmarkLearner(Learner[_C,_A]):
    
    @property
    def family(self) -> str:
        try:
            return self._learner.family
        except AttributeError:
            return self._learner.__class__.__name__

    @property
    def params(self) -> Dict[str, Any]:
        try:
            return self._learner.params
        except AttributeError:
            return {}
    
    @property
    def full_name(self) -> str:
        if len(self.params) > 0:
            return f"{self.family}({','.join(f'{k}={v}' for k,v in self.params.items())})"
        else:
            return self.family

    def __init__(self, learner: Learner[_C,_A]) -> None:
        self._learner = learner

    def init(self) -> None:
        try:
            self._learner.init()
        except AttributeError:
            pass

    def choose(self, key: Key, context: _C, actions: Sequence[_A]) -> Choice:
        return self._learner.choose(key, context, actions)
    
    def learn(self, key: Key, context: _C, action: _A, reward: Reward) -> None:
        self._learner.learn(key, context, action, reward)

class Benchmark(Generic[_C,_A]):
    """An on-policy Benchmark using samples drawn from simulations to estimate performance statistics."""

    @staticmethod
    def from_file(filename:str) -> 'Benchmark[Context,Action]':
        """Instantiate a Benchmark from a config file."""

        suffix = Path(filename).suffix
        
        if suffix == ".json":
            return Benchmark.from_json(Path(filename).read_text())

        raise Exception(f"The provided file type ('{suffix}') is not a valid format for benchmark configuration")

    @staticmethod
    def from_json(json_val:Union[str, Dict[str,Any]]) -> 'Benchmark[Context,Action]':
        """Create a UniversalBenchmark from json text or object.

        Args:
            json_val: Either a json string or the decoded json object.

        Returns:
            The UniversalBenchmark representation of the given JSON string or object.
        """

        if isinstance(json_val, str):
            config = cast(Dict[str,Any], json.loads(json_val))
        else:
            config = json_val

        config = ExecutionContext.Templating.parse(config)

        if not isinstance(config["simulations"], collections.Sequence):
            config["simulations"] = [ config["simulations"] ]

        simulations  = [ JsonSimulation(sim_config) for sim_config in config["simulations"] ]
        batcher      = Batcher.from_json(config.get("batches","{'size': 1}"))
        ignore_first = config.get("ignore_first", True)
        ignore_raise = config.get("ignore_raise", True)
        shuffle      = config.get("shuffle", [None])

        return Benchmark(simulations, batcher, ignore_first=ignore_first, ignore_raise=ignore_raise, shuffle_seeds=shuffle)

    @overload
    def __init__(self, 
        simulations : Sequence[Simulation[_C,_A]], 
        *,
        batch_count: int,
        min_interactions: float = 1,
        max_interactions: float = math.inf,
        ignore_first: bool = True,
        ignore_raise: bool = True,
        shuffle_seeds: Sequence[Optional[int]] = [None],
        processes: int = None,
        maxtasksperchild: int = None) -> None: ...

    @overload
    def __init__(self, 
        simulations : Sequence[Simulation[_C,_A]], 
        *,
        batch_sizes: Sequence[int],
        ignore_first: bool = True,
        ignore_raise: bool = True,
        shuffle_seeds: Sequence[Optional[int]] = [None],
        processes: int = None,
        maxtasksperchild: int = None) -> None: ...

    @overload
    def __init__(self, 
        simulations : Sequence[Simulation[_C,_A]],
        *,
        batch_size: int,
        min_interactions: float = 1,
        max_interactions: float = math.inf,
        ignore_first: bool = True,
        ignore_raise: bool = True,
        shuffle_seeds: Sequence[Optional[int]] = [None],
        processes: int = None,
        maxtasksperchild: int = None) -> None: ...

    @overload
    def __init__(self,
        simulations : Sequence[Simulation[_C,_A]], 
        batcher: Batcher,
        *,
        ignore_first: bool = True,
        ignore_raise: bool = True,
        shuffle_seeds: Sequence[Optional[int]] = [None],
        processes: int = None,
        maxtasksperchild: int = None) -> None: ...

    def __init__(self,*args, **kwargs) -> None:
        """Instantiate a UniversalBenchmark.

        Args:
            simulations: The sequence of simulations to benchmark against.
            batcher: How each simulation is broken into evaluation batches.
            ignore_first: Should the first batch should be excluded from Result .
            ignore_raise: Should exceptions be raised or logged during evaluation.
            shuffle_seeds: A sequence of seeds for interaction shuffling. None means no shuffle.
            processes: The number of process to spawn during evalution (overrides coba config).
            maxtasksperchild: The number of tasks each process will perform before a refresh.
        
        See the overloads for more information.
        """

        self._simulations = args[0]
        
        self._batcher: Batcher

        if 'batch_count' in kwargs:
            self._batcher = CountBatcher(
                kwargs['batch_count'],
                kwargs.get('min_interactions',1),
                kwargs.get('max_interactions', math.inf))
        elif 'batch_size' in kwargs:
            self._batcher = SizeBatcher(
                kwargs['batch_size'],
                kwargs.get('min_interactions',1),
                kwargs.get('max_interactions', math.inf))
        elif 'batch_sizes' in kwargs:
            self._batcher = SizesBatcher(kwargs['batch_sizes'])
        else:
            self._batcher = args[1]

        self._ignore_first     = kwargs.get('ignore_first', True)
        self._ignore_raise     = kwargs.get('ignore_raise', True)
        self._seeds            = kwargs.get('shuffle_seeds', [None])
        self._processes        = kwargs.get('processes', None)
        self._maxtasksperchild = kwargs.get('maxtasksperchild', None)

    def ignore_raise(self, value:bool=True) -> 'Benchmark[_C,_A]':
        self._ignore_raise = value
        return self

    def ignore_first(self, value:bool=True) -> 'Benchmark[_C,_A]':
        self._ignore_first = value
        return self

    def processes(self, value:int) -> 'Benchmark[_C,_A]':
        self._processes = value
        return self

    def maxtasksperchild(self, value:int) -> 'Benchmark[_C,_A]':
        self._maxtasksperchild = value
        return self

    def evaluate(self, learners: Sequence[Learner[_C,_A]], transaction_log:str = None) -> Result:
        """Collect observations of a Learner playing the benchmark's simulations to calculate Results.

        Args:
            factories: See the base class for more information.

        Returns:
            See the base class for more information.
        """
        bench_learners       = [ BenchmarkLearner(learner) for learner in learners ] #type: ignore
        restored             = Result.from_transaction_log(transaction_log)
        task_source          = TaskSource(self._simulations, self._seeds, bench_learners, restored)
        task_to_transactions = TaskToTransactions(self._ignore_first, self._ignore_raise, self._batcher)
        transaction_sink     = TransactionSink(transaction_log, restored)

        n_given_learners    = len(bench_learners)
        n_given_simulations = len(self._simulations)
        n_given_seeds       = len(self._seeds)
        given_batcher       = self._batcher.__class__.__name__
        ignore_first        = self._ignore_first
 
        if len(restored.benchmark) != 0:
            assert n_given_learners    == restored.benchmark['n_learners'   ], "The currently evaluating benchmark doesn't match the given transaction log"
            assert n_given_simulations == restored.benchmark['n_simulations'], "The currently evaluating benchmark doesn't match the given transaction log"
            assert n_given_seeds       == restored.benchmark['n_seeds'      ], "The currently evaluating benchmark doesn't match the given transaction log"
            assert given_batcher       == restored.benchmark['batcher'      ], "The currently evaluating benchmark doesn't match the given transaction log"
            assert ignore_first        == restored.benchmark['ignore_first' ], "The currently evaluating benchmark doesn't match the given transaction log"

        preamble_transactions = []
        preamble_transactions.append(Transaction.version(TransactionPromote.CurrentVersion))
        preamble_transactions.append(Transaction.benchmark(n_given_learners, n_given_simulations, n_given_seeds, given_batcher, ignore_first))
        preamble_transactions.extend(Transaction.learners(bench_learners))

        mp = self._processes if self._processes else ExecutionContext.Config.processes
        mt = self._maxtasksperchild if self._maxtasksperchild else ExecutionContext.Config.maxtasksperchild
        
        Pipe.join(MemorySource(preamble_transactions), []                    , transaction_sink).run(1,None)
        Pipe.join(task_source                        , [task_to_transactions], transaction_sink).run(mp,mt)

        return transaction_sink.result