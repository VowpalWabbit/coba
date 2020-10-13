"""The benchmarks module contains core benchmark functionality and protocols.

This module contains the abstract interface expected for Benchmark implementations. This 
module also contains several Benchmark implementations and Result data transfer class.

TODO Add docstrings to Batchers
"""

import json
import collections

from abc import ABC, abstractmethod
from itertools import count, repeat, groupby, product, islice, accumulate
from statistics import median
from ast import literal_eval
from pathlib import Path
from typing import (
    Iterable, Tuple, Hashable, Union, Sequence, List, Callable, 
    Generic, TypeVar, Dict, Any, cast, Type, Optional
)

from coba.simulations import Interaction, LazySimulation, Simulation, Context, Action, Key, Choice, Reward
from coba.preprocessing import Batcher
from coba.learners import Learner
from coba.execution import ExecutionContext
from coba.statistics import BatchMeanEstimator, StatisticalEstimate
from coba.utilities import check_pandas_support
from coba.json import CobaJsonDecoder, CobaJsonEncoder, JsonSerializable
from coba.data import AsyncFileWriter
from coba.random import Random

_K  = TypeVar("_K", bound=Hashable)
_C  = TypeVar('_C', bound=Context)
_A  = TypeVar('_A', bound=Action)
_T  = TypeVar('_T')

class Table(JsonSerializable, Generic[_K]):
    """A container class for storing tabular data."""

    def __init__(self, name:str, default: Any=float('nan')):
        """Instantiate a Table.
        
        Args:
            name: The name of the table.
            default: The default values to fill in missing values with
        """
        self._name    = name
        self._default = default

        self._columns: List[str]               = []
        self._rows   : Dict[_K, Dict[str,Any]] = {}

    def add_row(self, key: _K, **kwargs) -> None:
        """Add an indexed row of data to the table.
        
        Arg:
            key: A lookup index for the row in the table
            kwargs: The row of data in `column_name`:`value` format.

        Remarks:
            When a row is added all rows are updated (including the added row
            if necessary) to make sure that all rows have the same columns.

        """
        new_columns = [col for col in kwargs if col not in self._columns]

        if new_columns:
            self._columns.extend(new_columns)
            for data in self._rows.values():
                data.update(zip(new_columns, repeat(self._default)))
        
        self._rows[key] = collections.OrderedDict({key:kwargs.get(key,self._default) for key in self._columns})

    def rmv_row(self, key: _K) -> None:
        self._rows.pop(key, None)

    def to_tuples(self) -> Sequence[Any]:
        """Convert a table into a sequence of namedtuples."""
        return list(self.to_indexed_tuples().values())

    def to_indexed_tuples(self) -> Dict[_K, Any]:
        """Convert a table into a mapping of keys to tuples."""

        my_type = collections.namedtuple(self._name, self._columns) #type: ignore #mypy doesn't like dynamic named tuples
        return { key:my_type(**value) for key,value in self._rows.items() } #type: ignore #mypy doesn't like dynamic named tuples

    def to_pandas(self) -> Any:
        """Convert a table into a pandas dataframe."""

        check_pandas_support('Table.to_pandas')
        import pandas as pd #type: ignore #mypy complains otherwise

        return pd.DataFrame(self.to_tuples())

    def __contains__(self, item) -> bool:
        return item in self._rows

    def __str__(self) -> str:
        return str({"Table": self._name, "Columns": self._columns, "Rows": len(self._rows)})

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def __from_json_obj__(json_obj: Dict[str,Any]) -> 'Table[Hashable]':
        rows    = { literal_eval(key):value for key,value in json_obj['rows'].items() }
        columns = json_obj['columns']

        obj          = Table[Hashable](json_obj['name'])
        obj._columns = columns
        obj._rows    = rows

        return obj

    def __to_json_obj__(self) -> Dict[str,Any]:

        literal_evalable = lambda key: str(key) if not isinstance(key, str) else f"'{key}'"

        return {
            'name'   : self._name,
            'columns': self._columns,
            'rows'   : { literal_evalable(key):value for key,value in self._rows.items() }
        }

class ResultWriter(ABC):

    def write_learner(self, learner_id:int, **kwargs):
        """Write learner metadata row to Result.
        
        Args:
            learner_id: The primary key for the given learner.
            kwargs: The metadata to store about the learner.
        """

        key       = learner_id
        key_items = [(cast(str,"learner_id"),learner_id)]
        row       = collections.OrderedDict(key_items + list(kwargs.items()))

        self._write("L", key, row)

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

        self._write("S", key, row)

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

        self._write("B", key, row)

    def __enter__(self) -> 'ResultWriter':
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        pass

    @abstractmethod
    def _write(self, name: str, key: Hashable, row: Dict[str,Any]) -> None:
        ...

class ResultDiskWriter(ResultWriter):

    def __init__(self, filename:str) -> None:
        self._json_encoder = CobaJsonEncoder()
        self._transactions_path = Path(filename)
        self._transactions_path.touch()
        self._async_file_writer = AsyncFileWriter(self._transactions_path, 'a')

    def __enter__(self) -> 'ResultDiskWriter':
        self._async_file_writer.open()
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        self._async_file_writer.close()

    def flush(self) -> None:
        self._async_file_writer.flush()

    def _write(self, name: str, key: Hashable, row: Dict[str,Any]):
        self._async_file_writer.async_write(self._json_encoder.encode([name,(key,row)]))
        self._async_file_writer.async_write("\n")

class ResultMemoryWriter(ResultWriter):

    def __init__(self, default: Any = float('nan')) -> None:
        self._learner_table    = Table[int]                             ("Learners"   , default)
        self._simulation_table = Table[int]                             ("Simulations", default)
        self._batch_table      = Table[Tuple[int,int,Optional[int],int]]("Batches"    , default)

    def _write(self, name: str, key: Any, row: Dict[str,Any]):
        if name == "L":
            self._learner_table.add_row(key, **row)

        if name == "S":
            self._simulation_table.add_row(key, **row)

        if name == "B":
            self._batch_table.add_row(key, **row)

class Result(JsonSerializable):
    """A class for creating and returning the result of a Benchmark evaluation."""

    @staticmethod
    def from_result_writer(result_writer: ResultWriter, default: Any = float('nan')) -> 'Result':
        if isinstance(result_writer, ResultMemoryWriter):
            return Result(
                result_writer._learner_table, 
                result_writer._simulation_table, 
                result_writer._batch_table
            )
        
        if isinstance(result_writer, ResultDiskWriter):
            return Result.from_transaction_log(str(result_writer._transactions_path), default)

        raise Exception(f"The given result_writer ({result_writer.__class__.__name__}) wasn't recognized.")

    @staticmethod
    def from_transaction_log(filename:Optional[str], default: Any = float('nan')) -> 'Result':
        """Create a Result from a transaction file."""
        
        if filename is None or not Path(filename).exists(): return Result()

        decoder          = CobaJsonDecoder()
        learner_table    = Table[int]                             ("Learners"   , default)
        simulation_table = Table[int]                             ("Simulations", default)
        batch_table      = Table[Tuple[int,int,Optional[int],int]]("Batches"    , default)

        lines = Path(filename).read_text().split("\n")

        for line in [ l for l in lines if l != '']:
            json_obj: Tuple[str,Tuple[Any, Dict[str,Any]]] = decoder.decode(line, [StatisticalEstimate])

            table: Union[Table[int], Table[Tuple[int,int,Optional[int],int]]]
            key  : Any 

            if json_obj[0] == "L":
                table = learner_table
                key   = json_obj[1][0]
            elif json_obj[0] == "S":
                table = simulation_table
                key   = json_obj[1][0]
            else:
                table = batch_table
                key   = tuple(json_obj[1][0])

            table.add_row(key, **json_obj[1][1])
        
        return Result(learner_table, simulation_table, batch_table)

    @staticmethod
    def from_json_file(filename:str) -> 'Result':
        """Create a Result from a json file."""
        needed_types: Sequence[Type[JsonSerializable]] = [Result, Table, StatisticalEstimate]
        return CobaJsonDecoder().decode(Path(filename).read_text(), needed_types)

    def to_json_file(self, filename:str) -> None:
        """Write a Result to a json file."""
        Path(filename).write_text(CobaJsonEncoder().encode(self))

    def __init__(self, 
        learner_table   : Table[int] = None, 
        simulation_table: Table[int] = None, 
        batch_table     : Table[Tuple[int,int,Optional[int],int]] = None) -> None:
        """Instantiate a Result class."""

        self._learner_table    = learner_table    if learner_table    else Table[int]("Learners")
        self._simulation_table = simulation_table if simulation_table else Table[int]("Simulations")
        self._batch_table      = batch_table      if batch_table      else Table[Tuple[int,int,Optional[int],int]]("Batches")

    def has_learner(self, learner_id:int) -> bool:
        return learner_id in self._learner_table

    def has_simulation(self, simulation_id:int) -> bool:
        return simulation_id in self._simulation_table

    def has_batch(self, learner_id:int, simulation_id:int, seed:Optional[int], batch_index:int) -> bool:
        return (learner_id, simulation_id, seed, batch_index) in self._batch_table

    def get_learner(self, learner_id:int) -> Dict[str,Any]:
        return self._learner_table._rows[learner_id]

    def get_simulation(self, simulation_id:int) -> Dict[str,Any]:
        return self._simulation_table._rows[simulation_id]

    def get_batch(self, learner_id:int, simulation_id:int, seed:Optional[int], batch_index:int) -> Dict[str,Any]:
        return self._batch_table._rows[(learner_id, simulation_id, seed, batch_index)]

    def rmv_learner(self, learner_id:int):
        self._learner_table.rmv_row(learner_id)
    
    def rmv_simulation(self, simulation_id:int):
        self._simulation_table.rmv_row(simulation_id)

    def rmv_batches(self, simulation_id:int):
        if not simulation_id in self._simulation_table: return
        for key in [ key for key in self._batch_table._rows if key[1] == simulation_id ]: self._batch_table.rmv_row(key)

    def to_tuples(self) -> Tuple[Sequence[Any], Sequence[Any], Sequence[Any]]:
        return (
            self._learner_table.to_tuples(),
            self._simulation_table.to_tuples(),
            self._batch_table.to_tuples()
        )

    def to_indexed_tuples(self) -> Tuple[Dict[int,Any], Dict[int,Any], Dict[Tuple[int,int,Optional[int],int],Any]]:
        return (
            self._learner_table.to_indexed_tuples(),
            self._simulation_table.to_indexed_tuples(),
            self._batch_table.to_indexed_tuples()
        )

    def to_pandas(self) -> Tuple[Any,Any,Any]:
        l = self._learner_table.to_pandas()
        s = self._simulation_table.to_pandas()
        b = self._batch_table.to_pandas()

        b.reward = b.reward.astype('float')

        return (l,s,b)

    @staticmethod
    def __from_json_obj__(obj:Dict[str,Any]) -> 'Result':
        return Result(
            obj['learner_table'],
            obj['simulation_table'],
            obj['batch_table']
        )

    def __to_json_obj__(self) -> Dict[str,Any]:
        return {
            'simulation_table': self._simulation_table,
            'learner_table'   : self._learner_table,
            'batch_table'     : self._batch_table
        }

    def __str__(self) -> str:
        return str({
            "Learners": len(self._learner_table._rows),
            "Simulations": len(self._simulation_table._rows),
            "Batches": len(self._batch_table._rows)
        })

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

class BenchmarkSimulation(Simulation[_C, _A]):
    
    def __init__(self, 
        index       :int, 
        seed        : Optional[int], 
        batch_sizes : Sequence[int],
        simulation  : Simulation[_C,_A]) -> None:

        self.index        = index
        self.seed         = seed
        self.batch_slices = list(accumulate([0] + list(batch_sizes)))

        self._interactions = Random(seed).shuffle(simulation.interactions) if seed else simulation.interactions
        self._interactions = self._interactions[0:sum(batch_sizes)]
        self._rewards      = simulation.rewards

    @property
    def batches(self) -> Iterable[Iterable[Interaction[_C, _A]]]:
        for i in range(len(self.batch_slices)-1):
            yield islice(self._interactions, self.batch_slices[i], self.batch_slices[i+1])

    @property
    def interactions(self) -> Sequence[Interaction[_C, _A]]:
        return self._interactions

    def rewards(self, choices: Sequence[Tuple[Key, Choice]]) -> Sequence[Reward]:
        return self._rewards(choices)

class BenchmarkLearner(Learner[_C, _A]):
    
    def __init__(self, index:int, factory: Callable[[],Learner[_C,_A]]) -> None:
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

    def choose(self, key: Key, context: _C, actions: Sequence[_A]) -> Choice:
        return self._choose(key, context, actions)
    
    def learn(self, key: Key, context: _C, action: _A, reward: Reward) -> None:
        self._learn(key, context, action, reward)

class UniversalBenchmark(Benchmark[_C,_A]):
    """An on-policy Benchmark using samples drawn from simulations to estimate performance statistics."""

    @staticmethod
    def from_file(filename:str) -> 'UniversalBenchmark':
        """Instantiate a Benchmark from a config file."""

        suffix = Path(filename).suffix
        
        if suffix == ".json":
            return UniversalBenchmark.from_json(Path(filename).read_text())

        raise Exception(f"The provided file type ('{suffix}') is not a valid format for benchmark configuration")

    @staticmethod
    def from_json(json_val:Union[str, Dict[str,Any]]) -> 'UniversalBenchmark':
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
        shuffle      = config.get("seeds", [None])

        return UniversalBenchmark(simulations, batcher, ignore_first, ignore_raise, shuffle)

    def __init__(self,
        simulations : Sequence[Simulation[_C,_A]], 
        batcher: Batcher,
        ignore_first: bool = True,
        ignore_raise: bool = True,
        shuffle_seeds: Sequence[Optional[int]] = [None]) -> None:
        """Instantiate a UniversalBenchmark.

        Args:
            simulations: A sequence of simulations to benchmark against.
            batcher: Determines how each simulation is broken into evaluation/learning batches.
            ignore_first: Determines if the first batch should be ignored since no learning has occured yet.
            ignore_raise: Determines if exceptions during benchmark evaluation are raised or simply logged.
            shuffle_seeds: A sequence of seeds to shuffle simulations by when evaluating. None means no shuffle.
        
        See the overloads for more information.
        """

        self._simulations   = simulations
        self._batcher       = batcher
        self._ignore_first  = ignore_first
        self._ignore_raise  = ignore_raise
        self._seeds         = shuffle_seeds

    def ignore_raise(self, value:bool=True) -> 'UniversalBenchmark[_C,_A]':
        return UniversalBenchmark(self._simulations, self._batcher, self._ignore_first, value, self._seeds)

    def ignore_first(self, value:bool=True) -> 'UniversalBenchmark[_C,_A]':
        return UniversalBenchmark(self._simulations, self._batcher, value, self._ignore_raise, self._seeds)

    def evaluate(self, learner_factories: Sequence[Callable[[],Learner[_C,_A]]], transaction_log:str = None) -> Result:
        """Collect observations of a Learner playing the benchmark's simulations to calculate Results.

        Args:
            learner_factories: See the base class for more information.

        Returns:
            See the base class for more information.
        """
        
        restored = Result.from_transaction_log(transaction_log)
        results  = ResultDiskWriter(transaction_log) if transaction_log else ResultMemoryWriter()

        n_restored_learners = len(restored._learner_table._rows)
        n_given_learners    = len(learner_factories)

        if n_restored_learners > 0 and n_restored_learners != n_given_learners:
            raise Exception("The number of learners differs from the transaction log.")

        with results:

            #write number of learners, number of simulations, batcher, shuffle seeds and ignore first
            #make sure all these variables are the same. If they've changed then fail gracefully.                    

            for simulation in self._make_simulations(self._simulations, restored, results):
                self._handle_exceptions(
                    lambda: self._process_simulation(simulation, learner_factories, restored, results)
                )

        return Result.from_result_writer(results)

    def _make_simulations(self, simulations, restored, results) -> Iterable[BenchmarkSimulation[_C, _A]]:
        for index, simulation in enumerate(simulations):
            try:
                if self._simulation_finished_in_restored(restored, index): continue

                with self._lazy_simulation(simulation) as loaded_simulation:

                    batch_sizes = self._batcher.batch_sizes(len(loaded_simulation.interactions))

                    if not restored.has_simulation(index):
                        results.write_simulation(index,
                            interaction_count = sum(batch_sizes[int(self._ignore_first):]),
                            batch_count       = len(batch_sizes[int(self._ignore_first):]),
                            seed_count        = len(self._seeds),
                            context_size      = int(median(self._context_sizes(loaded_simulation))),
                            action_count      = int(median(self._action_counts(loaded_simulation)))
                        )

                    for seed in self._seeds: yield BenchmarkSimulation(index, seed, batch_sizes, simulation)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                ExecutionContext.Logger.log_exception(e, "unhandled exception:")
                if not self._ignore_raise: raise e

    def _make_learners(self, factories, simulation, restored, results) -> Iterable[BenchmarkLearner[_C,_A]]:
        for learner in map(BenchmarkLearner, *zip(*enumerate(factories))):

            if not restored.has_learner(learner.index):
                learner_row = {"family":learner.family, "full_name": learner.full_name, **learner.params}
                results.write_learner(learner.index, **learner_row)

            if not self._simulation_learner_finished_in_restored(restored, simulation, learner.index):
                yield learner

    def _process_simulation(self, simulation, factories, restored, results):

        for learner in self._make_learners(factories, simulation, restored, results):
            ExecutionContext.Logger.log(f"Processing simulation ({simulation.index},{simulation.seed}) with {learner.full_name}")
            self._handle_exceptions(
                lambda: self._process_learner(simulation, learner, restored, results)
            )

        restored.rmv_batches(simulation.index) # to reduce memory in case we restore a large log

    def _process_learner(self, simulation, learner, restored, results)-> None:
        for batch in enumerate(simulation.batches):
            self._process_batch(simulation, learner, batch, restored, results)

    def _process_batch(self, simulation, learner, batch, restored, results) -> None:
        
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

        if batch_index >= 0 and not restored.has_batch(learner.index, simulation.index, simulation.seed, batch_index):
            row = {"N":len(rewards), "reward":BatchMeanEstimator(rewards)}
            results.write_batch(learner.index, simulation.index, simulation.seed, batch_index, **row)

    #Begin utility classes
    def _handle_exceptions(self, func: Callable[[], _T]) -> _T:
        try:
            func()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            ExecutionContext.Logger.log_exception(e, "unhandled exception:")
            if not self._ignore_raise: raise e

    def _simulation_finished_in_restored(self, restored_result: Result, simulation_index: int) -> bool:

        if not restored_result.has_simulation(simulation_index):
            return False #this simulation has never been processed

        n_learners = len(restored_result._learner_table._rows)
        n_batches  = restored_result.get_simulation(simulation_index)['batch_count']
        n_seeds    = len(self._seeds)

        total_batch_count    = n_learners * 1 * n_seeds * n_batches
        restored_batch_count = 0

        for learner_index, shuffle_seed, batch_index in product(range(n_learners), self._seeds, range(n_batches)):
            restored_batch_count += int(restored_result.has_batch(learner_index, simulation_index, shuffle_seed, batch_index))

        return restored_batch_count == total_batch_count

    def _simulation_learner_finished_in_restored(self, restored, simulation, learner_index) -> bool:

        if not restored.has_simulation(simulation.index):
            return False #this simulation has never been processed

        restored_simulation = restored.get_simulation(simulation.index)

        if restored_simulation['batch_count'] == 0:
            return True #this simulation was previously processed and found to be too small to batch

        fully_evaluated_batch_count = restored_simulation['batch_count']
        restored_batch_count        = 0

        for batch_index in range(restored_simulation['batch_count']):
            restored_batch_count += int(restored.has_batch(learner_index, simulation.index, simulation.seed, batch_index))

        # true if all batches were evaluated previously
        return restored_batch_count == fully_evaluated_batch_count

    def _lazy_simulation(self, simulation: Simulation) -> LazySimulation:
        return simulation if isinstance(simulation, LazySimulation) else LazySimulation(lambda: simulation)

    def _context_sizes(self, simulation: Simulation) -> Iterable[int]:
        for context in [i.context for i in simulation.interactions]:
            yield 0 if context is None else len(context) if isinstance(context,tuple) else 1
    
    def _action_counts(self, simulation: Simulation) -> Iterable[int]:
        for actions in [i.actions for i in simulation.interactions]:
            yield len(actions)