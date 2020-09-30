"""The benchmarks module contains core benchmark functionality and protocols.

This module contains the abstract interface expected for Benchmark implementations. This 
module also contains several Benchmark implementations and Result data transfer class.
"""

import json
import collections

from abc import ABC, abstractmethod
from itertools import count, repeat, groupby
from statistics import median
from ast import literal_eval
from pathlib import Path
from typing import (
    Iterable, Tuple, Hashable, Union, Sequence, List, Callable, 
    Generic, TypeVar, Dict, Any, overload, cast, Type
)

from coba.simulations import Interaction, LazySimulation, Simulation, Context, Action
from coba.learners import Learner
from coba.execution import ExecutionContext, LoggedException
from coba.statistics import BatchMeanEstimator, StatisticalEstimate
from coba.utilities import check_pandas_support
from coba.json import CobaJsonDecoder, CobaJsonEncoder, JsonSerializable

_K = TypeVar("_K", bound=Hashable)
_C = TypeVar('_C', bound=Context)
_A = TypeVar('_A', bound=Action)
_C_inner = TypeVar('_C_inner', bound=Context)
_A_inner = TypeVar('_A_inner', bound=Action)

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

        self._columns     : List[str]               = []
        self._rows        : Dict[_K, Dict[str,Any]] = {}

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

class ResultWriter:

    def write_learner(self, learner_id:int, **kwargs):
        """Write learner metadata row to Result.
        
        Args:
            learner_id: The primary key for the given learner.
            kwargs: The metadata to store about the learner.
        """

        items = [("learner_id",learner_id)] + list(kwargs.items())  #type:ignore
        self._write("L", learner_id, collections.OrderedDict(items))

    def write_simulation(self, simulation_id: int, **kwargs):
        """Write simulation metadata row to Result.
        
        Args:
            simulation_id: The primary key for the given simulation.
            kwargs: The metadata to store about the learner.
        """    
        items = [("simulation_id",simulation_id)] + list(kwargs.items())  #type:ignore
        self._write("S", simulation_id, collections.OrderedDict(items))

    def write_batch(self, learner_id:int, simulation_id:int, batch_index:int, **kwargs):
        """Write batch metadata row to Result.
        
        Args:
            learner_id: The primary key for the learner we observed the batch for.
            simulation_id: The primary key for the simulation the batch came from.
            batch_index: The index of the batch within the simulation.
            kwargs: The metadata to store about the batch.
        """
        key       = (learner_id, simulation_id, batch_index)
        key_items = [("learner_id",learner_id), ("simulation_id",simulation_id), ("batch_index",batch_index)]
        row       = collections.OrderedDict(key_items + list(kwargs.items()))

        self._write("B", key, row)

    def _write(self, name: str, key: Hashable, row: Dict[str,Any]):
        pass

class ResultDiskWriter(ResultWriter):

    def __init__(self, filename:str) -> None:
        self._json_encoder = CobaJsonEncoder()
        self._transactions_path = Path(filename)
        self._transactions_path.touch()

    def _write(self, name: str, key: Hashable, row: Dict[str,Any]):
        with open(self._transactions_path, "a") as f:
            f.write(self._json_encoder.encode([name,(key,row)]))
            f.write("\n")

class ResultMemoryWriter(ResultWriter):

    def __init__(self, default: Any = float('nan')) -> None:
        self._learner_table    = Table[int]               ("Learners"   , default)
        self._simulation_table = Table[int]               ("Simulations", default)
        self._batch_table      = Table[Tuple[int,int,int]]("Batches"    , default)

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
            return Result.from_transaction_file(str(result_writer._transactions_path), default)

        raise Exception(f"The given result_writer ({result_writer.__class__.__name__}) wasn't recognized.")

    @staticmethod
    def from_transaction_file(filename:str, default: Any = float('nan')) -> 'Result':
        """Create a Result from a transaction file."""
        
        if not Path(filename).exists(): return Result()

        decoder          = CobaJsonDecoder()
        learner_table    = Table[int]               ("Learners"   , default)
        simulation_table = Table[int]               ("Simulations", default)
        batch_table      = Table[Tuple[int,int,int]]("Batches"    , default)

        lines = Path(filename).read_text().split("\n")

        for line in [ l for l in lines if l != '']:
            json_obj: Tuple[str,Tuple[Any, Dict[str,Any]]] = decoder.decode(line, [StatisticalEstimate])

            table: Union[Table[int], Table[Tuple[int,int,int]]]
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
        learner_table: Table[int] = None, 
        simulation_table: Table[int] = None, 
        batch_table:Table[Tuple[int,int,int]] = None) -> None:
        """Instantiate a Result class."""

        self._learner_table = Table[int]("Learners") if learner_table is None else learner_table
        self._simulation_table = Table[int]("Simulations") if simulation_table is None else simulation_table
        self._batch_table = Table[Tuple[int,int,int]]("Batches") if batch_table is None else batch_table

    def has_learner(self, learner_id:int) -> bool:
        return learner_id in self._learner_table

    def has_simulation(self, simulation_id:int) -> bool:
        return simulation_id in self._simulation_table

    def has_batch(self, learner_id:int, simulation_id:int, batch_index:int) -> bool:
        return (learner_id, simulation_id, batch_index) in self._batch_table

    def get_learner(self, learner_id:int) -> Dict[str,Any]:
        return self._learner_table._rows[learner_id]

    def get_simulation(self, simulation_id:int) -> Dict[str,Any]:
        return self._simulation_table._rows[simulation_id]

    def get_batch(self, learner_id:int, simulation_id:int, batch_index:int) -> Dict[str,Any]:
        return self._batch_table._rows[(learner_id, simulation_id, batch_index)]

    def rmv_learner(self, learner_id:int):
        self._learner_table.rmv_row(learner_id)
    
    def rmv_simulation(self, simulation_id:int):
        self._simulation_table.rmv_row(simulation_id)

    def rmv_batches(self, simulation_id:int):

        if not simulation_id in self._simulation_table: return

        n_learners = len(self._learner_table._rows)
        n_batches  = self.get_simulation(simulation_id)['batch_count']

        for batch_key in [ (l,simulation_id,b) for l in range(n_learners) for b in range(n_batches)]:
            self._batch_table.rmv_row(batch_key)

    def to_tuples(self) -> Tuple[Sequence[Any], Sequence[Any], Sequence[Any]]:
        return (
            self._learner_table.to_tuples(),
            self._simulation_table.to_tuples(),
            self._batch_table.to_tuples()
        )

    def to_indexed_tuples(self) -> Tuple[Dict[int,Any], Dict[int,Any], Dict[Tuple[int,int,int],Any]]:
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

class UniversalBenchmark(Benchmark[_C,_A]):
    """An on-policy Benchmark using samples drawn from simulations to estimate performance statistics."""

    class EvaluationContext(Generic[_C_inner,_A_inner]):
        """A class to maintain the state of the current evaluation."""
        restored_result: Result 
        result_writer    : ResultWriter
        simulations      : Sequence[Simulation[_C_inner,_A_inner]]
        learner_factories: Sequence[Callable[[],Learner[_C_inner,_A_inner]]]

        batch_sizes      : Sequence[int]
        batch_indexes    : Sequence[int]

        simulation_index: int
        simulation      : Simulation[_C_inner,_A_inner]

        learner_index   : int
        learner         : Learner[_C_inner,_A_inner]

        batch_index     : int
        batch           : Iterable[Tuple[int,Interaction[_C_inner,_A_inner]]]

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

        config = ExecutionContext.TemplatingEngine.parse(config)

        is_singular = isinstance(config["simulations"], dict)
        sim_configs = config["simulations"] if not is_singular else [ config["simulations"] ]

        #by default load simulations lazily
        for sim_config in sim_configs:
            if "lazy" not in sim_config:
                sim_config["lazy"] = True

        simulations = [ Simulation.from_json(sim_config) for sim_config in sim_configs ]

        if "count" in config["batches"]:
            return UniversalBenchmark(simulations, batch_count=config["batches"]["count"], ignore_first=config["ignore_first"])
        else:
            return UniversalBenchmark(simulations, batch_size=config["batches"]["size"], ignore_first=config["ignore_first"])

    @overload
    def __init__(self, 
        simulations: Sequence[Simulation[_C,_A]],
        *,
        batch_count : int,
        ignore_first: bool = True,
        ignore_raise: bool = True) -> None:
        """Instantiate a UniversalBenchmark.
        
        Args:
            simulations: A sequence of simulations to benchmark against.
            batch_count: How many batches per simulation to make during evaluation (batch_size will be spread evenly).
            ignore_first: Determines if the first batch should be ignored since no learning has occured yet.
            ignore_raise: Determines if exceptions during benchmark evaluation are passed or raised.
        """
        ...

    @overload
    def __init__(self, 
        simulations: Sequence[Simulation[_C,_A]],
        *, 
        batch_size  : Union[int, Sequence[int], Callable[[int],int]],
        ignore_first: bool = True,
        ignore_raise: bool = True) -> None:
        ...
        """Instantiate a UniversalBenchmark.

            Args:
                simulations: A sequence of simulations to benchmark against.
                batch_size: An indication of how large every batch should be. If batch_size is an integer
                    then simulations will run until completion with batches of the given size. If 
                    batch_size is a sequence of integers then `sum(batch_size)` interactions will be 
                    pulled from simulations and batched according to the sequence. If batch_size is a 
                    function of batch index then each batch size will be determined by a call to the function.
                ignore_first: Determines if the first batch should be ignored since no learning has occured yet.
                raise_except: Determines if exceptions during benchmark evaluation are passed or raised.
        """

    def __init__(self,
        simulations : Sequence[Simulation[_C,_A]], 
        *,
        batch_count : int = None, 
        batch_size  : Union[int, Sequence[int], Callable[[int],int]] = None,
        ignore_first: bool = True,
        ignore_raise: bool = True) -> None:
        """Instantiate a UniversalBenchmark.
        
        See the overloads for more information.
        """

        self._simulations  = simulations
        self._batch_count  = batch_count
        self._batch_size   = batch_size
        self._ignore_first = ignore_first
        self._ignore_raise = ignore_raise

    def ignore_raise(self, value:bool=True) -> 'UniversalBenchmark[_C,_A]':

        if self._batch_count is not None:
            return UniversalBenchmark(self._simulations, batch_count=self._batch_count, ignore_first=self._ignore_first, ignore_raise=value)

        if self._batch_size is not None:
            return UniversalBenchmark(self._simulations, batch_size=self._batch_size, ignore_first=self._ignore_first, ignore_raise=value)

        raise Exception("An invalid instantiation of UniversalBenchmark occured")

    def ignore_first(self, value:bool=True) -> 'UniversalBenchmark[_C,_A]':

        if self._batch_count is not None:
            return UniversalBenchmark(self._simulations, batch_count=self._batch_count, ignore_first=value, ignore_raise=self._ignore_raise)

        if self._batch_size is not None:
            return UniversalBenchmark(self._simulations, batch_size=self._batch_size, ignore_first=value, ignore_raise=self._ignore_raise)

        raise Exception("An invalid instantiation of UniversalBenchmark occured")

    def evaluate(self, learner_factories: Sequence[Callable[[],Learner[_C,_A]]], transaction_file:str = None) -> Result:
        """Collect observations of a Learner playing the benchmark's simulations to calculate Results.

        Args:
            learner_factories: See the base class for more information.

        Returns:
            See the base class for more information.
        """

        # using a context class to maintain the state of the evaluation
        # reduces the amount of parameters we need to pass/maintain but
        # has the negative side-effect of making dependencies less clear
        # I'm not sure which way is better. I think this code is more 
        # readable but perhaps harder for developers to debug or maintain?
        ec                   = UniversalBenchmark.EvaluationContext[_C,_A]()
        ec.learner_factories = learner_factories

        if transaction_file is not None:
            ec.restored_result = Result.from_transaction_file(transaction_file)
            ec.result_writer   = ResultDiskWriter(transaction_file)
        else:
            ec.restored_result = Result()
            ec.result_writer   = ResultMemoryWriter()

        for ec.learner_index, ec.learner in enumerate(f() for f in learner_factories):
            if not ec.restored_result.has_learner(ec.learner_index):
                ec.result_writer.write_learner(ec.learner_index, 
                    family    = self._safe_family(ec.learner),
                    full_name = self._safe_full(ec.learner),
                    **self._safe_params(ec.learner)
                )

        self._process_simulations(ec)

        return Result.from_result_writer(ec.result_writer)

    #Begin evaluation classes. These are called in a waterfall pattern.
    def _process_simulations(self, ec: 'UniversalBenchmark.EvaluationContext'):
        for ec.simulation_index, ec.simulation in enumerate(self._simulations):
            with ExecutionContext.Logger.log(f"evaluating simulation {ec.simulation_index}..."):
                try:
                    self._process_simulation(ec)
                    ec.restored_result.rmv_batches(ec.simulation_index)

                except KeyboardInterrupt:
                    raise
                except LoggedException as e:
                    if not self._ignore_raise: raise
                except Exception as e:
                    ExecutionContext.Logger.log(f"unhandled exception: {e}")
                    if not self._ignore_raise: raise

    def _process_simulation(self, ec: 'UniversalBenchmark.EvaluationContext'):

        max_batch_index  = self._max_batch_index(ec.simulation_index, ec.restored_result)
        max_batch_tuples = zip(range(len(ec.learner_factories)), repeat(ec.simulation_index), repeat(max_batch_index))

        if all(ec.restored_result.has_batch(*max_batch_tuple) for max_batch_tuple in max_batch_tuples):
            return # we already have all the batches done so we can skip this simulation

        with self._lazy_simulation(ec.simulation) as ec.simulation:
            ec.batch_sizes   = self._batch_sizes(len(ec.simulation.interactions))
            ec.batch_indexes = [b for index,size in enumerate(ec.batch_sizes) for b in repeat(index,size)]

            assert not any([ s == 0 for s in ec.batch_sizes]), "The simulation was not large enough to fill all batches."

            if not ec.restored_result.has_simulation(ec.simulation_index):
                
                int_ignore_first = int(self._ignore_first)

                ec.result_writer.write_simulation(ec.simulation_index,
                    interaction_count = sum(ec.batch_sizes[int_ignore_first:]),
                    batch_count       = len(ec.batch_sizes[int_ignore_first:]),
                    context_size      = median(self._context_sizes(ec.simulation)),
                    action_count      = median(self._action_counts(ec.simulation))
                )

            self._process_learners(ec)

    def _process_learners(self, ec: 'UniversalBenchmark.EvaluationContext'):
        with ExecutionContext.Logger.log(f"evaluating learners..."):
            max_batch_index  = self._max_batch_index(ec.simulation_index, ec.restored_result)
            for ec.learner_index, ec.learner in enumerate(f() for f in ec.learner_factories):
                if not ec.restored_result.has_batch(ec.learner_index, ec.simulation_index, max_batch_index):
                    self._process_learner(ec)

    def _process_learner(self, ec: 'UniversalBenchmark.EvaluationContext'):
        with ExecutionContext.Logger.log(f"evaluating {self._safe_full(ec.learner)}..."):
            self._process_batches(ec)

    def _process_batches(self, ec: 'UniversalBenchmark.EvaluationContext'):
        for ec.batch_index, ec.batch in groupby(zip(ec.batch_indexes, ec.simulation.interactions), lambda t: t[0]):
            self._process_batch(ec)

    def _process_batch(self, ec: 'UniversalBenchmark.EvaluationContext'):
        keys     = []
        contexts = []
        choices  = []
        actions  = []

        if self._ignore_first:
            ec.batch_index -= 1

        for _, interaction in ec.batch:

            choice = ec.learner.choose(interaction.key, interaction.context, interaction.actions)

            assert choice in range(len(interaction.actions)), "An invalid action was chosen by the learner"

            keys    .append(interaction.key)
            contexts.append(interaction.context)
            choices .append(choice)
            actions .append(interaction.actions[choice])

        rewards = ec.simulation.rewards(list(zip(keys, choices))) 

        for (key,context,action,reward) in zip(keys,contexts,actions,rewards):
            ec.learner.learn(key,context,action,reward)

        if ec.batch_index >= 0:
            ec.result_writer.write_batch(
                ec.learner_index,
                ec.simulation_index,
                ec.batch_index,
                N      = len(rewards),
                reward = BatchMeanEstimator(rewards)
            )

    #Begin utility classes
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

    def _safe_family(self, learner: Any) -> str:
        try:
            return learner.family
        except:
            return learner.__class__.__name__
    
    def _safe_params(self, learner: Any) -> Dict[str,Any]:
        try:
            return learner.params
        except:
            return {}

    def _safe_full(self, learner: Any) -> str:
        family = self._safe_family(learner)
        params = self._safe_params(learner)

        if len(params) > 0:
            return f"{family}({','.join(f'{k}={v}' for k,v in params.items())})"
        else:
            return family

    def _lazy_simulation(self, simulation: Simulation) -> LazySimulation:
        return simulation if isinstance(simulation, LazySimulation) else LazySimulation(lambda: simulation)

    def _context_sizes(self, simulation: Simulation) -> Iterable[int]:
        for context in [i.context for i in simulation.interactions]:
            yield 0 if context is None else len(context) if isinstance(context,tuple) else 1
    
    def _action_counts(self, simulation: Simulation) -> Iterable[int]:
        for actions in [i.actions for i in simulation.interactions]:
            yield len(actions)

    def _max_batch_index(self, simulation_id, result:Result) -> int:
        return result.get_simulation(simulation_id)['batch_count']-1 if result.has_simulation(simulation_id) else -1