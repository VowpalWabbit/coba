"""The benchmarks module contains core benchmark functionality and protocols.

This module contains the abstract interface expected for Benchmark implementations. This 
module also contains several Benchmark implementations and Result data transfer class.

TODO Add docstrings to Result
TODO Write unit tests for Result
"""

import json
import collections

from abc import ABC, abstractmethod
from typing import Tuple, Hashable, Union, Sequence, List, Callable, Generic, TypeVar, Dict, Any, overload, cast
from itertools import count, repeat, groupby
from statistics import median
from ast import literal_eval
from pathlib import Path

from coba.simulations import Interaction, LazySimulation, Simulation, Context, Action
from coba.learners import Learner
from coba.execution import ExecutionContext, LoggedException
from coba.statistics import BatchMeanEstimator, StatisticalEstimate
from coba.utilities import check_pandas_support
from coba.json import CobaJsonDecoder, CobaJsonEncoder, JsonSerializable

_C = TypeVar('_C', bound=Context)
_A = TypeVar('_A', bound=Action)
_K = TypeVar("_K", bound=Hashable)

class Table(JsonSerializable, Generic[_K]):

    def __init__(self, default: Any=float('nan')):
        self._default = default
        self._columns: List[str] = []
        self._rows   : Dict[_K, Dict[str,Any]] = {}

    def add_row(self, key: _K, **kwargs) -> None:
        new_columns = [col for col in kwargs if col not in self._columns]

        if new_columns:
            self._columns.extend(new_columns)
            for data in self._rows.values():
                data.update(zip(new_columns, repeat(self._default)))

        self._rows[key] = collections.OrderedDict({key:kwargs.get(key,self._default) for key in self._columns})

    def to_tuples(self) -> Sequence[Any]:
        return list(self.to_indexed_tuples().values())

    def to_indexed_tuples(self) -> Dict[_K, Any]:
        my_type = collections.namedtuple('_T', self._columns) #type: ignore #mypy doesn't like dynamic named tuples
        return { key:my_type(**value) for key,value in self._rows.items() } #type: ignore #mypy doesn't like dynamic named tuples

    @staticmethod
    def __from_json_obj__(json_obj: Dict[str,Any]) -> 'Table':
        rows    = { literal_eval(key):value for key,value in json_obj['rows'].items() }
        columns = json_obj['columns']

        obj          = Table()
        obj._columns = columns
        obj._rows    = rows

        return obj

    def __to_json_obj__(self) -> Dict[str,Any]:

        literal_evalable = lambda key: str(key) if not isinstance(key, str) else f"'{key}'"

        return {
            'columns': self._columns,
            'rows'   : { literal_evalable(key):value for key,value in self._rows.items() }
        }

class Result(JsonSerializable):

    @staticmethod
    def from_json_file(filename:str) -> 'Result':
        needed_types = [Result, Table, StatisticalEstimate]
        return CobaJsonDecoder().decode(Path(filename).read_text(), needed_types)

    def to_json_file(self, filename:str) -> None:
        Path(filename).write_text(CobaJsonEncoder().encode(self))

    def __init__(self, default:Any = float('nan')) -> None:
        self._simulation_table  = Table[int](default)
        self._learner_table     = Table[int](default)
        self._performance_table = Table[Tuple[int,int,int]](default)

    def add_simulation_row(self, simulation_id: int, **kwargs):
        kwargs = collections.OrderedDict(kwargs)
        kwargs["simulation_id"] = simulation_id
        kwargs.move_to_end("simulation_id",last=False)
        self._simulation_table.add_row(simulation_id, **kwargs)

    def add_learner_row(self, learner_id:int, **kwargs):
        kwargs = collections.OrderedDict(kwargs)
        kwargs["learner_id"] = learner_id
        kwargs.move_to_end("learner_id",last=False)

        self._learner_table.add_row(learner_id, **kwargs)

    def add_performance_row(self, learner_id:int, simulation_id:int, batch_id:int, **kwargs):
        kwargs = collections.OrderedDict(kwargs)
        kwargs["learner_id"]    = learner_id
        kwargs["simulation_id"] = simulation_id
        kwargs["batch_id"]      = batch_id
        kwargs.move_to_end("batch_id", last=False)
        kwargs.move_to_end("simulation_id", last=False)
        kwargs.move_to_end("learner_id", last=False)

        self._performance_table.add_row((simulation_id,learner_id,batch_id), **kwargs)

    def to_tuples(self) -> Tuple[Sequence[Any], Sequence[Any], Sequence[Any]]:
        return (
            self._learner_table.to_tuples(),
            self._simulation_table.to_tuples(),
            self._performance_table.to_tuples()
        )

    def to_indexed_tuples(self) -> Tuple[Dict[int,Any], Dict[int,Any], Dict[Tuple[int,int,int],Any]]:
        return (
            self._learner_table.to_indexed_tuples(),
            self._simulation_table.to_indexed_tuples(),
            self._performance_table.to_indexed_tuples()
        )

    def to_pandas(self) -> Tuple[Any,Any,Any]:
        check_pandas_support('abc')
        import pandas as pd

        l,s,p = self.to_tuples()

        return pd.DataFrame(l), pd.DataFrame(s), pd.DataFrame(p)

    @staticmethod
    def __from_json_obj__(obj:Dict[str,Any]) -> 'Result':
        result = Result()

        result._simulation_table  = obj['simulation_table']
        result._learner_table     = obj['learner_table']
        result._performance_table = obj['performance_table']

        return result

    def __to_json_obj__(self) -> Dict[str,Any]:
        return {
            'simulation_table' : self._simulation_table,
            'learner_table'    : self._learner_table,
            'performance_table': self._performance_table
        }

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

        config = ExecutionContext.TemplatingEngine.parse(config)

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
    def __init__(self, 
        simulations: Sequence[Simulation[_C,_A]],
        *, 
        batch_count: int) -> None:
        ...

    @overload
    def __init__(self, 
        simulations: Sequence[Simulation[_C,_A]],
        *, 
        batch_size: Union[int, Sequence[int], Callable[[int],int]]) -> None:
        ...

    def __init__(self,
        simulations: Sequence[Simulation[_C,_A]], 
        batch_count: int = None, 
        batch_size : Union[int, Sequence[int], Callable[[int],int]] = None) -> None:
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

    def evaluate(self, learner_factories: Sequence[Callable[[],Learner[_C,_A]]]) -> Result:
        """Collect observations of a Learner playing the benchmark's simulations to calculate Results.

        Args:
            learner_factories: See the base class for more information.

        Returns:
            See the base class for more information.
        """

        def feature_count(i: Interaction) -> int:
            return len(i.context) if isinstance(i.context,tuple) else 0 if i.context is None else 1

        def action_count(i: Interaction) -> int:
            return len(i.actions)

        result = Result()

        for learner_index, learner in enumerate(f() for f in learner_factories):
            result.add_learner_row(learner_index, 
                name=self._safe_name(learner) #type: ignore #pylance incorrectly flags this as wrong
            )

        for simulation_index, simulation in enumerate(self._simulations):
            try:
                if isinstance(simulation, LazySimulation):
                    with ExecutionContext.Logger.log(f"loading simulation {simulation_index}..."):
                        simulation.load()

                batch_sizes   = self._batch_sizes(len(simulation.interactions))
                batch_indexes = [b for index,size in enumerate(batch_sizes) for b in repeat(index,size)]

                result.add_simulation_row(simulation_index, 
                    interaction_count = sum(batch_sizes), 
                    feature_count     = median([feature_count(i) for i in simulation.interactions]),
                    action_count      = median([action_count(i) for i in simulation.interactions])
                )

                with ExecutionContext.Logger.log(f"evaluating learners on simulation {simulation_index}..."):

                    for learner_index, learner_factory in enumerate(learner_factories):

                        learner = learner_factory()
                        name    = self._safe_name(learner) #type: ignore #pylance indicates an incorrect error here

                        with ExecutionContext.Logger.log(f"evaluating {name}..."):

                            for batch in groupby(zip(batch_indexes,simulation.interactions), lambda t: t[0]):

                                batch_index = batch[0]

                                keys     = []
                                contexts = []
                                choices  = []
                                actions  = []

                                for _, interaction in batch[1]:

                                    choice = learner.choose(interaction.key, interaction.context, interaction.actions)

                                    keys    .append(interaction.key)
                                    contexts.append(interaction.context)
                                    choices .append(choice)
                                    actions .append(interaction.actions[choice])

                                rewards = simulation.rewards(list(zip(keys, choices))) 

                                for (key,context,action,reward) in zip(keys,contexts,actions,rewards):
                                    learner.learn(key,context,action,reward)

                                result.add_performance_row(
                                    learner_index,
                                    simulation_index,
                                    batch_index,
                                    N           = len(rewards),
                                    mean_reward = BatchMeanEstimator(rewards)
                                )

            except LoggedException as e:
                pass #if we've already logged it no need to do it again

            except Exception as e:
                ExecutionContext.Logger.log(f"unhandled exception: {e}")

            if isinstance(simulation, LazySimulation):
                simulation.unload()

        return result
    
    def _safe_name(self, learner: Learner[_C,_A]) -> str:
        try:
            return learner.name
        except:
            return learner.__class__.__name__

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