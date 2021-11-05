from typing import Any, Iterable, Optional, Sequence

from coba.learners import Learner
from coba.learners.core import SafeLearner
from coba.pipes import Pipe, Filter, Sink, Cartesian, JsonEncode, DiskIO, MemoryIO

from coba.benchmarks.results import Result, ResultPromote

class Transaction:

    @staticmethod
    def version(version = None) -> Any:
        return ['version', version or ResultPromote.CurrentVersion]

    @staticmethod
    def benchmark(n_learners, n_simulations) -> Any:
        data = {
            "n_learners"   : n_learners,
            "n_simulations": n_simulations,
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
    def learners(learners: Sequence[Learner]) -> Iterable[Any]:
        for index, learner in enumerate(map(SafeLearner,learners)):
            yield Transaction.learner(index, full_name=learner.full_name, **learner.params)

    @staticmethod
    def simulation(simulation_id: int, **kwargs) -> Any:
        """Write simulation metadata row to Result.
        
        Args:
            simulation_index: The index of the simulation in the benchmark's simulations.
            kwargs: The metadata to store about the learner.
        """
        return ["S", simulation_id, kwargs]

    @staticmethod
    def interactions(simulation_id:int, learner_id:int, **kwargs) -> Any:
        """Write interaction evaluation metadata row to Result.

        Args:
            learner_id: The primary key for the learner we observed on the interaction.
            simulation_id: The primary key for the simulation the interaction came from.
            kwargs: The metadata to store about the interaction with the learner.
        """

        return ["I", (simulation_id, learner_id), kwargs]

class TransactionIsNew(Filter):

    def __init__(self, existing: Result):

        self._existing = existing

    def filter(self, transactions: Iterable[Any]) -> Iterable[Any]:
        
        for transaction in transactions:
            
            tipe  = transaction[0]

            if tipe == "version" and self._existing.version is not None:
                continue
            
            if tipe == "benchmark" and len(self._existing.benchmark) != 0:
                continue

            if tipe == "I" and transaction[1] in self._existing._interactions:
                continue

            if tipe == "S" and transaction[1] in self._existing._simulations:
                continue

            if tipe == "L" and transaction[1] in self._existing._learners:
                continue

            yield transaction

class TransactionSink(Sink):

    def __init__(self, transaction_log: Optional[str], restored: Result) -> None:

        json_encode = Cartesian(JsonEncode())

        final_sink = Pipe.join([json_encode], DiskIO(transaction_log)) if transaction_log else MemoryIO()
        self._sink = Pipe.join([TransactionIsNew(restored)], final_sink)

    def write(self, items: Sequence[Any]) -> None:
        self._sink.write(items)

    @property
    def result(self) -> Result:
        if isinstance(self._sink, Pipe.FiltersSink):
            final_sink = self._sink.final_sink()
        else:
            final_sink = self._sink

        if isinstance(final_sink, MemoryIO):
            return Result.from_transactions(final_sink.items)

        if isinstance(final_sink, DiskIO):
            return Result.from_file(final_sink.filename)

        raise Exception("Transactions were written to an unrecognized sink.")
