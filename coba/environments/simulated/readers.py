from typing import Dict, Any, Iterable, Union

from coba.pipes import DiskIO, Filter, Source, MemoryIO, Structure, CsvReader, ArffReader, LibSvmReader, ManikReader

from coba.environments.simulated.primitives import SimulatedEnvironment, SimulatedInteraction, ClassificationSimulation


class ReaderSimulation(SimulatedEnvironment):

    def __init__(self, 
        reader   : Filter[Iterable[str], Any], 
        source   : Union[str,Source[Iterable[str]]], 
        label_col: Union[str,int]) -> None:
        
        self._reader       = reader
        self._source       = DiskIO(source) if isinstance(source, str) else source
        self._label_column = label_col

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the simulation."""
        if isinstance(self._source,DiskIO):
            return {"source": str(self._source._filename) }
        elif isinstance(self._source,MemoryIO):
            return {"source": 'memory' }
        else:
            return {"source": self._source.__class__.__name__}

    def read(self) -> Iterable[SimulatedInteraction]:
        """Read the interactions in this simulation."""
        parsed_rows_iter = iter(self._reader.filter(self._source.read()))
        structured_rows = Structure([None, self._label_column]).filter(parsed_rows_iter)

        return ClassificationSimulation(structured_rows).read()

class CsvSimulation(ReaderSimulation):
    def __init__(self, source:Union[str,Source[Iterable[str]]], label_column:Union[str,int], with_header:bool=True) -> None:
        super().__init__(CsvReader(with_header), source, label_column)

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the simulation."""
        return { "csv": super().params["source"] }

class ArffSimulation(ReaderSimulation):
    def __init__(self, source:Union[str,Source[Iterable[str]]], label_column:Union[str,int]) -> None:
        super().__init__(ArffReader(skip_encoding=[label_column]), source, label_column)

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the simulation."""
        return { "arff": super().params["source"] }

class LibsvmSimulation(ReaderSimulation):
    def __init__(self, source:Union[str,Source[Iterable[str]]]) -> None:
        super().__init__(LibSvmReader(), source, 0)

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the simulation."""
        return { "libsvm": super().params["source"] }

class ManikSimulation(ReaderSimulation):
    def __init__(self, source:Union[str,Source[Iterable[str]]]) -> None:
        super().__init__(ManikReader(), source, 0)

    @property
    def params(self) -> Dict[str, Any]:
        """Paramaters describing the simulation."""
        return { "manik": super().params["source"] }

