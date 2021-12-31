from typing import Dict, Any, Iterable, Union

from coba.pipes import DiskIO, Filter, Source, MemoryIO, Structure, CsvReader, ArffReader, LibSvmReader, ManikReader

from coba.environments.simulated.primitives import SimulatedEnvironment, SimulatedInteraction, ClassificationSimulation

class ReaderSimulation(SimulatedEnvironment):
    """Use a format reader to create a SimulatedEnvironment."""
    
    def __init__(self, 
        source   : Union[str,Source[Iterable[str]]], 
        reader   : Filter[Iterable[str], Any],         
        label_col: Union[str,int]) -> None:
        """Instantiate a ReaderSimulation.
        
        Args:
            source: Either a path to the file or an IO object able to iterate over the csv data.
            reader: A reader to parse the source.            
            label_col: The col that contains label for each line in the source. 
        """

        self._reader       = reader
        self._source       = DiskIO(source) if isinstance(source, str) else source
        self._label_column = label_col

    @property
    def params(self) -> Dict[str, Any]:
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
    """Create a SimulatedEnvironment from a supervised CSV dataset."""

    def __init__(self, source:Union[str,Source[Iterable[str]]], label_col:Union[str,int], with_header:bool=True) -> None:
        """Instantiate a CsvSimulation.

        Args:
            source: Either a path to the file or an IO object able to iterate over the csv data.
            label_col: The col that contains label for each line in the source. 
            with_header: Whether the CSV data contains a header row.
        """

        super().__init__(source, CsvReader(with_header), label_col)

    @property
    def params(self) -> Dict[str, Any]:
        return { "csv": super().params["source"] }

class ArffSimulation(ReaderSimulation):
    """Create a SimulatedEnvironment from a supervised ARFF dataset."""

    def __init__(self, source:Union[str,Source[Iterable[str]]], label_column:Union[str,int]) -> None:
        """Instantiate an ArffSimulation.
    
        Args:
            source: Either a path to the file or an IO object able to iterate over the arff data.
            label_col: The col that contains label for each line in the source. 
        """
        super().__init__(source, ArffReader(skip_encoding=[label_column]), label_column)

    @property
    def params(self) -> Dict[str, Any]:
        return { "arff": super().params["source"] }

class LibsvmSimulation(ReaderSimulation):
    """Create a SimulatedEnvironment from a libsvm dataset."""

    def __init__(self, source:Union[str,Source[Iterable[str]]]) -> None:
        """Instantiate a LibsvmSimulation.

        Args:
            source: Either a path to the file or an IO object able to iterate over the libsvm data.
        """
        super().__init__(source, LibSvmReader(), 0)

    @property
    def params(self) -> Dict[str, Any]:
        return { "libsvm": super().params["source"] }

class ManikSimulation(ReaderSimulation):
    """Create a SimulatedEnvironment from a manik dataset."""

    def __init__(self, source:Union[str,Source[Iterable[str]]]) -> None:
        """Instantiate a ManikSimulation.

        Args:
            source: Either a path to the file or an IO object able to iterate over the manik data.
        """
        super().__init__(source, ManikReader(), 0)

    @property
    def params(self) -> Dict[str, Any]:
        return { "manik": super().params["source"] }
