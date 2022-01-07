from typing import Dict, Any, Iterable, Union
from coba.backports import Literal

from coba.pipes import Source, CsvReader, ArffReader, LibSvmReader, ManikReader

from coba.environments.primitives import ReaderEnvironment
from coba.environments.simulated.primitives import SupervisedToSimulation

class CsvSimulation(ReaderEnvironment):
    """Create a SimulatedEnvironment from a supervised CSV dataset."""

    def __init__(self, 
        source: Union[str,Source[Iterable[str]]],
        label: Union[str,int], 
        has_header: bool=True,
        label_type: Literal["C","R"] = "C",
        **dialect) -> None:
        """Instantiate a CsvSimulation.

        Args:
            source: Either file path or a Source object iterating over csv data.
            label: The column that contains the label for each line in the source. 
            has_header: Whether the CSV data contains a header row.
            label_type: Whether the label is a continuous real value or a class. 
            dialect: Reader configurations identical to the stdlib csv.reader(dialect).
        """

        super().__init__(source, CsvReader(**dialect), SupervisedToSimulation(label,has_header,label_type))

    @property
    def params(self) -> Dict[str, Any]:
        return { "csv": super().params["source"] }

class ArffSimulation(ReaderEnvironment):
    """Create a SimulatedEnvironment from a supervised ARFF dataset."""

    def __init__(self, source:Union[str,Source[Iterable[str]]], label:str, label_type: Literal["C","R"] = "C") -> None:
        """Instantiate an ArffSimulation.
    
        Args:
            source: Either a file path or a Source object iterating over arff data.
            label: The column that contains the label for each line in the source.
            label_type: Whether the label is a continuous real value or a class. 
        """
        super().__init__(source, ArffReader(skip_encoding=[label]), SupervisedToSimulation(label,True,label_type))

    @property
    def params(self) -> Dict[str, Any]:
        return { "arff": super().params["source"] }

class LibsvmSimulation(ReaderEnvironment):
    """Create a SimulatedEnvironment from a libsvm dataset."""

    def __init__(self, source:Union[str,Source[Iterable[str]]]) -> None:
        """Instantiate a LibsvmSimulation.

        Args:
            source: Either a file path or a Source object iterating over libsvm data.
        """
        super().__init__(source, LibSvmReader(), SupervisedToSimulation(0,False))

    @property
    def params(self) -> Dict[str, Any]:
        return { "libsvm": super().params["source"] }

class ManikSimulation(ReaderEnvironment):
    """Create a SimulatedEnvironment from a manik dataset."""

    def __init__(self, source:Union[str,Source[Iterable[str]]]) -> None:
        """Instantiate a ManikSimulation.

        Args:
            source: Either a path to the file or an IO object able to iterate over the manik data.
        """
        super().__init__(source, ManikReader(), SupervisedToSimulation(0,False))

    @property
    def params(self) -> Dict[str, Any]:
        return { "manik": super().params["source"] }
