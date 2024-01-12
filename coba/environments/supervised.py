from itertools import chain
from typing import Any, Iterable, Union, Sequence, overload, Dict, Literal, Tuple

from coba.pipes import Pipes, IterableSource, LabelRows, Reservoir, UrlSource, CsvReader
from coba.pipes import CsvReader, ArffReader, LibsvmReader, ManikReader

from coba.utilities    import peek_first
from coba.primitives   import Categorical, Source, Environment, Dense, Sparse
from coba.interactions import SimulatedInteraction
from coba.rewards      import L1Reward, BinaryReward, HammingReward

class CsvSource(Source[Iterable[Dense]]):
    """Load a csv dataset.

    This is primarily used by SupervisedSimulation to create Environments for Experiments.
    """

    def __init__(self, source: Union[str,Source[Iterable[str]]], has_header:bool=False, **dialect) -> None:
        """Instantiate a CsvSource.

        Args:
            source: The data source. Accepts either a string representing the source location or another Source.
            has_header: Indicates if the CSV files has a header row.
        """
        source = UrlSource(source) if isinstance(source,str) else source
        reader = CsvReader(has_header, **dialect)
        self._source = Pipes.join(source, reader)

    def read(self) -> Iterable[Dense]:
        """Read and parse the csv source."""
        return self._source.read()

    @property
    def params(self) -> Dict[str, Any]:
        """Parameters describing the csv source."""
        return self._source.params

    def __str__(self) -> str:
        return str(self._source)

class ArffSource(Source[Union[Iterable[Dense], Iterable[Sparse]]]):
    """Load an arff dataset.

    Remarks:
        https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/
    """

    def __init__(self,source: Union[str,Source[Iterable[str]]]) -> None:
        """Instantiate an ArffSource.

        Args:
            source: The data source. Accepts either a string representing the source location or another Source.
        """
        source       = UrlSource(source) if isinstance(source,str) else source
        self._source = Pipes.join(source, ArffReader())

    def read(self) -> Union[Iterable[Dense], Iterable[Sparse]]:
        """Read and parse the arff source."""
        return self._source.read()

    @property
    def params(self) -> Dict[str, Any]:
        """Parameters describing the arff source."""
        return self._source.params

    def __str__(self) -> str:
        return str(self._source)

class LibSvmSource(Source[Iterable[Sparse]]):
    """Load a libsvm dataset.

    Remarks:
        https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html
    """

    def __init__(self, source: Union[str,Source[Iterable[str]]]) -> None:
        """Instantiate a LibsvmSource.

        Args:
            source: The data source. Accepts either a string representing the source location or another Source.
        """
        source       = UrlSource(source) if isinstance(source,str) else source
        self._source = Pipes.join(source, LibsvmReader())

    def read(self) -> Iterable[Sparse]:
        """Read and parse the libsvm source."""
        return self._source.read()

    @property
    def params(self) -> Dict[str, Any]:
        """Parameters describing the libsvm source."""
        return self._source.params

    def __str__(self) -> str:
        return str(self._source)

class ManikSource(Source[Iterable[Sparse]]):
    """Load a manik dataset.

    Remarks:
        http://manikvarma.org/downloads/XC/XMLRepository.html

    """

    def __init__(self, source: Union[str,Source[Iterable[str]]]) -> None:
        """Instantiate a ManikSource.

        Args:
            source: The data source. Accepts either a string representing the source location or another Source.
        """
        source       = UrlSource(source) if isinstance(source,str) else source
        self._source = Pipes.join(source, ManikReader())

    def read(self) -> Iterable[Sparse]:
        """Read and parse the manik source."""
        return self._source.read()

    @property
    def params(self) -> Dict[str, Any]:
        """Parameters describing the manik source."""
        return self._source.params

    def __str__(self) -> str:
        return str(self._source)

class SupervisedSimulation(Environment):
    """A contextual bandit environment created from supervised data."""

    @overload
    def __init__(self,
        source: Source[Union[Iterable[Dense], Iterable[Sparse], Iterable[Tuple[Any,Any]]]] = None,
        label_col: Union[int,str] = None,
        label_type: Literal["c","r","m"] = None,
        take: int = None) -> None:
        """Instantiate a SupervisedSimulation.

        Args:
            source: A source object that reads the supervised data.
            label_col: The header name or index which identifies the label feature in each example. If
                label_col is None the source must return an iterable of tuple pairs where the first item
                are the features and the second item is the label.
            label_type: Indicates whether the label column is a classification or regression value. If an explicit
                label_type is not provided then the label_type will be inferred based on the data source.
            take: The number of random examples you'd like to draw from the given data set for the environment.
        """
        ...

    @overload
    def __init__(self,
        X: Sequence[Any],
        Y: Sequence[Any],
        label_type: Literal["c","r","m"]) -> None:
        """Instantiate a SupervisedSimulation.

        Args:
            X: A sequence of example features that will be used to create interaction contexts in the simulation.
            Y: A sequence of supervised labels that will be used to construct actions and rewards in the simulation.
            label_type: Indicates whether the label column is a classification or regression value.
        """
        ...

    def __init__(self, *args, **kwargs) -> None:
        """Instantiate a SupervisedSimulation."""

        if 'source' in kwargs or (args and hasattr(args[0], 'read')):
            source     = args[0] if len(args) > 0 else kwargs['source']
            label_col  = args[1] if len(args) > 1 else kwargs.get("label_col", None)
            label_type = args[2] if len(args) > 2 else kwargs.get("label_type", None)
            take       = args[3] if len(args) > 3 else kwargs.get("take", None)
            if take      is not None: source = Pipes.join(source, Reservoir(take))
            if label_col is not None: source = Pipes.join(source, LabelRows(label_col,label_type))
            params = source.params

        else:
            X          = args[0]
            Y          = args[1]
            label_type = args[2] if len(args) > 2 else kwargs.get("label_type", None)
            params     = {"source": "[X,Y]"}
            source     = IterableSource(zip(X,Y))

        self._label_type = label_type
        self._source     = source
        self._params     = {**params, "label_type": self._label_type, "env_type": "SupervisedSimulation" }

    @property
    def params(self) -> Dict[str,Any]:
        return self._params

    def read(self) -> Iterable[SimulatedInteraction]:

        first, rows = peek_first(self._source.read())
        first_row_type = 0 if hasattr(first,'label') else 1

        if not rows: return []

        first_label      = first.label if first_row_type == 0 else first[1]
        first_label_type = first.tipe  if first_row_type == 0 else None

        if first_label_type is None:
            label_type = self._label_type or ("r" if isinstance(first_label, (int,float)) else "c")
        else:
            label_type = self._label_type or first_label_type

        label_type = label_type.lower()

        if label_type == "r":
            actions = []
            reward  = L1Reward
            self._params['n_actions'] = float('inf')

        elif label_type == "c" and isinstance(first_label, Categorical):
            #Handling the categoricals separately allows for a performance optimization
            #since we can use the Categorical's as_int property rather than action_indexes
            actions = [ Categorical(l,first_label.levels) for l in first_label.levels ]
            reward  = BinaryReward
            self._params['n_actions'] = len(actions)

        else:
            #we need to know all labels in the dataset to determine actions
            rows = list(rows)
            lbls = [r.label for r in rows] if first_row_type == 0 else [r[1] for r in rows]
            if label_type == "m":
                actions = sorted(set(list(chain(*lbls))))
                reward = HammingReward
            else:
                delist  = lambda l: l[0] if isinstance(l,list) else l
                actions = sorted(set(map(delist,lbls)))
                reward  = lambda l: BinaryReward(delist(l))
            self._params['n_actions'] = len(actions)

        if first_row_type == 0:
            for row in rows:
                yield {'context':row.feats,'actions':actions,'rewards':reward(row.label)}
        else:
            for row in rows:
                yield {'context':row[0],'actions':actions,'rewards':reward(row[1])}
