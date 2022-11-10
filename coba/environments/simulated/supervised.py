import collections.abc

from itertools import chain, repeat
from typing import Any, Iterable, Union, Sequence, overload, Dict, MutableSequence, MutableMapping
from coba.backports import Literal

from coba.pipes import Pipes, Source, IterableSource, Structure, Reservoir, UrlSource, CsvReader
from coba.pipes import CsvReader, ArffReader, LibsvmReader, ManikReader

from coba.environments.primitives import SimulatedEnvironment, SimulatedInteraction
from coba.environments.primitives import L1Reward, DiscreteReward, HammingReward

class CsvSource(Source[Iterable[MutableSequence]]):
    """Load a source (either local or remote) in CSV format.

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

    def read(self) -> Iterable[MutableSequence]:
        """Read and parse the csv source."""
        return self._source.read()

    @property
    def params(self) -> Dict[str, Any]:
        """Parameters describing the csv source."""
        return self._source.params

    def __str__(self) -> str:
        return str(self._source)

class ArffSource(Source[Union[Iterable[MutableSequence], Iterable[MutableMapping]]]):
    """Load a source (either local or remote) in ARFF format.

    This is primarily used by SupervisedSimulation to create Environments for Experiments.
    """

    def __init__(self,
        source: Union[str,Source[Iterable[str]]],
        cat_as_str: bool = False) -> None:
        """Instantiate an ArffSource.

        Args:
            source: The data source. Accepts either a string representing the source location or another Source.
            cat_as_str: Indicates that categorical features should be encoded as a string rather than one hot encoded.
        """
        source = UrlSource(source) if isinstance(source,str) else source
        reader = ArffReader(cat_as_str)
        self._source = Pipes.join(source, reader)

    def read(self) -> Union[Iterable[MutableSequence], Iterable[MutableMapping]]:
        """Read and parse the arff source."""
        return self._source.read()

    @property
    def params(self) -> Dict[str, Any]:
        """Parameters describing the arff source."""
        return self._source.params

    def __str__(self) -> str:
        return str(self._source)

class LibSvmSource(Source[Iterable[MutableMapping]]):
    """Load a source (either local or remote) in libsvm format.

    This is primarily used by SupervisedSimulation to create Environments for Experiments.
    """

    def __init__(self, source: Union[str,Source[Iterable[str]]]) -> None:
        """Instantiate a LibsvmSource.

        Args:
            source: The data source. Accepts either a string representing the source location or another Source.
        """
        source = UrlSource(source) if isinstance(source,str) else source
        reader = LibsvmReader()
        self._source = Pipes.join(source, reader)

    def read(self) -> Iterable[MutableMapping]:
        """Read and parse the libsvm source."""
        return self._source.read()

    @property
    def params(self) -> Dict[str, Any]:
        """Parameters describing the libsvm source."""
        return self._source.params

    def __str__(self) -> str:
        return str(self._source)

class ManikSource(Source[Iterable[MutableMapping]]):
    """Load a source (either local or remote) in Manik format.

    This is primarily used by SupervisedSimulation to create Environments for Experiments.
    """

    def __init__(self, source: Union[str,Source[Iterable[str]]]) -> None:
        """Instantiate a ManikSource.

        Args:
            source: The data source. Accepts either a string representing the source location or another Source.
        """
        source = UrlSource(source) if isinstance(source,str) else source
        reader = ManikReader()
        self._source = Pipes.join(source, reader)

    def read(self) -> Iterable[MutableMapping]:
        """Read and parse the manik source."""
        return self._source.read()

    @property
    def params(self) -> Dict[str, Any]:
        """Parameters describing the manik source."""
        return self._source.params

    def __str__(self) -> str:
        return str(self._source)

class SupervisedSimulation(SimulatedEnvironment):
    """Create a contextual bandit simulation using an existing supervised regression or classification dataset."""

    @overload
    def __init__(self,
        source: Source = None,
        label_col: Union[int,str] = None,
        label_type: Literal["C","R"] = None,
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
        label_type: Literal["C","R"] = None) -> None:
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
            if label_col is not None: source = Pipes.join(source, Structure((None,label_col)))
            params = source.params

        else:
            X          = args[0]
            Y          = args[1]
            label_type = args[2] if len(args) > 2 else kwargs.get("label_type", None)
            source     = IterableSource(list(zip(X,Y)))
            params     = {"source": "[X,Y]"}

        self._label_type = label_type
        self._source     = source
        self._params     = {**params, "label_type": self._label_type, "type": "SupervisedSimulation" }

    @property
    def params(self) -> Dict[str,Any]:
        return self._params

    def read(self) -> Iterable[SimulatedInteraction]:

        items = list(self._source.read())

        if not items: return []

        features,labels = zip(*items)

        self._label_type = self._label_type or ("R" if isinstance(labels[0], (int,float)) else "C")
        self._params['label_type'] = self._label_type

        if self._label_type == "R":
            actions = []
            reward  = L1Reward
        else:
            #how can we tell the difference between featurized labels and multilabels????
            #for now we will assume multilables will be passed in as arrays as opposed to tuples...
            actions = labels if not isinstance(labels[0], list) else list(chain(*labels))
            actions = sorted(set(actions),reverse=isinstance(actions[0],tuple))
            
            if isinstance(labels[0],list):
                reward = HammingReward
            else:
                reward = lambda label: DiscreteReward(actions, [ int(a == label) for a in actions])

        contexts = features
        rewards  = map(reward,labels)

        for c,a,r in zip(contexts, repeat(actions), rewards):
            yield SimulatedInteraction(c,a,r)
