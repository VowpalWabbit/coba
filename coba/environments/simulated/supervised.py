import collections.abc

from itertools import chain, repeat
from typing import Any, Iterable, Union, Sequence, overload, Dict
from coba.backports import Literal
from coba.encodings import OneHotEncoder
from coba.pipes.filters import Reservoir, Structure

from coba.random import CobaRandom
from coba.pipes import Source, CsvReader, Reader, DiskIO, IdentityIO, Pipe
from coba.statistics import percentile

from coba.environments.simulated.primitives import SimulatedEnvironment, SimulatedInteraction

#         #also a filter applied later???
#         #take: int = None,

#         # these can be a filter applied later
#         # missing_val: str = "?",
#         # missing_rep: Any = float('nan'),
#         # drop_missing: bool = False

#         #define function all labels to action subset
#         #define function label,action to reward

class SupervisedSimulation(SimulatedEnvironment):

    @overload
    def __init__(self,
        source: Union[str, Source[Iterable[str]]], 
        reader: Reader = CsvReader(), 
        label_col: Union[int,str] = 0,
        label_type: Literal["C","R"] = "C",
        take: int = None) -> None:
        ...

    @overload
    def __init__(self,
        X: Sequence[Any],
        Y: Sequence[Any],
        label_type: Literal["C","R"] = "C") -> None:
        ...

    def __init__(self, *args, **kwargs) -> None:
        
        if isinstance(args[0],str) or hasattr(args[0], 'read'):
            source     = DiskIO(args[0]) if isinstance(args[0],str) else args[0]
            reader     = args[1] if len(args) > 1 else kwargs.get('reader', CsvReader())
            label_col  = args[2] if len(args) > 2 else kwargs.get("label_col", 0)
            label_type = args[3] if len(args) > 3 else kwargs.get("label_type", "C")
            take       = args[4] if len(args) > 4 else kwargs.get("take", None)

            self._source     = Pipe.join(source, [reader, Reservoir(take), Structure((None,label_col))])
            self._label_type = label_type

        else:
            X          = args[0]
            Y          = args[1]
            label_type = args[2] if len(args) > 2 else kwargs.get("label_type", "C")

            self._source = IdentityIO(list(zip(X,Y)))
            self._label_type = label_type

    @property
    def params(self) -> Dict[str,Any]:
        return {}

    def read(self) -> Iterable[SimulatedInteraction]:
        features,labels = zip(*self._source.read())

        if self._label_type == "R":
            max_n_actions = 10

            #Scale the labels so their range is 1.
            min_l, max_l = min(labels), max(labels)
            labels = [float(l)/(max_l-min_l) for l in labels]

            if len(labels) <= max_n_actions:
                actions = labels
            else:
                actions = percentile(labels, [i/(max_n_actions+1) for i in range(1,max_n_actions+1)])

            values  = dict(zip(OneHotEncoder().fit_encodes(actions), actions))
            actions = list(values.keys())

            reward = lambda action,label: 1-abs(values[action]-float(label))
        else:
            #how can we tell the difference between featurized labels and multilabels????
            #for now we will assume multilables will be passed in as arrays not tuples...
            if not isinstance(labels[0], collections.abc.Hashable):
                actions = list(chain.from_iterable(labels))
            else:
                actions = list(labels)

            is_label      = lambda action,label: action == label
            in_multilabel = lambda action,label: isinstance(label,collections.abc.Sequence) and action in label
            reward        = lambda action,label: int(is_label(action,label) or in_multilabel(action,label))

        contexts = features
        actions  = CobaRandom(1).shuffle(sorted(set(actions)))
        rewards  = [ [ reward(action,label) for action in actions ] for label in labels ]

        for c,a,r in zip(contexts, repeat(actions), rewards):
            yield SimulatedInteraction(c,a,rewards=r)

#         source  = self._get_source(source)
#         reader  = self._get_reader(format)
#         actions = self._get_actions()

#         make reader passed in

    # def _get_source(self, source: Union[str, Source[Iterable[str]], Sequence[str]]) -> Source[Iterable[str]]:
        
    #     if isinstance(source,str):
    #         return DiskIO(source)
        
    #     if hasattr(source, "read"):
    #         return source

    #     return MemoryIO(list(source))

    # def _get_reader(self, format: Literal["csv","arff","libsvm","manik"]) -> Filter[Iterable[str], Iterable[Any]]:

    #     if format == "csv":
    #         return CsvReader()
        
    #     if format == "arff":
    #         return ArffReader()
        
    #     if format == "libsvm":
    #         return LibSvmReader()

    #     if format == "manik":
    #         return ManikReader()

    # def _get_actions(self, source, reader, format, label_col):
    #     pass