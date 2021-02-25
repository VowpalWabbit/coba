import math
import collections

from copy import deepcopy
from statistics import mean
from itertools import product, groupby, chain, count, repeat
from statistics import median
from pathlib import Path
from typing import (
    Iterable, Tuple, Sequence, Dict, Any, cast, Optional,
    overload, List, Mapping, MutableMapping, Union
)
from coba.random import CobaRandom
from coba.learners import Learner
from coba.simulations import Context, Action, Key, Interaction, Simulation, BatchedSimulation, OpenmlSimulation, Take, Shuffle, Batch, PCA, Sort
from coba.statistics import OnlineMean, OnlineVariance
from coba.tools import PackageChecker, CobaRegistry, CobaConfig

from coba.data.structures import Table
from coba.data.filters import Filter, IdentityFilter, JsonEncode, JsonDecode, Cartesian, ResponseToText, StringJoin
from coba.data.sources import HttpSource, Source, MemorySource, DiskSource
from coba.data.sinks import Sink, MemorySink, DiskSink
from coba.data.pipes import Pipe, StopPipe

class BenchmarkLearner:

    @property
    def family(self) -> str:
        try:
            return self._learner.family
        except AttributeError:
            return self._learner.__class__.__name__

    @property
    def params(self) -> Dict[str, Any]:
        try:
            return self._learner.params
        except AttributeError:
            return {}

    @property
    def full_name(self) -> str:
        if len(self.params) > 0:
            return f"{self.family}({','.join(f'{k}={v}' for k,v in self.params.items())})"
        else:
            return self.family

    def __init__(self, learner: Learner, seed: Optional[int]) -> None:
        self._learner = learner
        self._random  = CobaRandom(seed)

    def init(self) -> None:
        try:
            self._learner.init()
        except AttributeError:
            pass

    def choose(self, key: Key, context: Context, actions: Sequence[Action]) -> Tuple[Action, float]:
        p = self._learner.predict(key, context, actions)
        c = list(zip(actions,p))
        
        return self._random.choice(c, p)
    
    def learn(self, key: Key, context: Context, action: Action, reward: float, probability: float) -> None:
        self._learner.learn(key, context, action, reward, probability)

class BenchmarkSimulation(Source[Simulation]):

    def __init__(self, source: Source[Simulation], filters: Sequence[Filter[Simulation,Simulation]] = None) -> None:
        self._pipe = source if filters is None else Pipe.join(source, filters)

    @property
    def source(self) -> Source[Simulation]:
        return self._pipe._source if isinstance(self._pipe, (Pipe.SourceFilters)) else self._pipe

    @property
    def filter(self) -> Filter[Simulation,Simulation]:
        return self._pipe._filter if isinstance(self._pipe, Pipe.SourceFilters) else IdentityFilter()

    def read(self) -> Simulation:
        return self._pipe.read()

    def __repr__(self) -> str:
        return self._pipe.__repr__()
