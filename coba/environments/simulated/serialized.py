from itertools import islice
from typing import Dict, Any, Iterable, Union

from coba.exceptions import CobaException
from coba.pipes import Sink, Source, UrlSource, JsonDecode, JsonEncode, LambdaSource

from coba.environments.primitives import SimulatedEnvironment, SimulatedInteraction

class SerializedSimulation(SimulatedEnvironment):

    def _make_serialized_source(self, sim: Source[Iterable[SimulatedInteraction]]) -> Source[Iterable[str]]:

        def serialized_generator() -> Iterable[str]:
            json_encoder = JsonEncode()

            yield json_encoder.filter(sim.params)

            for interaction in sim.read():
                context = json_encoder.filter(interaction.context)
                actions = json_encoder.filter(interaction.actions)
                rewards = json_encoder.filter(list(interaction.rewards))
                kwargs  = json_encoder.filter(interaction.kwargs)
                yield f"[{context},{actions},{rewards},{kwargs}]"

        return LambdaSource(serialized_generator)

    def __init__(self, source: Union[str, Source[Iterable[str]], Source[Iterable[SimulatedInteraction]]]) -> None:

        if isinstance(source,str):
            self._source= UrlSource(source)
        else:
            _first = next(iter(source.read()))

            if isinstance(_first,str):
                self._source = source
            elif isinstance(_first,SimulatedInteraction):
                self._source = self._make_serialized_source(source)
            else:
                raise CobaException("We were unable to determine how to handle the source given to SerializedSimulation.")

        self._decoder = JsonDecode()

    @property
    def params(self) -> Dict[str, Any]:
        return self._decoder.filter(next(iter(self._source.read())))

    def read(self) -> Iterable[SimulatedInteraction]:
        for interaction_json in islice(self._source.read(), 1, None):
            context,actions,rewards,kwargs = tuple(self._decoder.filter(interaction_json))
            yield SimulatedInteraction(context,actions,rewards,**kwargs)

    def write(self, sink: Sink[str]):
        for line in self._source.read():
            sink.write(line)
