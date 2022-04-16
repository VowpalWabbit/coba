from itertools import islice
from typing import Dict, Any, Iterable, Union

from coba.pipes import Sink, Source, UrlSource, JsonDecode, JsonEncode, LambdaSource
from coba.environments.simulated.primitives import SimulatedEnvironment, SimulatedInteraction

class SerializedSimulation(SimulatedEnvironment):

    def _make_serialized_source(self, sim: SimulatedEnvironment) -> Source[Iterable[str]]:

        def serialized_generator() -> Iterable[str]:
            json_encoder = JsonEncode()

            yield json_encoder.filter(sim.params)

            for interaction in sim.read():
                context = json_encoder.filter(interaction.context)
                actions = json_encoder.filter(interaction.actions)
                rewards = json_encoder.filter(interaction.rewards) 
                kwargs  = json_encoder.filter(interaction.kwargs)
                yield f"[{context},{actions},{rewards},{kwargs}]"

        return LambdaSource(serialized_generator)

    def __init__(self, source: Union[str, Source[Iterable[str]], SimulatedEnvironment]) -> None:

        if isinstance(source,str):
            self._source= UrlSource(source)
        elif isinstance(source, SimulatedEnvironment):
            self._source = self._make_serialized_source(source)
        else:
            self._source = source

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
