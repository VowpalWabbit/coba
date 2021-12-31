import collections.abc

from itertools import islice, chain
from typing import Optional, Sequence, Union, Iterable, Dict, Any

from coba import pipes
from coba.random import CobaRandom

from coba.environments.primitives import Interaction
from coba.environments.logged.primitives import LoggedInteraction
from coba.environments.simulated.primitives import SimulatedInteraction
from coba.environments.filters.primitives import EnvironmentFilter

class Take(pipes.Take):
    
    @property
    def params(self) -> Dict[str, Any]:
        return { "take": self._count }

    def __str__(self) -> str:
        return str(self.params)

class Reservoir(pipes.Reservoir):

    def __init__(self, count: Optional[int], seed:int=1)-> None:
        super().__init__(count, seed)

    @property
    def params(self) -> Dict[str, Any]:
        return { "reservoir_count": self._count, "reservoir_seed": self._seed }

    def __str__(self) -> str:
        return str(self.params)

class Shuffle(pipes.Shuffle):
    
    @property
    def params(self) -> Dict[str, Any]:
        return { "shuffle": self._seed }

    def __str__(self) -> str:
        return str(self.params)

class Sort(EnvironmentFilter):

    def __init__(self, *keys: Union[str,int,Sequence[Union[str,int]]]) -> None:
        
        self._keys = []
        
        for key in keys:
            if not isinstance(key, collections.abc.Sequence) or isinstance(key,str):
                self._keys.append(key)
            else:
                self._keys.extend(key)

    @property
    def params(self) -> Dict[str, Any]:
        return { "sort": self._keys }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        return sorted(interactions, key=lambda interaction: tuple(interaction.context[key] for key in self._keys))

class ToWarmStart(EnvironmentFilter):

    def __init__(self, n_warmstart:int, seed:int = 1):
        self._n_warmstart = n_warmstart
        self._seed = seed

    @property
    def params(self) -> Dict[str, Any]:
        return { "n_warmstart": self._n_warmstart }

    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[Interaction]:

        self._rng = CobaRandom(self._seed)

        underlying_iterable    = iter(interactions)
        logged_interactions    = map(self._to_logged_interaction, islice(underlying_iterable, self._n_warmstart))
        simulated_interactions = underlying_iterable

        return chain(logged_interactions, simulated_interactions)

    def _to_logged_interaction(self, interaction: SimulatedInteraction) -> LoggedInteraction:
        num_actions   = len(interaction.actions)
        probabilities = [1/num_actions] * num_actions 
        
        selected_index       = self._rng.choice(list(range(num_actions)), probabilities)
        selected_action      = interaction.actions[selected_index]
        selected_probability = probabilities[selected_index]
        
        kwargs = {"probability":selected_probability, "actions":interaction.actions}

        if "reveals" in interaction.kwargs:
            kwargs["reveal"] = interaction.kwargs["reveals"][selected_index]

        if "rewards" in interaction.kwargs:
            kwargs["reward"] = interaction.kwargs["rewards"][selected_index]

        return LoggedInteraction(interaction.context, selected_action, **kwargs)
