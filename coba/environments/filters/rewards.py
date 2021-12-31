from itertools import islice
from typing import Iterable, Dict, Any

from coba.environments.simulated.primitives import SimulatedInteraction
from coba.environments.filters.primitives import EnvironmentFilter

class Cycle(EnvironmentFilter):
    """Cycle all rewards associated with actions by one place.
    
    This filter is useful for testing an algorithms response to a non-stationary shock.
    """

    def __init__(self, after:int = 0):
        """Instantiate a Cycle filter.
        
        Args:
            after: How many interactions should be seen before applying the cycle filter.
        """
        self._after = after

    @property
    def params(self) -> Dict[str, Any]:
        return { "cycle_after": self._after }

    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[SimulatedInteraction]:

        underlying_iterable     = iter(interactions)
        sans_cycle_interactions = islice(underlying_iterable, self._after)
        with_cycle_interactions = underlying_iterable

        for interaction in sans_cycle_interactions:
            yield interaction

        for interaction in with_cycle_interactions:
            kwargs = {k:v[1:]+v[:1] for k,v in interaction.kwargs.items()}
            yield SimulatedInteraction(interaction.context, interaction.actions, **kwargs)

class Binary(EnvironmentFilter):
    """Binarize all rewards to either 1 (max rewards) or 0 (all others)."""

    @property
    def params(self) -> Dict[str, Any]:
        return { "binary": True }

    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[SimulatedInteraction]:

        for interaction in interactions:
            kwargs  = interaction.kwargs.copy()
            max_rwd = max(kwargs["rewards"])
            
            kwargs["rewards"] = [int(r==max_rwd) for r in kwargs["rewards"]]

            yield SimulatedInteraction(interaction.context, interaction.actions, **kwargs)
