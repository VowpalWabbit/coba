from typing import Dict, Any, Iterable

from coba.exceptions import CobaException
from coba.random import CobaRandom
from coba.environments.simulated.primitives import SimulatedInteraction
from coba.environments.filters import EnvironmentFilter

class ToInteractionGrounded(EnvironmentFilter):
    def __init__(self, n_users: int, n_normal:int, n_words:int, n_good:int, seed:int) -> None:
        self._n_users   = n_users
        self._n_normal  = n_normal
        self._n_words   = n_words
        self._n_good    = n_good
        self._seed      = seed

        if n_normal > n_users:
            raise CobaException("Igl conversion can't have more normal users (n_normal) than total users (n_users).")

        if n_good > n_words:
            raise CobaException("Igl conversion can't have more good words (n_good) than total words (n_words).")

        self.userids    = list(range(self._n_users))
        self.normalids  = self.userids[:self._n_normal]
        self.wordids    = list(range(self._n_words))
        self.good_words = self.wordids[:self._n_good]
        self.bad_words  = self.wordids[self._n_good:]

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "n_users" : self._n_users,
            "n_normal": self._n_normal,
            "n_good"  : self._n_good,
            "n_words" : self._n_words,
            "igl_seed": self._seed
        }

    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[SimulatedInteraction]:
        rng = CobaRandom(self._seed)

        normalids = set(self.normalids) #we make it a set for faster contains checks

        for interaction in interactions:

            if len(set(interaction.rewards)-{0,1}) == 0:
                igl_rewards = interaction.rewards
            else:
                max_index              = interaction.rewards.index(max(interaction.rewards))
                igl_rewards            = [0]*len(interaction.rewards)
                igl_rewards[max_index] = 1

            userid    = rng.choice(self.userids)
            good, bad = (self.good_words, self.bad_words) if userid in normalids else (self.bad_words, self.good_words)
            words     = [ (rng.choice(good),) if r==1 else (rng.choice(bad),) for r in igl_rewards ]
            kwargs    = { "userid":userid, "feedbacks":words, "isnormal": userid in self.normalids }

            try:
                igl_context = dict(userid=userid,**interaction.context)
            except:
                try:
                    igl_context = tuple([userid]+list(interaction.context))
                except:
                    igl_context = (userid, interaction.context)

            yield SimulatedInteraction(igl_context, interaction.actions, igl_rewards, **kwargs, **interaction.kwargs)
