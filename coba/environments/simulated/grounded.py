from collections import abc
from itertools import chain
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

        self.userids   = list(range(self._n_users))
        self.normalids = self.userids[:self._n_normal]
        self.wordids   = list(range(self._n_words))
        self.goodwords = self.wordids[:self._n_good]
        self.badwords  = self.wordids[self._n_good:]

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "userids"  : self.userids,
            "wordids"  : self.wordids,
            "normalids": self.normalids,
            "goodwords": self.goodwords,
            "n_users"  : self._n_users,
            "n_normal" : self._n_normal,
            "n_good"   : self._n_good,
            "n_words"  : self._n_words,
            "igl_seed" : self._seed
        }

    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[SimulatedInteraction]:
        rng = CobaRandom(self._seed)
        
        #we make it a set for faster contains checks
        normalids       = set(self.normalids) 
        isnormal        = [u in normalids for u in self.userids]
        userid_isnormal = list(zip(self.userids,isnormal))

        interactions = iter(interactions)
        first        = next(interactions)

        not_10_rewards = bool((set(first.rewards)-{0,1}))

        goodwords = self.goodwords
        badwords  = self.badwords

        if isinstance(first.context,abc.Mapping):
            context_type = 0
        elif isinstance(first.context,abc.Sequence) and not isinstance(first.context,str):
            context_type = 1
        else:
            context_type = 2

        def batched_rand_int_iter(sequence):
            b=len(sequence)-1
            while True:
                for r in rng.randints(1000,0,b):
                    yield sequence[r]

        goods = batched_rand_int_iter([ (g,) for g in goodwords])
        bads  = batched_rand_int_iter([ (b,) for b in badwords])
        users = batched_rand_int_iter(userid_isnormal)

        for interaction in chain([first], interactions):

            igl_rewards = interaction.rewards

            if not_10_rewards:                
                max_index              = igl_rewards.index(max(igl_rewards))
                igl_rewards            = [0]*len(igl_rewards)
                igl_rewards[max_index] = 1

            userid,isnormal = next(users)

            if isnormal:
                words = tuple( next(goods) if r==1 else next(bads) for r in igl_rewards)
            else:
                words = tuple( next(goods) if r==0 else next(bads) for r in igl_rewards)

            if context_type == 0:
                igl_context = dict(userid=userid,**interaction.context)
            elif context_type == 1:
                igl_context = (userid,)+tuple(interaction.context)
            else:
                igl_context = (userid, interaction.context)

            yield SimulatedInteraction(igl_context, interaction.actions, igl_rewards, **interaction.kwargs, userid=userid, feedbacks=words, isnormal=isnormal)