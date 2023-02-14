
from typing import Iterable, Optional

from pandas import DataFrame

from coba.primitives.rewards import SequenceReward
from coba.environments.primitives import Environment, LoggedInteraction
from coba.pipes import IterableSource
from coba.utilities import peek_first



class BanditReplay(Environment):
    def __init__(self, df: DataFrame,take: Optional[int] = None):
        self._df = df[:take] if take is not None else df

    def read(self) -> Iterable[LoggedInteraction]:
        source = IterableSource(self._df.iterrows())
        first, rows = peek_first(source.read())
        for _, row in rows:
            kwargs = {
                'context'    : row['context'],
                'action'     : row['action'],
                'reward'     : row['reward'],
                'probability': row['probability'],
                'actions'    : row['actions'],
            }
            if row.get('rewards') is not None:
                kwargs.update({'rewards': SequenceReward(row['rewards'])})
            yield LoggedInteraction(**kwargs)
