from collections import Counter
from typing import Sequence

from coba.primitives import Pipe

def resolve_params(pipes: Sequence[Pipe]):

    params = [p.params for p in pipes if hasattr(p,'params')]
    keys   = [ k for p in params for k in p.keys() ]
    counts = Counter(keys)
    index  = {}

    def resolve_key_conflicts(key):
        if counts[key] == 1:
            return key
        else:
            index[key] = index.get(key,0)+1
            return f"{key}{index[key]}"

    return { resolve_key_conflicts(k):v for p in params for k,v in p.items() }
