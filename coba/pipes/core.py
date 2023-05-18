from typing import Union

from coba.exceptions import CobaException

from coba.pipes.primitives import Filter, Source, Sink, SourceFilters, FiltersFilter, FiltersSink, Foreach
from coba.pipes.multiprocessing import AsyncableLine

class Pipes:

    @staticmethod
    def join(*pipes: Union[Source, Filter, Sink]) -> Union[Source, Filter, Sink, AsyncableLine]:
        """Join a sequence of pipes into a single pipe.

        Args:
            pipes: a sequence of pipes.

        Returns:
            A single pipe that is a composition of the given pipes. The type of pipe returned
            is determined by the sequence given. A sequence of Filters will return a Filter. A
            sequence that begins with a Source and is followed by Filters will return a Source.
            A sequence that starts with Filters and ends with a Sink will return a Sink. A
            sequence that begins with a Source and ends with a Sink will return a completed pipe.
        """
        if len(pipes) == 0:
            raise CobaException("No pipes were passed to join.")

        if len(pipes) == 1 and any(hasattr(pipes[0],attr) for attr in ['read','filter','write']):
            return pipes[0]

        first = pipes[0 ] if not isinstance(pipes[0 ], Foreach) else pipes[0 ]._pipe
        last  = pipes[-1] if not isinstance(pipes[-1], Foreach) else pipes[-1]._pipe

        if hasattr(first,'read') and hasattr(last,'write'):
            return AsyncableLine(*pipes)

        if hasattr(first,'read') and hasattr(last,'filter'):
            return SourceFilters(*pipes)

        if hasattr(first,'filter') and hasattr(last,'filter'):
            return FiltersFilter(*pipes)

        if hasattr(first,'filter') and hasattr(last,'write'):
            return FiltersSink(*pipes)

        raise CobaException("An unknown pipe was passed to join.")
