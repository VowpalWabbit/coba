from typing import Mapping, Union, Iterable, Callable, Any

from coba.exceptions import CobaException
from coba.primitives import Filter, Source, Sink, Line

from coba.pipes.sources import SourceFilters
from coba.pipes.filters import FiltersFilter
from coba.pipes.sinks   import FiltersSink
from coba.pipes.lines   import SourceSink

class Foreach(Filter[Iterable[Any], Iterable[Any]], Sink[Iterable[Any]]):
    """A pipe that wraps an inner pipe and passes items to it one at a time."""

    def __init__(self, pipe: Union[Filter[Any,Any],Sink[Any]]):
        """Instantiate a Foreach pipe.

        Args:
            pipe: The pipe that we wish to pass a sequence of items one at a time.
        """
        self._pipe = pipe

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        """Filter the items using the inner pipe. This method only works if the inner pipe is a Filter."""
        yield from map(self._pipe.filter,items)

    def write(self, items: Iterable[Any]):
        """Write the items using the inner pipe. This method only works if the inner pipe is a Sink."""
        write = self._pipe.write
        for item in items:
            write(item)

    @property
    def params(self) -> Mapping[str,Any]:
        return self._pipe.params

    def __str__(self):
        return str(self._pipe)

class Pipes:
    @staticmethod
    def join(*pipes: Union[Source, Filter, Sink]) -> Union[Source, Filter, Sink, Line]:
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

        if len(pipes) == 1 and hasattr(pipes[0],'read'):
            return SourceFilters(*pipes)

        if len(pipes) == 1 and hasattr(pipes[0],'filter'):
            return FiltersFilter(*pipes)

        if len(pipes) == 1 and hasattr(pipes[0],'write'):
            return FiltersSink(*pipes)

        first = pipes[0 ] if not isinstance(pipes[0 ], Foreach) else pipes[0 ]._pipe
        last  = pipes[-1] if not isinstance(pipes[-1], Foreach) else pipes[-1]._pipe

        if hasattr(first,'read') and hasattr(last,'write'):
            return SourceSink(*pipes)

        if hasattr(first,'read') and hasattr(last,'filter'):
            return SourceFilters(*pipes)

        if hasattr(first,'filter') and hasattr(last,'filter'):
            return FiltersFilter(*pipes)

        if hasattr(first,'filter') and hasattr(last,'write'):
            return FiltersSink(*pipes)

        raise CobaException("An unknown pipe was passed to join.")

def join(*pipes: Union[Source, Filter, Sink]) -> Union[Source, Filter, Sink, Line]:
    return Pipes.join(*pipes)
