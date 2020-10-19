"""The data module contains core classes and types for reading and writing data sources."""

import collections

from abc import abstractmethod, ABC
from build.lib.coba.json import CobaJsonDecoder
from pathlib import Path
from typing import Any, List, Iterable, Sequence, Dict, Hashable, overload

from coba.utilities import check_pandas_support
from coba.json import CobaJsonEncoder

class Table:
    """A container class for storing tabular data."""

    def __init__(self, name:str, primary: Sequence[str], default=float('nan')):
        """Instantiate a Table.
        
        Args:
            name: The name of the table.
            default: The default values to fill in missing values with
        """
        self._name    = name
        self._primary = primary
        self._columns = list(primary)
        self._default = default

        self._rows: Dict[Hashable, Sequence[Any]] = {}

    @overload
    def add_row(self, *row) -> None:
        """Add a row of data to the table. The row must contain all primary columns.
        
        Arg:
            row: The row of data in ordered `value` format. The value order must match `Table.columns`.
        """
        ...

    @overload
    def add_row(self, **kwrow) -> None:
        """Add a row of data to the table. The row must contain all primary columns.
        
        Arg:
            kwrow: The row of data in `column_name`:`value` format.
        """
        ...

    def add_row(self, *row, **kwrow) -> None:
        """Add a row of data to the table. The row must contain all primary columns."""

        if kwrow:
            self._columns.extend([col for col in kwrow if col not in self._columns])
            row = [ kwrow.get(col, self._default) for col in self._columns ]
        
        self._rows[row[0] if len(self._primary) == 1 else tuple(row[0:len(self._primary)])] = row

    def get_row(self, key: Hashable) -> Dict[str,Any]:
        row = self._rows[key]
        row = list(row) + [self._default] * (len(self._columns) - len(row))

        return {k:v for k,v in zip(self._columns,row)}

    def rmv_row(self, key: Hashable) -> None:
        self._rows.pop(key, None)

    def rmv_where(self, **kwrow) -> None:

        idx_val = [ (self._columns.index(col), val) for col,val in kwrow.items() ]
        rmv_keys  = []

        for key,row in self._rows.items():
            if all( row[i]==v for i,v in idx_val):
                rmv_keys.append(key)

        for key in rmv_keys: 
            del self._rows[key] 

    def to_tuples(self) -> Sequence[Any]:
        """Convert a table into a sequence of namedtuples."""
        return list(self.to_indexed_tuples().values())

    def to_indexed_tuples(self) -> Dict[Hashable, Any]:
        """Convert a table into a mapping of keys to tuples."""

        my_type = collections.namedtuple(self._name, self._columns) #type: ignore #mypy doesn't like dynamic named tuples
        my_type.__new__.__defaults__ = (self._default, ) * len(self._columns)
        
        return { key:my_type(*value) for key,value in self._rows.items() } #type: ignore #mypy doesn't like dynamic named tuples

    def to_pandas(self) -> Any:
        """Convert a table into a pandas dataframe."""

        check_pandas_support('Table.to_pandas')
        import pandas as pd #type: ignore #mypy complains otherwise

        return pd.DataFrame(self.to_tuples())

    def __contains__(self, primary) -> bool:

        if isinstance(primary, collections.Mapping):
            primary = list(primary.values())[0] if len(self._primary) == 1 else tuple([primary[col] for col in self._primary])

        return primary in self._rows

    def __str__(self) -> str:
        return str({"Table": self._name, "Columns": self._columns, "Rows": len(self._rows)})

    def __repr__(self) -> str:
        return str(self)

class ReadWrite(ABC):
    @abstractmethod
    def write(self, obj:Any) -> None:
        ...

    @abstractmethod
    def read(self) -> Iterable[Any]:
        ...

class DiskReadWrite(ReadWrite):
    
    def __init__(self, filename:str):
        self._json_encoder = CobaJsonEncoder()
        self._json_decoder = CobaJsonDecoder()
        self._filepath     = Path(filename)
        self._filepath.touch()

    def write(self, obj: Any) -> None:
        with open(self._filepath, "a") as f:
            f.write(self._json_encoder.encode(obj))
            f.write("\n")
    
    def read(self) -> Iterable[Any]:
        with open(self._filepath, "r") as f:
            for line in f.readlines():
                yield self._json_decoder.decode(line)

class MemoryReadWrite(ReadWrite):
    def __init__(self, memory: List[Any] = None):
        self._memory = memory if memory else []

    def write(self, obj:Any) -> None:
        self._memory.append(obj)

    def read(self) -> Iterable[Any]:
        return self._memory

class QueueReadWrite(ReadWrite):
    def __init__(self, queue: Any) -> None:
        self._queue = queue

    def write(self, obj:Any) -> None:
        self._queue.put(obj)

    def read(self) -> Iterable[Any]:
        while not self._queue.empty():
            yield self._queue.get()