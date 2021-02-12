"""The data.structures module contains basic coba datastructures. """

import collections

from typing import Sequence, Hashable, Any, Dict, Iterable

from coba.tools import check_pandas_support

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

        self.rows: Dict[Hashable, Sequence[Any]] = {}

    def add_row(self, *row, **kwrow) -> None:
        """Add a row of data to the table. The row must contain all primary columns."""

        if kwrow:
            self._columns.extend([col for col in kwrow if col not in self._columns])

        row = row + tuple( kwrow.get(col, self._default) for col in self._columns[len(row):] ) #type:ignore
        self.rows[row[0] if len(self._primary) == 1 else tuple(row[0:len(self._primary)])] = row

    def get_row(self, key: Hashable) -> Dict[str,Any]:
        row = self.rows[key]
        row = list(row) + [self._default] * (len(self._columns) - len(row))

        return {k:v for k,v in zip(self._columns,row)}

    def rmv_row(self, key: Hashable) -> None:
        self.rows.pop(key, None)

    def get_where(self, **kwargs) -> Iterable[Dict[str,Any]]:

        if any([k not in self._columns for k in kwargs]):
            return

        idx_val = [ (self._columns.index(col), val) for col,val in kwargs.items() ]

        for key,row in self.rows.items():
            if all( row[i]==v for i,v in idx_val):
                yield {k:v for k,v in zip(self._columns,row)}

    def rmv_where(self, **kwrow) -> None:
        idx_val = [ (self._columns.index(col), val) for col,val in kwrow.items() ]
        rmv_keys  = []

        for key,row in self.rows.items():
            if all( row[i]==v for i,v in idx_val):
                rmv_keys.append(key)

        for key in rmv_keys: 
            del self.rows[key] 

    def to_tuples(self) -> Sequence[Any]:
        """Convert a table into a sequence of namedtuples."""
        return list(self.to_indexed_tuples().values())

    def to_indexed_tuples(self) -> Dict[Hashable, Any]:
        """Convert a table into a mapping of keys to tuples."""

        my_type = collections.namedtuple(self._name, self._columns) #type: ignore #mypy doesn't like dynamic named tuples
        my_type.__new__.__defaults__ = (self._default, ) * len(self._columns) #type: ignore #mypy doesn't like dynamic named tuples
        
        return { key:my_type(*value) for key,value in self.rows.items() } #type: ignore #mypy doesn't like dynamic named tuples

    def to_pandas(self) -> Any:
        """Convert a table into a pandas dataframe."""

        check_pandas_support('Table.to_pandas')
        import pandas as pd #type: ignore #mypy complains otherwise

        return pd.DataFrame(self.to_tuples())

    def __contains__(self, primary) -> bool:

        if isinstance(primary, collections.Mapping):
            primary = list(primary.values())[0] if len(self._primary) == 1 else tuple([primary[col] for col in self._primary])

        return primary in self.rows

    def __getitem__(self, key) -> Dict[str,Any]:
        return self.get_row(key)

    def __str__(self) -> str:
        return str({"Table": self._name, "Columns": self._columns, "Rows": len(self.rows)})

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return len(self.rows)