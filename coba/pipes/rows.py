from collections import abc
from typing import Any, Union, Callable, Iterator, Mapping, Sequence, overload

from coba.exceptions import CobaException
from coba.pipes.primitives import Filter

# parse -> encode -> sparse/dense row (which won't be evaluated until __getitem__ is called)

# this is the supervised simulation pipe...
#parse -> encode -> drop -> structure -> ({feats}, lbl) or ([feats], lbl)

# this is the pipe for anyone who just wants the data...
#parse -> encode -> drop -> [rows with headers] or {rows}

class Row:

    def prep(item: Union[Sequence,Mapping]) -> Union['DenseRow','SparseRow']:
        if isinstance(item,(DenseRow,SparseRow)):
            return item
        elif isinstance(item,abc.Sequence):
            return DenseRow(item, False)
        elif isinstance(item,abc.Mapping):
            return SparseRow(item, False)
        else: #pragma: no cover
            raise CobaException("An unrecognized item was passed to Row.prep")

class ParseRow:

    def __init__(self, line:str, parser: Callable[[str], Union[Sequence,Mapping]] = None, any_missing: bool = False) -> None:
        self._row         = line
        self._parser      = parser or (lambda x:x)
        self._any_missing = any_missing

        def default_get(key):
            return self.row.__getitem__(key)

        self._keys = [ lambda: list(range(len(self.row))) if isinstance(self.row, abc.Sequence) else self.row.keys()]
        self._gets = [ default_get ]

    def add_decorator(self, keys_decorator, gets_decorator) -> None:

        keys_decorator = keys_decorator or (lambda x  :x)
        gets_decorator = gets_decorator or (lambda x,y:x)

        new_keys = keys_decorator(self._keys[-1])
        new_gets = gets_decorator(self._gets[-1], new_keys)

        self._keys.append(new_keys)
        self._gets.append(new_gets)

    @property
    def row(self)->Union[list,dict]:
        if self._parser:
            self._row = self._parser(self._row)
            self._parser = None
        return self._row

    @property
    def any_missing(self) -> bool:
        return self._any_missing

    def keys(self) -> Sequence[Union[int,str]]:
        return self._keys[-1]()

    def __getitem__(self, key: Union[int, str]) -> Any:
        return self._gets[-1](key)

class DenseRow(abc.Sequence):
    
    @overload
    def __init__(self, sequence: Sequence, any_missing: bool=False) -> None:
        ...
    
    @overload
    def __init__(self, line:str, parse: Callable[[str],Sequence], any_missing: bool) -> None:
        ...
    
    def __init__(self, *args) -> None:
        if len(args)==3:
            self._row = ParseRow(args[0],args[1],args[2]) 
        elif len(args) ==2: 
            self._row = ParseRow(args[0],any_missing=args[1])
        else:
            self._row = ParseRow(args[0],any_missing=False)

    def add_decorator(self, keys_decorator, gets_decorator) -> None:
        self._row.add_decorator(keys_decorator,gets_decorator)

    @property
    def any_missing(self) -> bool:
        return self._row.any_missing

    def keys(self) -> Sequence[int]:
        return self._row.keys()

    def __getitem__(self, key: int) -> Any:
        return self._row[key]

    def __len__(self) -> int:
        return len(self.keys())

    def __eq__(self, o: object) -> bool:
        return isinstance(o,abc.Sequence) and list(self) == list(o)

    def __repr__(self) -> str:
        return list(self).__repr__()

    def __str__(self) -> str:
        return list(self).__str__()

class SparseRow(abc.Mapping):
    @overload
    def __init__(self, mapping: Mapping, any_missing: bool = False) -> None:
        ...
    @overload
    def __init__(self, line:str, parse: Callable[[str],Sequence], any_missing: bool) -> None:
        ...
    def __init__(self, *args) -> None:        
        
        if len(args)==3:
            self._row = ParseRow(args[0],args[1],args[2]) 
        elif len(args) ==2: 
            self._row = ParseRow(args[0],any_missing=args[1])
        else:
            self._row = ParseRow(args[0],any_missing=False)

    def add_decorator(self, keys_decorator, gets_decorator) -> None:
        self._row.add_decorator(keys_decorator,gets_decorator)

    @property
    def any_missing(self) -> bool:
        return self._row.any_missing

    def __getitem__(self, key: int) -> Any:
        try:
            return self._row[key]
        except KeyError:
            return 0

    def __iter__(self) -> Iterator[str]:
        return iter(self._row.keys())

    def __len__(self) -> int:
        return len(self._row.keys())

    def __eq__(self, o: object) -> bool:
        return isinstance(o,abc.Mapping) and dict(self) == dict(o)

    def __repr__(self) -> str:
        return dict(self).__repr__()

    def __str__(self) -> str:
        return dict(self).__str__()

class DropRow(Filter[Union[Sequence,Mapping],Union[Sequence,Mapping]]):

    def __init__(self, cols: Sequence) -> None:
        self._is_int = cols and isinstance(list(cols)[0],int)
        self._cols   = sorted(set(cols),reverse=True)

    def filter(self, item: Union[Sequence,Mapping]) -> Union[Sequence,Mapping]:

        if not self._cols: return item

        if isinstance(item,dict):
            item = dict(item)
            for c in self._cols: del item[c]
        
        elif isinstance(item,list):
            item = list(item)
            for c in self._cols: del item[c]
        
        elif isinstance(item,(DenseRow,SparseRow)):    
            def int_keys_decorator(old_keys):
                def decorated():
                    is_not_excluded_int = lambda i,k: (self._is_int and i not in self._cols)
                    is_not_excluded_str = lambda i,k: (not self._is_int and k not in self._cols)
                    return [i for i,k in enumerate(old_keys()) if is_not_excluded_int(i,k) or is_not_excluded_str(i,k) ]
                return decorated

            def str_keys_decorator(old_keys):
                def decorated():
                    return [k for k in old_keys() if k not in self._cols]
                return decorated

            def int_gets_decorator(old_get,new_keys):
                def decorated(key):
                    return old_get(new_keys()[key] if isinstance(key,int) else key)
                return decorated

            def str_gets_decorator(old_get,new_keys):
                def decorated(key):
                    return old_get(key)
                return decorated

            if isinstance(item, DenseRow):
                item.add_decorator(int_keys_decorator, int_gets_decorator)
            else:
                item.add_decorator(str_keys_decorator, str_gets_decorator)
        else:
            raise CobaException("Unrecognized row type passed to DropRow.")
        return item

class EncodeRow(Filter[Union[Sequence,Mapping],Union[Sequence,Mapping]]):

    def __init__(self, encoders: Union[Sequence,Mapping]) -> None:
        self._encoders = encoders

    def filter(self, item: Union[Sequence,Mapping]) -> Union[Sequence,Mapping]:
        row = Row.prep(item)
        
        def gets_decorator(old_get,new_keys):
            def decorated(key):
                return self._encoders[key](old_get(key))
            return decorated

        row.add_decorator(None, gets_decorator) 
        
        return row

class IndexRow(Filter[Union[Sequence,Mapping],Union[Sequence,Mapping]]):

    def __init__(self, index: Mapping) -> None:
        self._fwd_index = index
        self._rev_index = { v:k for k,v in index.items()}

    def filter(self, item: Union[Sequence,Mapping]) -> Union[Sequence,Mapping]:
        row = Row.prep(item)
        
        def keys_decorator(old_keys):
            def decorator():
                return [ self._rev_index[k] for k in old_keys() ]
            return decorator

        def gets_decorator(old_get,new_keys):
            def decorator(key):
                return old_get(self._fwd_index[key] if isinstance(key,str) else key)
            return decorator

        row.add_decorator(keys_decorator,gets_decorator)

        return row

class LabelRow(Filter[Union[Sequence,Mapping],Union[Sequence,Mapping]]):

    def __init__(self, label: Union[int,str]) -> None:
        self._label = label

    def filter(self, item: Union[Sequence,Mapping]) -> Union[Sequence,Mapping]:
        row = Row.prep(item)
        
        def gets_decorator(old_get,new_keys):
            def decorator(key):
                return old_get(key if isinstance(key,str) else new_keys()[key])
            return decorator

        def keys_decorator(old_keys):
            def decorator():
                return [self._label] + [ k for k in old_keys() if k != self._label ]
            return decorator

        row.add_decorator(keys_decorator,gets_decorator)

        return row
