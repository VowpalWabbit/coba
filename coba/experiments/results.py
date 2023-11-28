import re
import collections
import collections.abc

from bisect import bisect_left, bisect_right
from pathlib import Path
from numbers import Number
from operator import truediv, sub, mul, itemgetter, methodcaller
from abc import abstractmethod
from dataclasses import dataclass, astuple, field, replace
from itertools import chain, repeat, accumulate, groupby, count, compress, groupby, tee, islice
from typing import Mapping, Tuple, Optional, Sequence, Iterable, Iterator, Union, Callable, List, Any, overload, Literal

from coba.primitives import is_batch
from coba.environments import Environment
from coba.statistics import mean, StdDevCI, StdErrCI, BootstrapCI, BinomialCI, PointAndInterval
from coba.context import CobaContext
from coba.exceptions import CobaException
from coba.utilities import PackageChecker, peek_first, KeyDefaultDict
from coba.pipes import Pipes, JsonEncode, JsonDecode, DiskSource, Source

def moving_average(values:Sequence[float], span:Union[int,Sequence[float]]=None, weights:Union[Literal['exp'],Sequence[float]]=None) -> Iterable[float]:

    assert weights == 'exp' or weights == None or len(weights)==len(values)

    if weights=='exp':
        #exponential moving average identical to Pandas' df.ewm(span=span).mean()
        alpha = 2/(1+span)
        cumwindow  = list(accumulate(values          , lambda a,v: v + (1-alpha)*a))
        cumdivisor = list(accumulate([1.]*len(values), lambda a,v: v + (1-alpha)*a))
        return map(truediv, cumwindow, cumdivisor)

    elif span == 1:
        return values

    elif span is None or span >= len(values):
        values  = accumulate(values) if not weights else accumulate(map(mul,values,weights))
        weights = count(1)           if not weights else accumulate(weights)
        return map(truediv, values, weights)

    else:
        v1,v2   = tee(values    if not weights else map(mul,values,weights),2)
        weights = repeat(1) if not weights else weights

        values  = accumulate(map(sub, v1     , chain(repeat(0,span),v2     )))
        weights = accumulate(map(sub, weights, chain(repeat(0,span),weights)))

        return map(truediv,values,weights)

#this adds one more check on average but avoids the worst case
#scenario, which can be common for certain types of experiments.
def my_bisect_left(c,a,l,h): return l if c[l]==a else bisect_left(c,a,l,h)
def my_bisect_right(c,a,l,h): return h if c[h-1]==a else bisect_right(c,a,l,h)

class MyComparable:
        __slots__ = ('obj',)

        #See this for more information: https://stackoverflow.com/q/68173281/1066291

        def __init__(self, obj):
            self.obj = obj

        def __lt__(self, other):
            try:
                return self.obj < other.obj
            except TypeError as e:
                err_msg = str(e)
                return err_msg.endswith("'NoneType'") if "'NoneType'" in err_msg else err_msg.endswith("'str'")

        __hash__ = None

class View:
    __slots__ = ('_data','_select')

    @staticmethod
    def _try_slice(select: Sequence[int]):
        if select and not isinstance(select,slice) and (select[-1]-select[0]+1) == len(select):
            return slice(select[0], select[-1]+1)
        else:
            return select

    class ListView:
        __slots__=('_seq','_sel')
        def __init__(self, seq:Sequence, sel:Sequence):
            self._seq = seq
            self._sel = sel

        def __len__(self):
            return len(self._sel)

        def __getitem__(self, key:Union[int,slice]):
            if not isinstance(key,slice):
                return self._seq[self._sel[key]]
            else:
                return list(map(self._seq.__getitem__,self._sel[key]))

        def __iter__(self):
            return iter(map(self._seq.__getitem__,self._sel))

    class SliceView:
        __slots__=('_seq','_sel')
        def __init__(self, seq:Sequence, sel:slice):
            self._seq = seq
            self._sel = sel

        def __len__(self):
            return (self._sel.stop or len(self._seq)) - (self._sel.start or 0)

        def __getitem__(self, key:Union[int,slice]):
            if not isinstance(key,slice):
                return self._seq[(self._sel.start or 0)+key]
            else:
                key_start = key.start or 0
                key_stop  = key.stop or len(self)
                new_start = (self._sel.start or 0) + key_start
                new_stop  = new_start + key_stop - key_start
                return self._seq[slice(new_start,new_stop)]

        def __iter__(self):
            #this isn't a true iter because it makes a copy. It
            #still runs slightly faster than doing a true iter
            #with islice.
            return iter(self._seq[self._sel])

    def __init__(self, data: Union[Mapping[str,Sequence],'View'], select: Union[Sequence[int],slice]) -> None:
        if isinstance(data,collections.abc.Mapping):
            self._data   = data
            self._select = select if isinstance(select,slice) else View._try_slice(select)

        else: #data is a 'View'
            self._data = data._data

            given_slice = isinstance(select,slice)
            have_slice  = isinstance(data._select,slice)

            if given_slice and have_slice:
                self._select = slice(data._select.start+select.start,data._select.start+select.stop)
            if given_slice and not have_slice:
                self._select = View._try_slice(data._select[select])
            if not given_slice and have_slice:
                self._select = View._try_slice([data._select.start + i for i in select])
            if not given_slice and not have_slice:
                self._select = View._try_slice([data._select[i] for i in select])

    def keys(self):
        return self._data.keys()

    def values(self):
        return [self[k] for k in self]

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self,key):
        if isinstance(self._select,slice):
            return View.SliceView(self._data[key],self._select)
        else:
            return View.ListView(self._data[key],self._select)

    def __setitem__(self,key,value):
        raise CobaException("A view of the data cannot be modified.")

    def __contains__(self,key) -> bool:
        return key in self._data

class Table:
    __slots__ = ('_data', '_columns', '_indexes', '_lohis')
    """A container class for storing tabular data."""
    #Potentially overkill, however, by having our own "simple" table implementation we can provide
    #several useful pieces of functionality out of the box. Additionally, when working with
    #very large experiments pandas can become quite slow while Table works acceptably.

    def __init__(self, data:Union[Mapping, Sequence[Mapping], Sequence[Sequence]] = (), columns: Sequence[str] = (), indexes: Sequence[str]= ()):

        self._columns = tuple(columns) or tuple(data)
        self._lohis   = None

        data_is_view            = isinstance(data,View)
        data_is_mapping_of_cols = isinstance(data,collections.abc.Mapping)

        if data_is_view:
            self._data = data
        elif data_is_mapping_of_cols and data.keys() == set(columns):
            self._data = data
        elif data_is_mapping_of_cols and not columns:
            self._data = data
        else:
            self._data = {c:[] for c in self._columns}
            self.insert(data)

        self._columns = self._columns or tuple(self._data.keys())

        self._indexes = tuple(indexes) #this should come last...

    @property
    def columns(self) -> Sequence[str]:
        return self._columns

    @property
    def indexes(self) -> Sequence[str]:
        return self._indexes

    def insert(self, data:Union[Mapping, Sequence[Mapping], Sequence[Sequence]]=()) -> 'Table':
        data_is_empty             = not data
        data_is_mapping_of_cols   = data and isinstance(data,collections.abc.Mapping)
        data_is_sequence_of_dicts = data and isinstance(data,collections.abc.Sequence) and isinstance(data[0],collections.abc.Mapping)

        if data_is_empty:
            return self

        if data_is_sequence_of_dicts:
            data = {k:[d.get(k) for d in data] for k in set().union(*(d.keys() for d in data))}
            data_is_mapping_of_cols = True

        if data_is_mapping_of_cols:
            old,new = set(self._columns), data.keys()

            new_cols = new-old
            old_cols = new&old
            pad_cols = old-new

            if new_cols: old_len = len(self)
            if pad_cols: new_len = old_len+len(next(iter(data.values())))

            for hdr in new_cols:
                self._data[hdr] = list(chain(repeat(None, old_len), data[hdr]))

            for hdr in old_cols:
                self._data[hdr].extend(data[hdr])

            for hdr in pad_cols:
                self._data[hdr].extend(repeat(None,new_len-old_len))

            if new_cols:
                self._columns += tuple(sorted(new_cols))

        else: #data is a sequence of rows

            assert len(data[0]) == len(self._columns), "The given data rows don't align with the table's headers."
            for hdr,col in zip(self._columns,zip(*data)):
                self._data[hdr].extend(col)

        if self._lohis: self._lohis = {}

        return self

    def index(self, *indx) -> 'Table':

        if not indx: return self
        if not self._data: return self
        indx = [col for col in indx if col in self._columns]
        if self._indexes == tuple(indx): return self

        lohis   = [(0,len(self))]
        indexes = list(range(len(self)))

        for col in indx:
            for lo,hi in lohis:
                indexes[lo:hi] = sorted(indexes[lo:hi],key=self._data[col].__getitem__)
            self._data[col][:] = map(self._data[col].__getitem__,indexes)
            if col != indx[-1]: lohis = list(chain.from_iterable(self._sub_lohis(lo, hi, self._data[col]) for lo, hi in lohis))

        for col in self._data.keys()-set(indx):
            self._data[col][:] = map(self._data[col].__getitem__,indexes)

        self._indexes = tuple(indx)
        self._lohis = self._calc_lohis()

        return self

    def where(self, row_pred:Callable[[Sequence],bool] = None, comparison:Literal['=','!=','<=','<','>','>=','match','in'] = None, **kwargs) -> 'Table':
        """Filter to specific rows.

        Args:
            pred: A predicate that accepts rows and returns true for rows that should be kept.
            kwargs: key value pairs where the key is the column and the value indicates what
                value a row should have in that column to be kept. Keeping logic depends on
                the row value type and the kwargs value type. If kwarg value == row value keep
                the row. If kwarg value is callable pass the row value to the predicate. If
                kwarg value is a collection keep the row if the row value is in the collection.
                If kwarg value is a string apply a regular expression match to the row value.
        """

        if not row_pred and not kwargs:
            return self

        if row_pred:
            selection = list(compress(count(),map(row_pred,self)))
            return Table(View(self._data,selection), self._columns, self._indexes)

        self._lohis = self._lohis or self._calc_lohis()

        if kwargs:
            selection = []
            for kw,arg in kwargs.items():
                if kw in self._indexes and comparison != "match" and not callable(kwargs[kw]):
                    for lo,hi in self._lohis[kw]:
                        for l,h in self._compare(lo,hi,self._data[kw],arg,comparison,"bisect"):
                            selection.extend(range(l,h))
                else:
                    selection.extend(self._compare(0,len(self),self._data[kw],arg,comparison,"foreach"))

            if len(kwargs) > 1: selection=sorted(set(selection))

            return Table(View(self._data,selection), self._columns, self._indexes)

    def groupby(self, level:int, select:Union[None,Literal['table','count'],str]='table') -> Iterable[Tuple[Tuple,Any]]:

        self._lohis = self._lohis or self._calc_lohis()
        grp_cols = [self._data[hdr] for hdr in self._indexes[:level]]

        if not select:
            for l,h in self._lohis[self._indexes[level]]:
                yield tuple(map(itemgetter(l),grp_cols))
        elif select=='table':
            for l,h in self._lohis[self._indexes[level]]:
                grp_idx = tuple(map(itemgetter(l),grp_cols))
                grp_tbl = Table(View(self._data, slice(l,h)), self._columns, self._indexes)
                yield (grp_idx,grp_tbl)
        elif select=='count':
            for l,h in self._lohis[self._indexes[level]]:
                grp_idx = tuple(map(itemgetter(l),grp_cols))
                yield (grp_idx,h-l)
        else:
            for l,h in self._lohis[self._indexes[level]]:
                grp_idx = tuple(map(itemgetter(l),grp_cols))
                yield (grp_idx,self._data[select][l:h])

    def copy(self) -> 'Table':
        t = Table(self._data, tuple(self._columns), tuple(self._indexes))
        t._lohis = self._lohis
        return t

    def to_pandas(self):
        """Turn the Table into a Pandas data frame."""

        PackageChecker.pandas("Table.to_pandas")
        import pandas as pd

        #data must be dict instance to work as desired
        data = {k:self._data[k] for k in self._data}
        return pd.DataFrame(data, columns=self.columns)

    def to_dicts(self) -> Iterable[Mapping[str,Any]]:
        """Turn the Table into a sequence of tuples."""

        keys = repeat(self._columns)
        vals = zip(*map(self._data.__getitem__,self._columns))
        yield from map(dict,map(zip,keys,vals))

    def __getitem__(self,idx1):

        if isinstance(idx1,str):
            return self._data[idx1]

        if isinstance(idx1,slice):
            return [self._data[col] for col in self._columns[idx1]]

        try:
            return [self._data[col] for col in idx1]
        except:
            raise KeyError(idx1)

    def __iter__(self) -> Iterator[Sequence[Any]]:
        return iter(zip(*self[:]))

    def __str__(self) -> str:
        return str({"Columns": self.columns, "Rows": len(self)})

    def __len__(self) -> int:
        return len(next(iter(self._data.values()))) if self._data else 0

    def __eq__(self, o: object) -> bool:
        return isinstance(o,Table) and o._data == self._data

    def _ipython_display_(self):
        #pretty print in jupyter notebook (https://ipython.readthedocs.io/en/stable/config/integrating.html)
        print(str(self))

    def _calc_lohis(self):

        if not self._indexes: return {}

        lohis = [[(0,len(self))]]

        for k in self._indexes[:-1]:
            indexcol = self._data[k]
            lohis.append([lh for lo,hi in lohis[-1] for lh in self._sub_lohis(lo,hi,indexcol)])

        return dict(zip(self._indexes,lohis))

    def _sub_lohis(self,lo,hi,col):
        while lo != hi:
            new_hi = my_bisect_right(col,col[lo],lo,hi)
            yield (lo,new_hi)
            lo = new_hi

    def _compare(self,lo,hi,col,arg,comparison,method):

        if isinstance(arg,dict) and len(arg) == 1:
            key,value = list(arg.items())[0]
            if key in ['=','!=','<=','<','>','>=','match','in']:
                comparison,arg = key,value

        if method != "bisect" or callable(arg):
            col = col[lo:hi]

        if callable(arg):
            return list(compress(count(lo),map(arg,col)))

        if comparison == "in" or (comparison is None and isinstance(arg,collections.abc.Iterable) and not isinstance(arg,str)):
            if method == "bisect":
                return [ (my_bisect_left(col,v,lo,hi),my_bisect_right(col,v,lo,hi)) for v in sorted(arg) ]
            else:
                return [ i for i,c in enumerate(col,lo) if c in arg ]

        if comparison == "!in":
            if method == "bisect":
                arg = [None]+list(sorted(arg))+[None]
                return [ (lo if v0 is None else my_bisect_right(col,v0,lo,hi), hi if v1 is None else my_bisect_left(col,v1,lo,hi)) for v0,v1 in zip(arg[0:],arg[1:])]
            else:
                return [ i for i,c in enumerate(col,lo) if c not in arg ]

        if comparison == "=" or (comparison is None and (not isinstance(arg,collections.abc.Iterable) or isinstance(arg,str))):
            if method == "bisect":
                return [ (my_bisect_left(col,arg,lo,hi),my_bisect_right(col,arg,lo,hi)) ]
            else:
                return [ i for i,c in enumerate(col,lo) if c == arg ]

        if comparison == "!=":
            if method == "bisect":
                return [ (lo,my_bisect_left(col,arg,lo,hi)), (my_bisect_right(col,arg,lo,hi), hi) ]
            else:
                return [ i for i,c in enumerate(col,lo) if c != arg ]

        if comparison == "<":
            if method == "bisect":
                return [ (lo,my_bisect_left(col,arg,lo,hi)) ]
            else:
                return [ i for i,c in enumerate(col,lo) if c < arg ]

        if comparison == "<=":
            if method == "bisect":
                return [ (lo,my_bisect_right(col,arg,lo,hi)) ]
            else:
                return [ i for i,c in enumerate(col,lo) if c <= arg ]

        if comparison == ">=":
            if method == "bisect":
                return [ (my_bisect_left(col,arg,lo,hi), hi) ]
            else:
                return [ i for i,c in enumerate(col,lo) if c >= arg ]

        if comparison == ">":
            if method == "bisect":
                return [ (my_bisect_right(col,arg,lo,hi), hi) ]
            else:
                return [ i for i,c in enumerate(col,lo) if c > arg ]

        if comparison == "match":
            if isinstance(arg,Number) and col and isinstance(col[0],Number):
                return [ i for i,c in enumerate(col,lo) if c == arg ]
            elif isinstance(arg,Number) and isinstance(col[0],str):
                _re = re.compile(f'(\D|^){arg}(\D|$)')
                return [ i for i,c in enumerate(col,lo) if _re.search(c) ]
            elif isinstance(arg,str) and isinstance(col[0],str):
                _re = re.compile(arg)
                return [ i for i,c in enumerate(col,lo) if _re.search(c) ]

class TransactionDecode:
    def filter(self, transactions:Iterable[str]) -> Iterable[Any]:

        transactions = iter(transactions)
        ver_row = JsonDecode().filter(next(transactions))

        if ver_row[1] == 4:
            yield ver_row
            yield from map(JsonDecode().filter,transactions)

class TransactionEncode:
    def __init__(self,restored):
        self._restored = restored

    def filter(self, transactions: Iterable[Any]) -> Iterable[str]:

        if not self._restored:
            yield JsonEncode().filter(["version",4])

        encoder = JsonEncode()

        for item in transactions:
            if item[0] == "T0":
                yield encoder.filter(['experiment', item[1]])

            elif item[0] == "T1":
                yield encoder.filter(["E", item[1], item[2]])

            elif item[0] == "T2":
                yield encoder.filter(["L", item[1], item[2]])

            elif item[0] == "T3":
                yield encoder.filter(["V", item[1], item[2]])

            elif item[0] == "T4":
                rows_T = collections.defaultdict(list)

                keys = sorted(set().union(*[r.keys() for r in item[2]]))

                for row in item[2]:
                    for key in keys:
                        rows_T[key].append(row.get(key,None))

                yield encoder.filter(["I", item[1], { "_packed": rows_T }])

class TransactionResult:

    def filter(self, transactions:Iterable[Any]) -> 'Result':
        env_rows = collections.defaultdict(dict)
        lrn_rows = collections.defaultdict(dict)
        val_rows = collections.defaultdict(dict)
        int_rows = {}
        exp_dict = {}

        transactions = iter(transactions)
        version      = next(transactions)[1]

        if version == 3:
            raise CobaException("Deprecated transaction format. Please revert to an older version of Coba to read it.")

        if version != 4:
            raise CobaException(f"Unrecognized transaction format: version equals {version}.")

        for trx in transactions:

            if not trx: continue

            if trx[0] == "experiment":
                exp_dict = trx[1]

            if trx[0] == "E":
                env_rows[trx[1]].update(trx[2])

            if trx[0] == "L":
                lrn_rows[trx[1]].update(trx[2])

            if trx[0] == "V":
                val_rows[trx[1]].update(trx[2])

            if trx[0] == "I":
                if len(trx[1]) == 2: trx[1] = [*trx[1],0]
                int_rows[tuple(trx[1])] = trx[2]

        rwd_col = ['reward'] if any('reward' in v.keys() for v in int_rows.values()) else []

        env_table = Table(columns=['environment_id'                                    ]          )
        lrn_table = Table(columns=[                 'learner_id'                       ]          )
        val_table = Table(columns=[                              'evaluator_id'        ]          )
        int_table = Table(columns=['environment_id','learner_id','evaluator_id','index'] + rwd_col)

        #we manually sort below so everything is indexed correctly
        #this is considerably faster than sorting after the fact
        env_table.index('environment_id'                                    )
        lrn_table.index(                 'learner_id'                       )
        val_table.index(                              'evaluator_id'        )
        int_table.index('environment_id','learner_id','evaluator_id','index')

        env_table.insert([{"environment_id":e,                                 **r} for e,r in sorted(env_rows.items())])
        lrn_table.insert([{                   "learner_id":l,                  **r} for l,r in sorted(lrn_rows.items())])
        val_table.insert([{                                  "evaluator_id":v, **r} for v,r in sorted(val_rows.items())])

        for (env_id, lrn_id, val_id), results in sorted(int_rows.items()):
            if '_packed' in results and results['_packed']:

                packed = results['_packed']
                N = len(next(iter(packed.values())))

                packed['environment_id'] = repeat(env_id,N)
                packed['learner_id'    ] = repeat(lrn_id,N)
                packed['evaluator_id'  ] = repeat(val_id,N)
                packed['index'         ] = range(1,N+1)

                int_table.insert(packed)

        return Result(env_table, lrn_table, val_table, int_table, exp_dict)

@dataclass
class Points:
    X    : List[Any]        = field(default_factory=list)
    Y    : List[float]      = field(default_factory=list)
    XE   : List[float]      = field(default_factory=list)
    YE   : List[float]      = field(default_factory=list)
    color: Union[str,int]   = None
    alpha: float            = 1
    label: Optional[str]    = None
    style: Literal['-','.'] = "-"
    zorder:int              = 1

    def add(self, x, y, ye):
        self.X.append(x)
        self.Y.append(y)
        self.YE.append(ye)

class Plotter:
    @abstractmethod
    def plot(self,
        ax,
        lines: Sequence[Points],
        title: str,
        xlabel: str,
        ylabel: str,
        xlim: Tuple[Optional[Number],Optional[Number]],
        ylim: Tuple[Optional[Number],Optional[Number]],
        xticks: bool,
        yticks: bool,
        xrotation: Optional[float],
        yrotation: Optional[float],
        xorder: Optional[Sequence[str]],
        out: Union[None,Literal['screen'],str]) -> None:
        pass

class MatplotPlotter(Plotter):

    def plot(self,
        ax,
        lines: Sequence[Points],
        title: str,
        xlabel: str,
        ylabel: str,
        xlim: Tuple[Optional[Number],Optional[Number]],
        ylim: Tuple[Optional[Number],Optional[Number]],
        xticks: bool,
        yticks: bool,
        xrotation: Optional[float],
        yrotation: Optional[float],
        xorder: Optional[Sequence[str]],
        out: Union[None,Literal['screen'],str]
    ) -> None:

        xlim = xlim or [None,None]
        ylim = ylim or [None,None]

        PackageChecker.matplotlib('MatplotPlotter')
        import matplotlib.pyplot as plt

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        ax: plt.Axes

        bad_xlim  = xlim and xlim[0] is not None and xlim[1] is not None and xlim[0] >= xlim[1]
        bad_ylim  = ylim and ylim[0] is not None and ylim[1] is not None and ylim[0] >= ylim[1]
        bad_lines = not lines

        if bad_xlim or bad_ylim or bad_lines:
            if bad_xlim:
                CobaContext.logger.log("The xlim end is less than the xlim start. Plotting is impossible.")
            if bad_ylim:
                CobaContext.logger.log("The ylim end is less than the ylim start. Plotting is impossible.")
            if bad_lines:
                CobaContext.logger.log(f"No data was found for plotting.")
        else:

            if not ax or isinstance(ax,int):
                if 'coba' in plt.get_figlabels():
                    f = plt.figure(num="coba")
                    ax = f.add_subplot(111) if not isinstance(ax,int) else f.add_subplot(ax)
                else:
                    ax = plt.subplot(111) if not isinstance(ax,int) else plt.subplot(ax)

            in_lim = lambda v,lim: lim is None or ((lim[0] or -float('inf')) <= v and v <= (lim[1] or float('inf')))

            any_label = False
            num_coalesce = lambda x1,x2: x1 if isinstance(x1,(int,float)) else x2

            artists = list(ax.get_legend_handles_labels()[0])
            xindexes = None

            is_ascending  = lambda _xorder: all(x0<=x1 for x0,x1 in zip(_xorder,_xorder[1:]))
            is_descending = lambda _xorder: all(x0>=x1 for x0,x1 in zip(_xorder,_xorder[1:]))

            if not xorder:
                xindexes = None
            elif isinstance(xorder[0],(int,float)) and is_ascending(xorder):
                xindexes = None
            elif isinstance(xorder[0],(int,float)) and is_descending(xorder):
                xindexes = KeyDefaultDict(lambda k: -k)
            else:
                nextindex = lambda c=count(len(set(xorder))): next(c)
                initindex = zip(reversed(xorder),count(len(xorder)-1,-1))
                xindexes  = collections.defaultdict(nextindex,initindex)

            for X, Y, XE, YE, c, a, l, fmt,z in map(astuple,lines):

                if l: any_label = True

                # we remove values outside of the given lims because matplotlib won't correctly scale otherwise
                if not YE:
                    XY = [(x,y) for i,x,y in zip(count(),X,Y) if in_lim(num_coalesce(x,i),xlim) and in_lim(num_coalesce(y,i),ylim)]
                    X,Y = map(list,zip(*XY)) if XY else ([],[])
                else:
                    XYE = [(x,y,e) for i,x,y,e in zip(count(),X,Y,YE) if in_lim(num_coalesce(x,i),xlim) and in_lim(num_coalesce(y,i),ylim)]
                    X,Y,YE = map(list,zip(*XYE)) if XYE else ([],[],[])

                if isinstance(c,int): c = color_cycle[c%len(color_cycle)]

                not_err_bar = lambda E: not E or all(not e for e in E)

                if X is not None and Y is not None:

                    if xindexes is not None: X = [xindexes[x] for x in X]

                    if all(map(not_err_bar,[XE,YE])):
                        artists.append(ax.plot(X, Y, fmt,  color=c, alpha=a, label=l, zorder=z)[0])
                    else:
                        XE = None if not_err_bar(XE) else list(zip(*XE)) if isinstance(XE[0],tuple) else XE
                        YE = None if not_err_bar(YE) else list(zip(*YE)) if isinstance(YE[0],tuple) else YE
                        errorevery = 1 if fmt == "-" else 1
                        elinewidth = 0.5 if 'elinewidth' not in CobaContext.store else CobaContext.store['elinewidth']
                        artists.append(ax.errorbar(X, Y, YE, XE, fmt, elinewidth=elinewidth, errorevery=errorevery, color=c, alpha=a, label=l, zorder=z))

            if xticks and xrotation is not None:
                plt.xticks(rotation=xrotation)

            if xticks and xindexes:
                labels,ticks = zip(*xindexes.items())
                plt.xticks(ticks,labels)

            if yrotation is not None:
                plt.yticks(rotation=yrotation)

            if xlim[0] is None or xlim[1] is None:
                ax.autoscale(axis='x')

            if ylim[0] is None or ylim[1] is None:
                ax.autoscale(axis='y')

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

            ax.autoscale(axis='both')

            ax.set_title(title, loc='left')
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)

            if ax.get_legend() is not None: #pragma: no cover
                ax.get_legend().remove()
            else: #pragma: no cover
                scale = 0.65
                box1 = ax.get_position()
                ax.set_position([box1.x0, box1.y0 + box1.height * (1-scale), box1.width, box1.height * scale])

            if any_label:
                L = zip(*sorted((zip(*ax.get_legend_handles_labels())), key=lambda al:artists.index(al[0])))
                ax.legend(*L, loc='upper left', bbox_to_anchor=(-.01, -.25), ncol=1, fontsize='medium')

            if not xticks:
                plt.xticks([])

            if not yticks:
                plt.yticks([])

            if out not in ['screen',None]:
                plt.tight_layout()
                plt.savefig(str(out), dpi=300)
                plt.show()
                plt.close()

            if out=="screen":
                plt.show()
                plt.close()

class Result:
    """A class representing the result of an Experiment."""

    @staticmethod
    def from_source(source: Source[Iterable[str]]) -> 'Result':
        """Create a Result from a transaction file."""
        return Pipes.join(source,TransactionDecode(),TransactionResult()).read()

    @staticmethod
    def from_save(filename: Union[str,Source[str]]) -> 'Result':
        """Create a Result from a transaction file."""
        if not Path(filename).exists():
            raise CobaException("We were unable to find the given Result file.")
        return Result.from_source(DiskSource(filename))

    @staticmethod
    def from_file(filename: str) -> 'Result':
        """Create a Result from a transaction file."""
        return Result.from_save(filename)

    @staticmethod
    def from_logged_envs(environments: Iterable[Environment], include_prob:bool=False):

        seen_env = set()
        seen_lrn = set()
        seen_val = set()

        env_table = Table(columns=['environment_id'                                             ])
        lrn_table = Table(columns=[                 'learner_id'                                ])
        val_table = Table(columns=[                              'evaluator_id'                 ])
        int_table = Table(columns=['environment_id','learner_id','evaluator_id','index','reward'])

        def determine_id(param,param_list,param_dict):
            try:
                key = frozenset(param.items())
                if key not in param_dict:
                    param_dict[key] = len(param_list)
                    param_list.append(param)
                return param_dict[key]
            except:
                try:
                    return param_list.index(param)
                except:
                    param_list.append(param)
                    return len(param_list)-1

        def determine_env_id(env_param,env_param_list=[],env_param_dict={}):
            return determine_id(env_param,env_param_list,env_param_dict)

        def determine_lrn_id(lrn_param,lrn_param_list=[],lrn_param_dict={}):
            return determine_id(lrn_param,lrn_param_list,lrn_param_dict)

        def determine_val_id(val_param,val_param_list=[],val_param_dict={}):
            return determine_id(val_param,val_param_list,val_param_dict)

        my_mean = lambda x: sum(x)/len(x)

        for env in environments:

            first, interactions = peek_first(env.read())

            if not env.params.get('logged'): continue
            if not interactions: continue

            is_batched = is_batch(first['reward']) or isinstance(first['reward'],(list,tuple))

            env_param = dict(env.params)
            lrn_param = env_param.pop('learner')
            val_param = env_param.pop('evaluator',{'eval_type':'unknown'})

            env_id = determine_env_id(env_param)
            lrn_id = determine_lrn_id(lrn_param)
            val_id = determine_val_id(val_param)

            if env_id not in seen_env:
                seen_env.add(env_id)
                env_table.insert([{'environment_id':env_id, **env_param}])

            if lrn_id not in seen_lrn:
                seen_lrn.add(lrn_id)
                lrn_table.insert([{'learner_id':lrn_id, **lrn_param}])

            if val_id not in seen_val:
                seen_val.add(val_id)
                val_table.insert([{'evaluator_id':val_id, **val_param}])

            keys = first.keys() - {'context', 'actions', 'rewards'}
            if not include_prob: keys -= {'probability'}

            int_cols = [[] for _ in keys]
            for interaction in interactions:
                for k,c in zip(keys,int_cols):
                    c.append(my_mean(interaction[k]) if is_batched else interaction[k])

            int_count = len(int_cols[0])
            int_maps = {
                'environment_id': [env_id]*int_count,
                'learner_id'    : [lrn_id]*int_count,
                'evaluator_id'  : [val_id]*int_count,
                'index'         : list(range(1,int_count+1))
            }
            int_maps.update(zip(keys,int_cols))
            int_table.insert(int_maps)

        return Result(env_table, lrn_table, val_table, int_table, {})

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self,
        env_rows: Union[Sequence,Table,None],
        lrn_rows: Union[Sequence,Table,None],
        val_rows: Union[Sequence,Table,None],
        int_rows: Union[Sequence,Table,None],
        exp_dict: Mapping = {}) -> None:
        ...

    def __init__(self,*args) -> None:
        """Instantiate a Result class.

        This constructor should never be called directly. Instead a Result file should be created
        from an Experiment and the result file should be loaded via Result.from_file(filename).
        """
        if len(args) > 0:
            env_rows,lrn_rows,val_rows,int_rows = args[:4]
        else:
            env_rows,lrn_rows,val_rows,int_rows = tuple([None]*4)

        env_rows = env_rows if env_rows is not None else Table(columns=['environment_id'                                    ])
        lrn_rows = lrn_rows if lrn_rows is not None else Table(columns=[                 'learner_id'                       ])
        val_rows = val_rows if val_rows is not None else Table(columns=[                              'evaluator_id'        ])
        int_rows = int_rows if int_rows is not None else Table(columns=['environment_id','learner_id','evaluator_id','index'])

        self.experiment = args[4] if len(args) == 5 else {}

        self._environments = env_rows if isinstance(env_rows,Table) else Table(columns=env_rows[0]).insert(env_rows[1:])
        self._learners     = lrn_rows if isinstance(lrn_rows,Table) else Table(columns=lrn_rows[0]).insert(lrn_rows[1:])
        self._evaluators   = val_rows if isinstance(val_rows,Table) else Table(columns=val_rows[0]).insert(val_rows[1:])
        self._interactions = int_rows if isinstance(int_rows,Table) else Table(columns=int_rows[0]).insert(int_rows[1:])

        self._environments.index('environment_id'                                    )
        self._learners    .index(                 'learner_id'                       )
        self._evaluators  .index(                              'evaluator_id'        )
        self._interactions.index('environment_id','learner_id','evaluator_id','index')

        self._env_cache = {d['environment_id']:d for d in self._environments.to_dicts()}
        self._lrn_cache = {d['learner_id'    ]:d for d in self._learners    .to_dicts()}
        self._val_cache = {d['evaluator_id'  ]:d for d in self._evaluators  .to_dicts()}

        for value in self._lrn_cache.values():
            lrn_id = value['learner_id']
            family = value.get('family',lrn_id)
            params = [f'{k}={v}' for k,v in value.items() if k and k not in ['family','learner_id'] and v is not None ]
            params = f"({','.join(params)})" if params else ''
            value['full_name'] = f"{lrn_id}. {family}{params}"

        self._plotter = MatplotPlotter()

    @property
    def learners(self) -> Table:
        """The collection of learners used in the Experiment.

        The primary key of this table is learner_id.
        """
        return self._learners

    @property
    def environments(self) -> Table:
        """The collection of environments used in the Experiment.

        The primary key of this table is environment_id.
        """
        return self._environments

    @property
    def evaluators(self) -> Table:
        """The collection of evaluators used in the Experiment.

        The primary key of this table is evaluator_id.
        """
        return self._evaluators

    @property
    def interactions(self) -> Table:
        """The collection of interactions evaluated by evaluators in the Experiment.

        The primary key of this Table is (environment_id, learner_id, evaluator_id, index).
        """
        return self._interactions

    def set_plotter(self, plotter: Plotter) -> None:
        """Manually set the underlying plotting tool. By default matplotlib is used though this can be changed."""
        self._plotter = plotter

    def copy(self) -> 'Result':
        """Create a copy of Result."""
        result_copy = Result()
        result_copy._environments = self.environments.copy()
        result_copy._learners     = self.learners.copy()
        result_copy._evaluators   = self.evaluators.copy()
        result_copy._interactions = self.interactions.copy()
        result_copy.experiment    = self.experiment
        result_copy._env_cache    = self._env_cache
        result_copy._lrn_cache    = self._lrn_cache
        result_copy._val_cache    = self._val_cache
        result_copy._plotter      = self._plotter

        return result_copy

    def filter_best(self,
        l:Union[str, Sequence[str]],
        p:Union[str, Sequence[str]],
        y:str = 'reward',
        n:int = None,
        full_l:Union[str, Sequence[str]]='learner_id',
        full_p:Union[str, Sequence[str]]='environment_id'):

        only_finished = self.filter_fin(l=full_l,p=full_p)

        environments = only_finished.environments
        learners     = only_finished.learners
        evaluators   = only_finished.evaluators
        interactions = only_finished.interactions

        full_id = ['environment_id','learner_id','evaluator_id']
        indexes = list(self._indexed_gs(p,l,full_l,full_id,select=y))

        to_keep, to_drop = [], []
        for _, group in groupby(indexes,key=itemgetter(0)):
            max_val = -float('inf')
            k, d = [], []
            for _, group in groupby(group,key=itemgetter(1)):
                ids,vals = zip(*((g[3], mean(islice(g[4],n))) for g in group))
                mean_val = mean(vals)
                if mean_val < max_val:
                    d.extend(ids)
                else:
                    max_val = mean_val
                    d.extend(k)
                    k = ids
            to_keep.extend(k)
            to_drop.extend(d)

        if to_drop:
            select = self._remove(to_drop)
            interactions = Table(View(interactions._data,select), interactions.columns, interactions.indexes)

        e_keep,l_keep,v_keep = map(set,zip(*to_keep)) if to_keep else ([],[],[])
        if len(e_keep) != len(environments): environments = environments.where(environment_id=e_keep)
        if len(l_keep) != len(learners)    : learners     = learners    .where(learner_id    =l_keep)
        if len(v_keep) != len(evaluators)  : evaluators   = evaluators  .where(evaluator_id  =v_keep)

        return Result(environments, learners, evaluators, interactions, self.experiment)

    def filter_fin(self,
        n: Union[int,Literal['min']] = None,
        l: Union[str, Sequence[str]] = None,
        p: Union[str, Sequence[str]] = None) -> 'Result':
        """Filter the results down to even outcomes so that plotted results will be meaningful.

        Args:
            n: The number of interactions a specific evaluation must have (None indicates no constraint).
            l: The level at which we wish to compare evalation outcomes.
            p: The pairs that must exist across all comparison levels in order to be included.
        """

        result = self._filter_fin(n,l,p)

        if len(result.interactions) == 0:
            CobaContext.logger.log(f"There was no {p} which was finished for every {l}.")

        return result

    def filter_env(self, pred:Callable[[Mapping[str,Any]],bool] = None, **kwargs: Any) -> 'Result':
        """Filter the result to only contain data about specific environments.

        Args:
            pred: A predicate that returns true for learner dictionaries that should be kept.
            **kwargs: key-value pairs to filter on. To see filtering options see Table.filter.
        """

        if len(self.environments) == 0: return self

        environments = self.environments.where(pred, **kwargs)
        learners     = self.learners
        evaluators   = self.evaluators
        interactions = self.interactions

        if len(environments) == len(self.environments):
            return self

        if len(environments) == 0:
            CobaContext.logger.log(f"No environments matched the given filter.")

        interactions = interactions.where(environment_id=set(environments["environment_id"]))
        learners     = learners    .where(learner_id    =set(interactions["learner_id"]))
        evaluators   = evaluators  .where(evaluator_id  =set(interactions["evaluator_id"]))

        return Result(environments,learners,evaluators,interactions,self.experiment)

    def filter_lrn(self, pred:Callable[[Mapping[str,Any]],bool] = None, **kwargs: Any) -> 'Result':
        """Filter the result to only contain data about specific learners.

        Args:
            pred: A predicate that returns true for learner dictionaries that should be kept.
            **kwargs: key-value pairs to filter on. To see filtering options see Table.filter.
        """
        if len(self.learners) == 0: return self

        environments = self.environments
        learners     = self.learners.where(pred, **kwargs)
        evaluators   = self.evaluators
        interactions = self.interactions

        if len(learners) == len(self.learners):
            return self

        if len(learners) == 0:
            CobaContext.logger.log(f"No learners matched the given filter.")

        interactions = interactions.where(learner_id    =set(learners["learner_id"]))
        environments = environments.where(environment_id=set(interactions["environment_id"]))
        evaluators   = evaluators  .where(evaluator_id  =set(interactions["evaluator_id"]))

        return Result(environments,learners,evaluators,interactions)

    def filter_val(self, pred:Callable[[Mapping[str,Any]],bool] = None, **kwargs: Any) -> 'Result':
        """Filter the result to only contain data about specific evaluators.

        Args:
            pred: A predicate that returns true for learner dictionaries that should be kept.
            **kwargs: key-value pairs to filter on. To see filtering options see Table.filter.
        """
        if len(self.learners) == 0: return self

        environments = self.environments
        learners     = self.learners
        evaluators   = self.evaluators.where(pred, **kwargs)
        interactions = self.interactions

        if len(evaluators) == len(self.evaluators):
            return self

        if len(evaluators) == 0:
            CobaContext.logger.log(f"No evaluators matched the given filter.")

        interactions = interactions.where(evaluator_id  =set(evaluators['evaluator_id']))
        environments = environments.where(environment_id=set(interactions["environment_id"]))
        learners     = learners    .where(learner_id    =set(interactions["learner_id"]))

        return Result(environments,learners,evaluators,interactions)

    def filter_int(self, pred:Callable[[Mapping[str,Any]],bool] = None, **kwargs: Any) -> 'Result':
        """Filter the result to only contain data about specific interactions.

        Args:
            pred: A predicate that returns true for learner dictionaries that should be kept.
            **kwargs: key-value pairs to filter on. To see filtering options see Table.filter.
        """
        if len(self.learners) == 0: return self

        environments = self.environments
        learners     = self.learners
        evaluators   = self.evaluators
        interactions = self.interactions.where(pred, **kwargs)

        if len(interactions) == len(self.interactions):
            return self

        if len(interactions) == 0:
            CobaContext.logger.log(f"No interactions matched the given filter.")

        to_keep = list(interactions.groupby(3,select=None))
        env,lrn,val = zip(*to_keep) if to_keep else ([],[],[])

        if len(env) != len(environments): environments = environments.where(environment_id=set(env))
        if len(lrn) != len(learners)    : learners     = learners    .where(learner_id    =set(lrn))
        if len(val) != len(evaluators)  : evaluators   = evaluators  .where(evaluator_id  =set(val))

        return Result(environments,learners,evaluators,interactions)

    def where(self, **kwargs) -> 'Result':

        env_kwargs = {}
        lrn_kwargs = {}
        val_kwargs = {}
        int_kwargs = {}

        for key,arg in kwargs.items():
            if key in self.environments.columns:
                env_kwargs[key] = arg
            elif key in self.learners.columns:
                lrn_kwargs[key] = arg
            elif key in self.evaluators.columns:
                val_kwargs[key] = arg
            elif key in self.interactions.columns:
                int_kwargs[key] = arg
            else:
                raise CobaException("An unrecognized column was provided to `where` for filtering.")

        out = self

        if env_kwargs: out = out.filter_env(**env_kwargs)
        if lrn_kwargs: out = out.filter_lrn(**lrn_kwargs)
        if val_kwargs: out = out.filter_val(**val_kwargs)
        if int_kwargs: out = out.filter_int(**int_kwargs)

        return out

    def raw_learners(self,
        x   : Union[str, Sequence[str]] = 'index',
        y   : str                       = 'reward',
        l   : Union[str, Sequence[str]] = 'full_name',
        p   : Union[str, Sequence[str]] = 'environment_id',
        span: int = None) -> Table:

        data = {}

        plottable = self._plottable(x,y)._finished(x,y,l,p)

        YS = list(plottable._indexed_ys(l,x,p,y=y,span=span))

        for _l, group in groupby(YS,key=itemgetter(0)):

            group = list(group)

            if 'x' not in data:
                data['p'] = list(map(itemgetter(2),group))
                data['x'] = list(map(itemgetter(1),group))
            data[_l] = list(map(itemgetter(3),group))

        return Table(data)

    def raw_contrast(self,
        l1  : Any,
        l2  : Any,
        x   : Union[str, Sequence[str]] = 'environment_id',
        y   : str                       = 'reward',
        l   : Union[str, Sequence[str]] = 'learner_id',
        p   : Union[str, Sequence[str]] = 'environment_id',
        span: int = None) -> Table:

        og_l = (l1,l2)

        list_like=(list,tuple)

        if     isinstance(l,list_like) and not isinstance(l1[0],list_like): l1 = [l1]
        if     isinstance(l,list_like) and not isinstance(l2[0],list_like): l2 = [l2]
        if not isinstance(l,list_like) and not isinstance(l1   ,list_like): l1 = [l1]
        if not isinstance(l,list_like) and not isinstance(l2   ,list_like): l2 = [l2]

        if any(_l1 in l2 for _l1 in l1):
            raise CobaException("A value cannot be in both `l1` and `l2`. Please make a change and run it again.")

        plottable = self._plottable(x,y)
        eid       = 'environment_id'
        lid       = 'learner_id'

        data = collections.defaultdict(list)

        if x != 'index':
            #this implementation is considerably slower but always gives the correct results

            l1_label = self._lrn_cache[l1[0]]['full_name'] if l=='learner_id' else 'l1'
            l2_label = self._lrn_cache[l2[0]]['full_name'] if l=='learner_id' else 'l2'

            L1,L2 = [],[]
            for _l, group in groupby(plottable._indexed_ys(l,eid,lid,x,y=y,span=span),key=itemgetter(0)):

                if _l in l1:
                    L1.extend(map(itemgetter(slice(1,None)),group))
                if _l in l2:
                    L2.extend(map(itemgetter(slice(1,None)),group))

            for _x, group in groupby(sorted(plottable._pairings(p,L1,L2),key=MyComparable),key=itemgetter(0)):
                _x = _x[0] if _x[0] == _x[1] else f"{_x[1]}-{_x[0]}"

                for _,(_y1,_y2),_p in group:
                    data['p'].append(_p)
                    data['x'].append(_x)
                    data[l1_label].append(_y1)
                    data[l2_label].append(_y2)
        else:
            #this implementation is considerably faster but only gives correct results under certain conditions
            for _x, _group in groupby(plottable._indexed_ys(x,l,eid,lid,x,y=y,span=span),key=itemgetter(0)):

                _group = list(map(itemgetter(slice(1,None)),_group))
                _L1    = [g[1:] for g in _group if g[0] in l1 ]
                _L2    = [g[1:] for g in _group if g[0] in l2 ]

                for _,(_y1,_y2),_p in plottable._pairings(p,_L1,_L2):
                    data['p'].append(_p)
                    data['x'].append(_x)
                    data['l1'].append(_y1)
                    data['l2'].append(_y2)

        if not data:
            raise CobaException(f"We were unable to create any pairings to contrast. Make sure l1={og_l[0]} and l2={og_l[1]} is correct.")

        return Table(data)

    def plot_learners(self,
        x       : Union[str, Sequence[str]] = 'index',
        y       : str                       = 'reward',
        l       : Union[str, Sequence[str]] = 'full_name',
        p       : Union[str, Sequence[str]] = 'environment_id',
        span    : int = None,
        err     : Union[Literal['se','sd','bs','bi'], None, PointAndInterval] = None,
        errevery: int = None,
        labels  : Sequence[str] = None,
        colors  : Union[int,Sequence[Union[str,int]]] = None,
        title   : str = None,
        xlabel  : str = None,
        ylabel  : str = None,
        xlim    : Tuple[Optional[Number],Optional[Number]] = None,
        ylim    : Tuple[Optional[Number],Optional[Number]] = None,
        xticks  : bool = True,
        yticks  : bool = True,
        xorder  : Literal['+','-'] = None,
        top_n   : int = None,
        out     : Union[None,Literal['screen'],str] = 'screen',
        ax = None) -> None:
        """Plot the performance of multiple learners on multiple environments. It gives a sense of the expected
        performance for different learners across independent environments. This plot is valuable in gaining
        insight into how various learners perform in comparison to one another.

        Args:
            x: The values to plot on the x-axis.
            y: The value to plot on the y-axis.
            l: The values to plot in the legend.
            p: The pairs that must exist across all items in the legend in order to be included.
            span: The number of y values to smooth together when reporting y. If this is None then the average of all y
                values up to current is shown otherwise a moving average with window size of span (the window will be
                smaller than span initially).
            err: This determines what kind of error bars to plot (if any). If `None` then no bars are plotted, if 'se'
                the standard error is shown, and if 'sd' the standard deviation is shown.
            errevery: This determines the frequency of errorbars. If `None` they appear 5% of the time.
            labels: The legend labels to use in the plot. These should be in order of the actual legend labels.
            colors: The colors used to plot the learners plot.
            title : The title give the plot.
            xlabel: The label on the x-axis.
            ylabel: The label on the y-axis.
            xlim: Define the x-axis limits to plot. If `None` the x-axis limits will be inferred.
            ylim: Define the y-axis limits to plot. If `None` the y-axis limits will be inferred.
            xticks: Whether the x-axis labels should be drawn.
            yticks: Whether the y-axis labels should be drawn.
            xorder: Indicates whether the x-axis should be in ascending (+) or descendeing (-) order.
            top_n: Only plot the top_n learners. If `None` all learners will be plotted. If negative the bottom will be plotted.
            out: Indicate where the plot should be sent to after plotting is finished.
            ax: Provide an optional axes that the plot will be drawn to. If not provided a new figure/axes is created.
        """
        try:
            xlim = xlim or [None,None]
            ylim = ylim or [None,None]

            if isinstance(labels,str): labels = [labels]

            raw_data = self.raw_learners(x,y,l,p,span)

            errevery = errevery or max(int(raw_data['x'][-1]*0.05),1) if x == 'index' else 1
            style    = "-" if x == 'index' else "."
            err      = self._confidence(err, errevery)

            lines: List[Points] = []
            for _l in raw_data.columns[2:]:
                color = self._get_color(colors,   len(lines))
                label = self._get_label(labels,_l,len(lines))

                lines.append(Points(style=style,color=color,label=label))
                for _xi, (_x, group) in enumerate(groupby(zip(*raw_data[['x',_l]]), key=itemgetter(0))):
                    Y = [g[-1] for g in group]
                    lines[-1].add(_x, *err(Y, _xi))

            lines  = sorted(lines, key=lambda line: line.Y[-1], reverse=True)
            labels = [l.label or str(l.label) for l in lines]
            colors = [l.color                 for l in lines]
            xlabel = xlabel or ("Interaction" if x=='index' else x[0] if len(x) == 1 else x)
            ylabel = ylabel or (y.capitalize().replace("_pct"," Percent"))

            y_location = "Total" if x != 'index' else ""
            y_avg_type = ("Instant" if span == 1 else f"Span {span}" if span else "Progressive")
            y_samples  = f"({len(Y)} Environments)"
            title      = title if title is not None else (' '.join(filter(None,[y_location, y_avg_type, ylabel, y_samples])))

            xrotation = 90 if (x != 'index' or xorder) and len(lines[0].X)>5 else 0
            yrotation = 0

            if top_n:
                if abs(top_n) > len(lines): top_n = len(lines)*abs(top_n)/top_n
                if top_n > 0: lines = [replace(l,color=self._get_color(colors,i),label=self._get_label(labels,l.label,i)) for i,l in enumerate(lines[:top_n],0    ) ]
                if top_n < 0: lines = [replace(l,color=self._get_color(colors,i),label=self._get_label(labels,l.label,i)) for i,l in enumerate(lines[top_n:],top_n) ]

            all_x = dict.fromkeys(chain.from_iterable(line.X for line in lines))
            xordered = list(all_x.keys()) if not xorder else sorted(all_x, reverse=xorder=='-')

            if x == 'index' and xorder in ['+',None]: xordered = None

            self._plotter.plot(ax, lines, title, xlabel, ylabel, xlim, ylim, xticks, yticks, xrotation, yrotation, xordered, out)

        except CobaException as e:
            CobaContext.logger.log(str(e))

    def plot_contrast(self,
        l1      : Any,
        l2      : Any,
        x       : Union[str, Sequence[str]] = "environment_id",
        y       : str = "reward",
        l       : Union[str, Sequence[str]] = 'learner_id',
        p       : Union[str, Sequence[str]] = 'environment_id',
        mode    : Union[Literal["diff","prob"], Callable[[float,float],float]] = "diff",
        span    : int = None,
        err     : Union[Literal['se','sd','bs','bi'], None, PointAndInterval] = None,
        errevery: int = None,
        labels  : Sequence[str] = None,
        colors  : Sequence[str] = None,
        title   : str = None,
        xlabel  : str = None,
        ylabel  : str = None,
        xlim    : Tuple[Optional[Number],Optional[Number]] = None,
        ylim    : Tuple[Optional[Number],Optional[Number]] = None,
        xticks  : bool = True,
        yticks  : bool = True,
        xorder  : Literal['+','-'] = None,
        boundary: bool = True,
        legend  : bool = True,
        out     : Union[None,Literal['screen'],str] = 'screen',
        ax = None) -> None:
        """Plot a direct contrast of the performance for two learners.

        Args:
            l1: The first set of parameter values we want to contrast.
            l2: The second set of parameter values we want to contrast.
            x: The value to plot on the x-axis. This can either be index or environment columns to group by.
            y: The value to plot on the y-axis.
            l: The level at which we want to contrast.
            p: The pairs that must exist across all comparison levels in order to be included.
            mode: 'diff' -- plot the pairwise difference; 'prob' plot the probability of l1 beating l2.
            span: The number of y values to smooth together when reporting y. If this is None then the average of all y
                values up to current is shown otherwise a moving average with window size of span (the window will be
                smaller than span initially).
            err: This determines what kind of error bars to plot (if any). If `None` then no bars are plotted, if 'se'
                the standard error is shown, and if 'sd' the standard deviation is shown.
            errevery: This determines the frequency of errorbars. If `None` they appear 5% of the time.
            labels: The legend labels to use in the plot. These should be in order of the actual legend labels.
            title : The title give the plot.
            colors: The colors used to plot the learners plot.
            xlabel: The label on the x-axis.
            ylabel: The label on the y-axis.
            xlim: Define the x-axis limits to plot. If `None` the x-axis limits will be inferred.
            ylim: Define the y-axis limits to plot. If `None` the y-axis limits will be inferred.
            xticks: Whether the x-axis labels should be drawn.
            yticks: Whether the y-axis labels should be drawn.
            xorder: Indicates whether the x-axis should be in ascending (+) or descendeing (-) order.
            boundary: Whether we want to plot the boundary line between which set is the best performing.
            out: Indicate where the plot should be sent to after plotting is finished.
            ax: Provide an optional axes that the plot will be drawn to. If not provided a new figure/axes is created.
        """

        try:
            xlim = xlim or [None,None]
            ylim = ylim or [None,None]

            raw_data = self.raw_contrast(l1,l2,x,y,l,p,span)

            list_like=(list,tuple)

            if     isinstance(l,list_like) and not isinstance(l1[0],list_like): l1 = [l1]
            if     isinstance(l,list_like) and not isinstance(l2[0],list_like): l2 = [l2]
            if not isinstance(l,list_like) and not isinstance(l1   ,list_like): l1 = [l1]
            if not isinstance(l,list_like) and not isinstance(l2   ,list_like): l2 = [l2]

            contraster = (lambda x,y: y-x) if mode == 'diff' else (lambda x,y: int(y-x>0)) if mode=='prob' else mode
            _boundary  = 0 if mode == 'diff' else .5

            errevery = errevery or max(int(raw_data['x'][-1]*0.05),1) if x == 'index' else 1
            style    = "-" if x == 'index' else "."
            err      = self._confidence(err, errevery)

            X_Y_YE = []
            for _xi, (_x, group) in enumerate(groupby(zip(*raw_data[raw_data.columns[-3:]]), key=itemgetter(0))):
                _Y = [ contraster(g[1],g[2]) for g in group ]
                if _Y: X_Y_YE.append((_x,)+err(_Y,_xi))

            l1_label = raw_data.columns[-2]
            l2_label = raw_data.columns[-1]

            if x == 'index':
                X,Y,YE = zip(*X_Y_YE)
                color  = self._get_color(colors,                         0)
                label  = self._get_label(labels,f'{l2_label}-{l1_label}',0)
                label  = f"{label}" if legend else None
                lines  = [Points(X, Y, None, YE, style=style, label=label, color=color)]

            elif x == l:
                if len(l1) > 1 and len(l2) == 1:
                    #Sort by l1. We assume _x is f"{l2}-{l1}".
                    l2_len = len(str(l2[0]))
                    l1 = list(map(str,l1))
                    X_Y_YE = sorted(X_Y_YE, key=lambda items: l1.index(items[0][l2_len+1:]))
                elif len(l2) > 1 and len(l1) == 1:
                    #Sort by l2. We assume _x is f"{l2}-{l1}".
                    l1_len = len(str(l1[0]))
                    l2 = list(map(str,l2))
                    X_Y_YE = sorted(X_Y_YE, key=lambda items: l2.index(items[0][:-(l1_len+1)]))
                else:
                    X_Y_YE = sorted(X_Y_YE)

                X,Y,YE = zip(*X_Y_YE)
                color  = self._get_color(colors, 0)
                lines  = [Points(X, Y, None, YE, style=style, label=None, color=color)]

            else:
                upper = lambda y,ye: y+ye[1] if isinstance(ye,(list,tuple)) else y+ye
                lower = lambda y,ye: y-ye[0] if isinstance(ye,(list,tuple)) else y-ye
                prep  = lambda _x: str(_x) if not xorder else _x

                #split into win,tie,loss
                l1_win = [(prep(x),y,ye) for x,y,ye in X_Y_YE if upper(y,ye) <  _boundary                             ]
                no_win = [(prep(x),y,ye) for x,y,ye in X_Y_YE if lower(y,ye) <= _boundary and _boundary <= upper(y,ye)]
                l2_win = [(prep(x),y,ye) for x,y,ye in X_Y_YE if                              _boundary <  lower(y,ye)]

                #sort by order of magnitude
                l1_win = sorted(l1_win,key=itemgetter(1))
                no_win = sorted(no_win,key=itemgetter(1))
                l2_win = sorted(l2_win,key=itemgetter(1))

                lines = []

                X,Y,YE = zip(*l1_win) if l1_win else ((),(),None)
                color  = self._get_color(colors,         0)
                label  = self._get_label(labels,l1_label,0)
                label  = f"{label} ({len(X)})" if legend else None
                lines.append(Points(X, Y, None, YE, style=style, label=label, color=color))

                X,Y,YE = zip(*no_win) if no_win else ((),(),None)
                color  = self._get_color(colors, 1)
                label  = 'Tie'
                label  = f"{label} ({len(X)})" if legend else None
                lines.append(Points(X, Y, None, YE, style=style, label=label, color=color))

                X,Y,YE = zip(*l2_win) if l2_win else ((),(),None)
                color  = self._get_color(colors,         2)
                label  = self._get_label(labels,l2_label,1)
                label  = f"{label} ({len(X)})" if legend else None
                lines.append(Points(X, Y, None, YE, style=style, label=label, color=color))

            xrotation = 90 if (x != 'index' or xorder) and len(X_Y_YE)>5 else 0
            yrotation = 0

            xlabel = xlabel or ("Interaction" if x=='index' else x[0] if len(x) == 1 else x)
            ylabel = ylabel or (f"$\Delta$ {y}" if mode=="diff" else f"P($\Delta$ {y} > 0)")
            title  = title if title is not None else (f"{ylabel} ({len(_Y)} Environments)")

            all_x = dict.fromkeys(chain.from_iterable(line.X for line in lines))
            xordered = list(all_x.keys()) if not xorder else sorted(all_x, reverse=xorder=='-')

            if boundary:
                lines.append(Points((xordered[0],xordered[-1]),(_boundary,_boundary), None, None , "#888", 1, None, '-',.5))

            if x == 'index' and xorder in ['+',None]:
                xordered = None

            self._plotter.plot(ax, lines, title, xlabel, ylabel, xlim, ylim, xticks, yticks, xrotation, yrotation, xordered, out)

        except CobaException as e:
            CobaContext.logger.log(str(e))

    def __str__(self) -> str:
        return str({"Learners": len(self._learners), "Environments": len(self._environments), "Interactions": len(self._interactions) })

    def __eq__(self, o: object) -> bool:
        return isinstance(o,Result) \
           and o.environments == self.environments \
           and o.learners == self.learners \
           and o.evaluators == self.evaluators

    def _ipython_display_(self):
        #pretty print in jupyter notebook (https://ipython.readthedocs.io/en/stable/config/integrating.html)
        print(str(self))

    def _get_color(self, colors:Union[None,Sequence[Union[str,int]]], i:int) -> Union[str,int]:
        try:
            return colors[i] if colors else i
        except IndexError:
            return i+max(colors) if isinstance(colors[0],(int,float)) else i
        except TypeError:
            return i+colors

    def _get_label(self, labels:Sequence[str], label:str, i:int) -> str:
        try:
            label = labels[i] if labels else label
        except:
            pass

        return 'None' if label is None else label

    def _pairings(self,
        p:Sequence[str],
        L1:Sequence[Tuple[int,int,float]],
        L2:Sequence[Tuple[int,int,float]]) -> Iterable[Tuple[float,float]]:

        unpack_p = isinstance(p,str)
        if isinstance(p,str): p = [p]

        env_eq_cols = list(set(p) & set(self.environments.columns))
        lrn_eq_cols = list(set(p) & set(self.learners.columns))

        env_eq_vals = { row[0]:row[1:] for row in zip(*self.environments[['environment_id']+env_eq_cols]) }
        lrn_eq_vals = { row[0]:row[1:] for row in zip(*self.learners    [['learner_id'    ]+lrn_eq_cols]) }

        #could this be made faster? I could think of special cases but not a general solution to speed it up.
        for e1,l1,x1,y1 in L1:
            for e2,l2,x2,y2 in L2:
                p1 = env_eq_vals[e1]+lrn_eq_vals[l1]
                p2 = env_eq_vals[e2]+lrn_eq_vals[l2]
                if p1==p2:
                    p = p1[0] if unpack_p else p1
                    yield ((x1,x2),(y1,y2), p)

    def _plottable(self, x:Sequence[str], y:str) -> 'Result':

        if not isinstance(x,str) and 'index' in x and len(x) > 1:
            raise CobaException('The x-axis cannot contain both indexes and parameters.')

        if len(self.interactions) == 0:
            raise CobaException("This result does not contain any data to plot.")

        if y not in self.interactions.columns:
            raise CobaException(f"This result does not contain column '{y}' in interactions.")

        return self

    def _finished(self, x:Sequence[str], y:str, l:Sequence[str], p:Sequence[str]) -> 'Result':
        only_finished = self._filter_fin('min' if x == 'index' else None, l, p)

        if len(only_finished.learners) == 0:
            raise CobaException(f"This result does not contain a {p} that has been finished for every {l}.")

        return only_finished

    def _confidence(self, err: Union[str,PointAndInterval], errevery:int = 1):

        if err == 'se':
            ci = StdErrCI(1.96)
        elif err == 'bs':
            ci = BootstrapCI(.95, mean)
        elif err == 'bi':
            ci = BinomialCI('wilson')
        elif err == 'sd':
            ci = StdDevCI()
        elif err is None or isinstance(err,str):
            ci = None
        else:
            ci = err

        def calc_ci(Z:Sequence[float],i:int = -1):
            if ci is None:
                return (mean(Z),0)
            else:
                skip_err = (i+1)%errevery
                return (ci.point(Z), (0,0)) if skip_err else ci.point_interval(Z)

        return calc_ci

    def _indexed_gs(self,*indexes, select:Union[None,Literal['table','count'],str]=None) -> Iterable[Tuple[Any]]:
        #WARNING: This has been highly highly optimized. Don't make changes without performance testing.

        def make_getter(K):
            if not isinstance(K,str):
                return lambda e,l,v,G=list(map(make_getter,K)): tuple(g(e,l,v) for g in G)
            elif K in self.environments.columns:
                return lambda e,l,v,k=K: e[k]
            elif K in self.learners.columns or K == 'full_name':
                return lambda e,l,v,k=K: l[k]
            elif K in self.evaluators.columns:
                return lambda e,l,v,k=K: v[k]
            else:
                return lambda e,l,v: None

        env_cache = self._env_cache
        lrn_cache = self._lrn_cache
        val_cache = self._val_cache

        def index_getter(env_id, lrn_id, val_id, getters=list(map(make_getter,indexes))):
            return map(methodcaller('__call__',env_cache[env_id],lrn_cache[lrn_id],val_cache[val_id]),getters)

        G = list(self.interactions.groupby(3,select))

        indexed_groups = []
        if select:
            for (env_id,lrn_id,val_id), selected in G:
                indexed_groups.append([*index_getter(env_id,lrn_id,val_id), selected])
        else:
            for env_id,lrn_id,val_id in G:
                indexed_groups.append(list(index_getter(env_id,lrn_id,val_id)))

        return sorted(indexed_groups,key=MyComparable)

    def _indexed_ys(self,*indexes,y,span) -> Iterable[Tuple[Any]]:
        #WARNING: This has been highly highly optimized. Don't make changes without performance testing.

        index_locs = [ i for i, I in enumerate(indexes) if I == 'index']
        index_1st  = index_locs[0] if index_locs else len(indexes)
        statistic  = moving_average if index_locs else (lambda v,_: [mean(v)])

        YS = list(self._indexed_gs(*indexes,select=y))

        for _,t_group in groupby(YS,key=itemgetter(slice(None,index_1st))):
            y_group = []
            for group in t_group:
                i_cols = list(map(repeat,group[:-1]))
                for i in index_locs: i_cols[i] = count(1)
                y_group.append(zip(*i_cols, statistic(group[-1],span)))

            yield from chain.from_iterable(zip(*y_group))

    def _global_n(self, n: Union[int,Literal['min']]):

        environments = self.environments
        learners     = self.learners
        evaluators   = self.evaluators
        interactions = self.interactions

        env_lengths = []
        to_drop     = []
        to_keep     = []
        if n == 'min':
            for env_idx, env_len in self._indexed_gs(['environment_id','learner_id','evaluator_id'],select='count'):
                env_lengths.append(env_len)
        else:
            for env_idx, env_len in self._indexed_gs(['environment_id','learner_id','evaluator_id'],select='count'):
                if env_len < n:
                    to_drop.append(env_idx)
                else:
                    env_lengths.append(env_len)
                    to_keep.append(env_idx)

        if to_drop:
            n_dropped = len(to_drop)
            interactions = Table(View(interactions._data,self._remove(to_drop,n)), interactions.columns, interactions.indexes)
            if n_dropped==1: CobaContext.logger.log(f"We removed {n_dropped} learner evaluation because it was shorter than {n} interactions.")
            if n_dropped>=2: CobaContext.logger.log(f"We removed {n_dropped} learner evaluations because they were shorter than {n} interactions.")

        env_lengths = collections.Counter(env_lengths)
        shorten_to = min(env_lengths) if n=='min' and env_lengths else n
        if len(env_lengths) > 1 or env_lengths.values() != {shorten_to}:
            n_shortened = sum(v for k,v in env_lengths.items() if k > shorten_to)
            interactions = interactions.where(index={'<=':shorten_to}) #.4
            if n_shortened==1: CobaContext.logger.log(f"We shortened {n_shortened} environment because it was longer than the shortest environment.")
            if n_shortened>=2: CobaContext.logger.log(f"We shortened {n_shortened} environments because they were longer than the shortest environment.")

        keep_envs,keep_lrns,keep_vals = map(set,zip(*to_keep)) if to_keep else ([],[],[])

        if to_drop and len(keep_envs) != len(environments):
            environments = environments.where(environment_id=keep_envs)

        if to_drop and len(keep_lrns) != len(learners):
            learners = learners.where(learner_id=keep_lrns)

        if to_drop and len(keep_vals) != len(evaluators):
            evaluators = evaluators.where(evaluator_id=keep_vals)

        return Result(environments, learners, evaluators, interactions, self.experiment)

    def _group_p(self, l:Union[str, Sequence[str]], p:Union[str, Sequence[str]]):

        environments = self.environments
        learners     = self.learners
        evaluators   = self.evaluators
        interactions = self.interactions

        indexes = list(self._indexed_gs(p,l,'environment_id','learner_id','evaluator_id',select=None))
        n_levels = len(set(map(itemgetter(1),indexes)))

        to_keep, to_remove, n_larger, n_smaller = [], [], 0, 0
        for _, group in groupby(indexes,key=itemgetter(0)):
            group = list(group)
            if len(group) > n_levels:
                n_larger += 1
                to_remove.extend(g[2:] for g in group)
            elif len(group) < n_levels:
                n_smaller += 1
                to_remove.extend(g[2:] for g in group)
            else:
                to_keep.extend(g[2:] for g in group)

        if n_larger:
            CobaContext.logger.log(f"We removed {n_larger} {p} because more than one existed for each {l}.")

        if n_smaller:
            CobaContext.logger.log(f"We removed {n_smaller} {p} because {'they' if n_smaller>1 else 'it'} did not exist for every {l}.")

        if to_remove:
            select = self._remove(to_remove)
            interactions = Table(View(interactions._data,select), interactions.columns, interactions.indexes)

        e_keep,l_keep,v_keep = map(set,zip(*to_keep)) if to_keep else ([],[],[])
        if len(e_keep) != len(environments): environments = environments.where(environment_id=e_keep)
        if len(l_keep) != len(learners)    : learners     = learners    .where(learner_id    =l_keep)
        if len(v_keep) != len(evaluators)  : evaluators   = evaluators  .where(evaluator_id  =v_keep)

        return Result(environments, learners, evaluators, interactions, self.experiment)

    def _filter_fin(self,
        n: Union[int,Literal['min'], None],
        l: Union[str, Sequence[str], None],
        p: Union[str, Sequence[str], None]) -> 'Result':
        """Filter the results down to even outcomes so that plotted results will be meaningful.

        Args:
            n: The number of interactions a specific evaluation must have (None indicates no constraint).
            l: The level at which we wish to compare evalation outcomes.
            p: The pairs that must exist across all comparison levels in order to be included.
        """

        result = self.copy()

        if l or p: result = result._group_p(l,p)
        if n     : result = result._global_n(n)

        return result

    def _remove(self, ids: Sequence[Tuple[int,int,int]], n=0) -> Sequence[int]:
        #this is much faster than any built in Table methods
        #to work interactions must be sorted by env_id,lrn_id,val_id
        loc            = 0
        select         = []
        n_interactions = len(self.interactions)
        env_ids,lrn_ids,val_ids = self.interactions[['environment_id','learner_id','evaluator_id']]
        ids = sorted(ids)
        for i in range(len(ids)):
            e,l,v = ids[i]
            lo1 = my_bisect_left (env_ids,e,loc,n_interactions)
            hi1 = my_bisect_right(env_ids,e,loc,n_interactions)
            if lo1==hi1: continue #the given ids aren't in interactions
            lo2 = my_bisect_left (lrn_ids,l,lo1,hi1)
            hi2 = my_bisect_right(lrn_ids,l,lo1,hi1)
            if lo2==hi2: continue
            lo3 = my_bisect_left (val_ids,v,lo2,hi2)
            hi3 = my_bisect_right(val_ids,v,lo2,hi2)
            if lo3==hi3: continue
            select.extend(range(loc,lo3+(n if hi3-lo3>n else 0)))
            loc = hi3

        select.extend(range(loc,n_interactions))
        return select
