import re
import collections
import collections.abc

from copy import copy
from pathlib import Path
from numbers import Number
from operator import truediv, sub, gt
from abc import abstractmethod
from itertools import chain, repeat, accumulate, groupby, count
from typing import Any, Dict, List, Set, Tuple, Optional, Sequence, Hashable, Iterable, Iterator, Union, Type, Callable, NamedTuple
from coba.backports import Literal

from coba.statistics import Mean, StandardDeviation, StandardErrorOfMean, BootstrapConfidenceInterval, BinomialConfidenceInterval, PointAndInterval
from coba.contexts import CobaContext
from coba.exceptions import CobaException
from coba.utilities import PackageChecker
from coba.pipes import Pipes, Sink, Source, JsonEncode, JsonDecode, DiskSource, DiskSink, IterableSource, ListSink, Foreach

def exponential_moving_average(values:Sequence[float], span:int=None) -> Iterable[float]:
    #exponential moving average identical to Pandas df.ewm(span=span).mean()
    alpha = 2/(1+span)
    cumwindow  = list(accumulate(values          , lambda a,v: v + (1-alpha)*a))
    cumdivisor = list(accumulate([1.]*len(values), lambda a,v: v + (1-alpha)*a)) #type: ignore
    return map(truediv, cumwindow, cumdivisor)

def moving_average(values:Sequence[float], span:int=None) -> Iterable[float]:

    if span == 1: 
        return values

    if span is None or span >= len(values):
        return (a/n for n,a in enumerate(accumulate(values),1))

    window_sums  = accumulate(map(sub, values, chain(repeat(0,span),values)))
    window_sizes = chain(range(1,span), repeat(span))

    return map(truediv,window_sums,window_sizes)

class Table:
    """A container class for storing tabular data."""

    def __init__(self, name:str, primary_cols: Sequence[str], rows: Sequence[Dict[str,Any]], preferred_cols: Sequence[str] = []):
        """Instantiate a Table.

        Args:
            name: The name of the table. Used for display purposes.
            primary_cols: Table columns used to make each row's tuple "key".
            rows: The actual rows that should be stored in the table. Each row is required to contain the given primary_cols.
            preferred_cols: A list of columns that we prefer be displayed immediately after primary columns. All remaining
                columns (i.e., neither primary nor preferred) will be ordered alphabetically.
        """
        self._name    = name
        self._primary = primary_cols

        for row in rows:
            assert len(row.keys() & primary_cols) == len(primary_cols), 'A Table row was provided without a primary key.'

        all_columns: Set[str] = set()
        for row in rows:
            all_columns |= {'index'} if '_packed' in row else set()
            all_columns |= row.keys()-{'_packed'}
            all_columns |= all_columns.union(row.get('_packed',{}).keys())

        col_priority = list(chain(primary_cols + ['index'] + preferred_cols + sorted(all_columns)))

        self._columns = sorted(all_columns, key=col_priority.index)
        self._rows_keys: Dict[Hashable, None         ] = {}
        self._rows_flat: Dict[Hashable, Dict[str,Any]] = {}
        self._rows_pack: Dict[Hashable, Dict[str,Any]] = {}

        for row in rows:
            row_key  = row[primary_cols[0]] if len(primary_cols) == 1 else tuple(row[col] for col in primary_cols)
            row_pack = row.pop('_packed',{})
            row_flat = row

            if row_pack:
                row_pack['index'] = list(range(1,len(list(row_pack.values())[0])+1))

            self._rows_keys[row_key] = None
            self._rows_pack[row_key] = row_pack
            self._rows_flat[row_key] = row_flat

        self._rows_keys = collections.OrderedDict(zip(sorted(list(self._rows_keys.keys())),repeat(None)))

    @property
    def name(self) -> str:
        """The name of the table."""
        return self._name

    @property
    def keys(self) -> Sequence[Hashable]:
        """Keys for accessing data in the table."""
        return list(self._rows_keys.keys())

    @property
    def columns(self) -> Sequence[str]:
        """The columns in the table."""
        return self._columns

    @property
    def dtypes(self) -> Sequence[Type[Union[int,float,bool,object]]]:
        """The dtypes for the columns in the table."""
        flats = self._rows_flat
        packs = self._rows_pack

        columns_packed = [ any([ col in packs[key] for key in self.keys]) for col in self.columns ]
        columns_values = [ [flats[key].get(col, packs[key].get(col, self._default(col))) for key in self.keys] for col in self.columns ]

        return [ self._infer_type(column_packed, column_values) for column_packed, column_values in zip(columns_packed,columns_values)]

    def filter(self, row_pred:Callable[[Dict[str,Any]],bool] = None, **kwargs) -> 'Table':
        """Filter to specific rows.

        Args:
            pred: A predicate that returns true for row dictionaries that should be kept.
            kwargs: key value pairs where the key is the column and the value indicates what
                value a row should have in that column to be kept. Keeping logic depends on
                the row value type and the kwargs value type. If kwarg value == row value keep
                the row. If kwarg value is callable pass the row value to the predicate. If
                kwarg value is a collection keep the row if the row value is in the collection.
                If kwarg value is a string apply a regular expression match to the row value.
        """

        def satisifies_filter(col_filter,col_value):
            if col_filter == col_value:
                return True

            if isinstance(col_filter,Number) and isinstance(col_value,str):
                return re.search(f'(\D|^){col_filter}(\D|$)', col_value)

            if isinstance(col_filter,str) and isinstance(col_value,str):
                return re.search(col_filter, col_value)

            if callable(col_filter):
                return col_filter(col_value)

            return False

        def satisfies_all_filters(key):
            row = self[key]

            row_filter_results = [ row_pred is None or row_pred(row) ]
            col_filter_results = [ ]

            for col,col_filter in kwargs.items():

                if isinstance(col_filter,collections.abc.Container) and not isinstance(col_filter,str):
                    col_filter_results.append(row[col] in col_filter or any([satisifies_filter(cf,row[col]) for cf in col_filter]))
                else:
                    col_filter_results.append(satisifies_filter(col_filter,row.get(col,self._default(col)) ))

            return all(row_filter_results+col_filter_results)

        new_result = copy(self)
        new_result._rows_keys = collections.OrderedDict(zip(filter(satisfies_all_filters,self.keys), repeat(None)))

        return new_result

    def to_pandas(self) -> Any:
        """Turn the Table into a Pandas data frame."""

        PackageChecker.pandas("Table.to_pandas")
        import pandas as pd #type: ignore
        import numpy as np  #type: ignore #pandas installs numpy so if we have pandas we have numpy

        col_numpy = { col: np.empty(len(self), dtype=dtype) for col,dtype in zip(self.columns,self.dtypes)}

        row_index = 0

        for key in self.keys:

            flat = self._rows_flat[key]
            pack = self._rows_pack[key]

            pack_size = 1 if not pack else len(pack['index'])

            for col in self.columns:
                if col in pack:
                    val = pack[col]

                elif col in flat:
                    if isinstance(flat[col], (tuple,list)):
                        val = [flat[col]]
                    else:
                        val = flat[col]

                else:
                    val = self._default(col)

                col_numpy[col][row_index:(row_index+pack_size)] = val

            row_index += pack_size

        return pd.DataFrame(col_numpy, columns=self.columns)

    def to_dicts(self) -> Sequence[Dict[str,Any]]:
        """Turn the Table into a sequence of tuples."""

        dicts = []

        for key in self.keys:

            flat = self._rows_flat[key]
            pack = self._rows_pack[key]

            if not pack:
                dicts.append( { col:flat.get(col,self._default(col)) for col in self.columns } )
            else:
                tuples = list(zip(*[pack.get(col,repeat(flat.get(col,self._default(col)))) for col in self.columns]))
                dicts.extend([ dict(zip(self.columns,t)) for t in tuples ])

        return dicts

    def _default(self, column:str) -> Any:
        return [1] if column == "index" else None

    def _infer_type(self, is_packed: bool, values: Sequence[Any]) -> Type[Union[int,float,bool,object]]:

        types: List[Optional[Type[Any]]] = []

        to_type = lambda value: None if value is None else type(value)

        for value in values:
            if is_packed and isinstance(value, (list,tuple)):
                types.extend([to_type(v) for v in value])
            else:
                types.append(to_type(value))

        return self._resolve_types(types)

    def _resolve_types(self, types: Sequence[Optional[Type[Any]]]) -> Type[Union[int,float,bool,object]]:
        types = list(set(types))

        if len(types) == 1 and types[0] in [dict,str]:
            return object

        if len(types) == 1 and types[0] in [int,float,bool]:
            return types[0]

        if all(t in [None,int,float] for t in types):
            return float

        return object

    def __iter__(self) -> Iterator[Dict[str,Any]]:
        for key in self.keys:
            yield self[key]

    def __contains__(self, key: Union[Hashable, Sequence[Hashable]]) -> bool:
        return key in self.keys

    def __str__(self) -> str:
        return str({"Table": self.name, "Columns": self.columns, "Rows": len(self)})

    def _ipython_display_(self):
        #pretty print in jupyter notebook (https://ipython.readthedocs.io/en/stable/config/integrating.html)
        print(str(self))

    def __len__(self) -> int:
        return sum([ len(self._rows_pack[key].get('index',[None])) for key in self.keys ])

    def __getitem__(self, key: Union[Hashable, Sequence[Hashable]]) -> Dict[str,Any]:
        if key not in self._rows_keys: raise KeyError(key)
        return dict(**self._rows_flat[key], **self._rows_pack[key])

class InteractionsTable(Table):
    pass

class TransactionIO_V3(Source['Result'], Sink[Any]):

    def __init__(self, log_file: Optional[str] = None, minify:bool = True) -> None:

        self._log_file = log_file
        self._minify   = minify
        self._source   = DiskSource(log_file) if log_file else IterableSource()
        self._sink     = DiskSink(log_file) if log_file else ListSink(self._source.iterable)

    def write(self, item: Any) -> None:
        if isinstance(self._sink, ListSink):
            self._sink.write(self._encode(item))
        else:
            if not Path(self._sink._filename).exists():self._sink.write('["version",3]')
            self._sink.write(JsonEncode(self._minify).filter(self._encode(item)))

    def read(self) -> 'Result':
        n_lrns   = None
        n_envs   = None
        lrn_rows = {}
        sim_rows = {}
        int_rows = {}

        if isinstance(self._source, IterableSource):
            decoded_source = self._source
        else:
            decoded_source = Pipes.join(self._source, Foreach(JsonDecode()))

        for trx in decoded_source.read():

            if not trx: continue

            if trx[0] == "benchmark":
                n_lrns = trx[1]["n_learners"]
                n_envs = trx[1]["n_simulations"]

            if trx[0] == "S":
                sim_rows[trx[1]] = trx[2]

            if trx[0] == "L":
                lrn_rows[trx[1]] = trx[2]

            if trx[0] == "I":
                int_rows[tuple(trx[1])] = trx[2]

        return Result(sim_rows, lrn_rows, int_rows, {"n_learners":n_lrns,"n_environments":n_envs})

    def _encode(self,item):
        if item[0] == "T0":
            return ['benchmark', {"n_learners":item[1], "n_simulations":item[2]}]

        if item[0] == "T1":
            return ["L", item[1], item[2]]

        if item[0] == "T2":
            return ["S", item[1], item[2]]

        if item[0] == "T3":
            rows_T = collections.defaultdict(list)

            for row in item[2]:
                for col,val in row.items():
                    if col == "rewards" : col="reward"
                    if col == "reveals" : col="reveal"
                    rows_T[col].append(val)

            return ["I", item[1], { "_packed": rows_T }]

        return None

class TransactionIO_V4(Source['Result'], Sink[Any]):

    def __init__(self, log_file: Optional[str] = None, minify:bool=True) -> None:

        self._log_file = log_file
        self._minify   = minify
        self._source   = DiskSource(log_file) if log_file else IterableSource()
        self._sink     = DiskSink(log_file)   if log_file else ListSink(self._source.iterable)

    def write(self, item: Any) -> None:
        if isinstance(self._sink, ListSink):
            self._sink.write(self._encode(item))
        else:
            if not Path(self._sink._filename).exists():self._sink.write('["version",4]')
            self._sink.write(JsonEncode(self._minify).filter(self._encode(item)))

    def read(self) -> 'Result':

        exp_dict = {}
        lrn_rows = {}
        env_rows = {}
        int_rows = {}

        if isinstance(self._source, IterableSource):
            decoded_source = self._source
        else:
            decoded_source = Pipes.join(self._source, Foreach(JsonDecode()))

        for trx in decoded_source.read():

            if not trx: continue

            if trx[0] == "experiment":
                exp_dict = trx[1]

            if trx[0] == "E":
                env_rows[trx[1]] = trx[2]

            if trx[0] == "L":
                lrn_rows[trx[1]] = trx[2]

            if trx[0] == "I":
                int_rows[tuple(trx[1])] = trx[2]

        return Result(env_rows, lrn_rows, int_rows, exp_dict)

    def _encode(self,item):
        if item[0] == "T0":
            return ['experiment', {"n_learners":item[1], "n_environments":item[2], "description":None} if len(item)==3 else item[1] ]

        if item[0] == "T1":
            return ["L", item[1], item[2]]

        if item[0] == "T2":
            return ["E", item[1], item[2]]

        if item[0] == "T3":
            rows_T = collections.defaultdict(list)

            for row in item[2]:
                for col,val in row.items():
                    if col == "rewards" : col="reward"
                    if col == "reveals" : col="reveal"
                    rows_T[col].append(val)

            return ["I", item[1], { "_packed": rows_T }]

        return None

class TransactionIO(Source['Result'], Sink[Any]):

    def __init__(self, transaction_log: Optional[str] = None) -> None:

        if not transaction_log or not Path(transaction_log).exists():
            version = None
        else:
            version = JsonDecode().filter(next(DiskSource(transaction_log).read()))[1]

        if version == 3:
            self._transactionIO = TransactionIO_V3(transaction_log)

        elif version == 4:
            self._transactionIO = TransactionIO_V4(transaction_log)

        elif version is None:
            self._transactionIO = TransactionIO_V4(transaction_log)

        else:
            raise CobaException("We were unable to determine the appropriate Transaction reader for the file.")

    def write(self, transactions: Iterable[Any]) -> None:
        for transaction in transactions:
            self._transactionIO.write(transaction)

    def read(self) -> 'Result':
        return self._transactionIO.read()

class Points(NamedTuple):
    X    : Sequence[Any]
    Y    : Sequence[float] 
    XE   : Sequence[float]  = None
    YE   : Sequence[float]  = None
    color: Union[str,int]   = None
    alpha: float            = 1
    label: Optional[str]    = None
    style: Literal['-','.'] = "-"

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
        out: Union[None,Literal['screen'],str]) -> None:
        pass

class MatplotlibPlotter(Plotter):

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
        out: Union[None,Literal['screen'],str]
    ) -> None:

        xlim = xlim or [None,None]
        ylim = ylim or [None,None]

        PackageChecker.matplotlib('Result.plot_learners')
        import matplotlib.pyplot as plt #type: ignore

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

            for X, Y, XE, YE, c, a, l, fmt in lines:

                if l: any_label = True

                # we remove values outside of the given lims because matplotlib won't correctly scale otherwise
                if not YE:
                    XY = [(x,y) for i,x,y in zip(count(),X,Y) if in_lim(num_coalesce(x,i),xlim) and in_lim(num_coalesce(y,i),ylim)]
                    X,Y = map(list,zip(*XY)) if XY else ([],[])
                else:
                    XYE = [(x,y,e) for i,x,y,e in zip(count(),X,Y,YE) if in_lim(num_coalesce(x,i),xlim) and in_lim(num_coalesce(y,i),ylim)]
                    X,Y,YE = map(list,zip(*XYE)) if XYE else ([],[],[])

                if isinstance(c,int): c = color_cycle[c%len(color_cycle)]

                not_err_bar = lambda E: not E or all([e is None for e in E])

                if X and Y:
                    if all(map(not_err_bar,[XE,YE])):
                        ax.plot(X, Y, fmt,  color=c, alpha=a, label=l)
                    else:
                        XE = None if not_err_bar(XE) else list(zip(*XE)) if isinstance(XE[0],tuple) else XE
                        YE = None if not_err_bar(YE) else list(zip(*YE)) if isinstance(YE[0],tuple) else YE
                        error_every = max(int(len(X)*0.05),1) if fmt == "-" else 1
                        ax.errorbar(X, Y, YE, XE, fmt, elinewidth=0.5, errorevery=error_every, color=c, alpha=a, label=l)

            if xrotation is not None:
                plt.xticks(rotation=xrotation)

            if yrotation is not None:
                plt.yticks(rotation=yrotation)

            if xlim[0] is None or xlim[1] is None:
                ax.autoscale(axis='x')

            if ylim[0] is None or ylim[1] is None:
                ax.autoscale(axis='y')

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

            ax.set_title(title, loc='left', pad=15)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)

            if ax.get_legend() is None: #pragma: no cover
                scale = 0.65
                box1 = ax.get_position()
                ax.set_position([box1.x0, box1.y0 + box1.height * (1-scale), box1.width, box1.height * scale])
            else:
                ax.get_legend().remove()

            if any_label:
                ax.legend(*ax.get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(-.01, -.25), ncol=1, fontsize='medium') #type: ignore

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

class FilterPlottingData:

    def filter(self, rows:Sequence[Dict[str,Any]], x:Sequence[str], y:str, learner_ids:Sequence[int] = None) -> Sequence[Dict[str,Any]]:

        if not rows: raise CobaException("This result doesn't contain any evaluation data to plot.")

        rows = [row for row in rows if not learner_ids or row["learner_id"] in learner_ids]

        learner_count   = len(set([row['learner_id'] for row in rows]))
        environ_counts  = collections.Counter([row['environment_id'] for row in rows])
        environ_lengths = set([len(row[y]) for row in rows])
        min_env_length  = min(environ_lengths) 

        if max(environ_counts.values()) != learner_count:
            raise CobaException("This result does not contain an environment which has been finished for every learner. Plotting has been stopped.")

        if min(environ_counts.values()) != learner_count:
            CobaContext.logger.log("This result contains environments not present for all learners. Environments not present for all learners have been excluded. To supress this warning in the future call <result>.filter_fin() before plotting.") 

        if len(environ_lengths) > 1 and x == ['index']:
            CobaContext.logger.log("This result contains environments of different lengths. The plot only includes interactions up to the shortest environment. To supress this warning in the future call <result>.filter_fin(n_interactions) before plotting.")

        complete    = lambda row: environ_counts[row['environment_id']] == learner_count
        subselect_y = lambda ys: ys[:min_env_length] if x == ['index'] else ys
        subselect_r = lambda row: {k:(row[k] if k != y else subselect_y(row[k])) for k in ['environment_id','learner_id',y]}

        return list(map(subselect_r,filter(complete,rows)))

class SmoothPlottingData:

    def filter(self, rows:Sequence[Dict[str,Any]], y:str, span:Optional[int]) -> Sequence[Dict[str,Any]]:
        e_key = "environment_id"
        l_key = "learner_id"
        return [{e_key:row[e_key], l_key:row[l_key], y:moving_average(row[y],span)} for row in rows]

class ContrastPlottingData:

    def filter(self, rows:Sequence[Dict[str,Any]], y:str, mode:str, learner_id1:int) -> Sequence[Dict[str,Any]]:

        sort_key  = lambda row: (row['environment_id'], 0 if row['learner_id']==learner_id1 else 1)
        group_key = lambda row: row['environment_id']

        if mode == "prob":
            contraster = lambda Y1,Y2: list(map(int,map(gt,Y1,Y2))) 
        elif mode == "diff": 
            contraster = lambda Y1,Y2: list(map(sub,Y1,Y2))
        else:
            contraster = lambda Y1,Y2: list(zip(Y1,Y2))

        contrast = lambda env_id, pair: {'environment_id': env_id, y: contraster(pair[0][y],pair[1][y]) }

        return [ contrast(env_id,list(pair)) for env_id, pair in groupby(sorted(rows,key=sort_key),key=group_key) ]

class TransformToXYE:

    def filter(self,
        rows: Sequence[Dict[str,Any]], 
        envs: Dict[int,Dict[str,Any]], 
        x:Sequence[str], 
        y:str,
        err:Union[str,None,PointAndInterval]) -> Sequence[Tuple[Any,float,Union[None,float,Tuple[float,float]]]]:

        if err == 'sd':
            avg = Mean()
            std = StandardDeviation()
            YE = lambda z: (avg.calculate(z), std.calculate(z))
        elif err == 'se':
            YE = StandardErrorOfMean().calculate
        elif err == "bs":
            YE = BootstrapConfidenceInterval(.95, Mean().calculate).calculate
        elif err == "bi":
            YE = BinomialConfidenceInterval('wilson').calculate
        elif isinstance(err,PointAndInterval):
            YE = err.calculate
        else:
            avg = Mean()
            YE = lambda z: (avg.calculate(z) if len(z) > 1 else z[0], None)

        iters = [ iter(row[y]) for row in rows ]
        first_val = next(iters[0])
        iters[0] = chain([first_val],iters[0])

        is_scatter = isinstance(first_val,(list,tuple))

        if x == ['index']:
            Z = zip(*iters)
            X = count(1)

            if not is_scatter:
                points = [(x,None)+YE(z) for x,z in zip(X,Z) ]
            else:
                points = [YE(z1)+YE(z2) for _,z in zip(X,Z) for z1,z2 in [tuple(zip(*z))]]
        else:
            XZ = collections.defaultdict(list)
            make_x = lambda env: env[x[0]] if len(x) == 1 else tuple(env[k] for k in x)
            for row,I in zip(rows,iters):
                XZ[str(make_x(envs[row["environment_id"]]))].append(list(I)[-1])

            if not is_scatter:
                points = [(x,None)+YE(z) for x,z in XZ.items()]
            else:
                points = [YE(z1)+YE(z2) for _,z in XZ.items() for z1,z2 in [tuple(zip(*z))]]

        return [ (x,y,xe,ye) for x,xe,y,ye in points ]

class Result:
    """A class representing the result of an Experiment."""

    @staticmethod
    def from_file(filename: str) -> 'Result':
        """Create a Result from a transaction file."""

        if not Path(filename).exists():
            raise CobaException("We were unable to find the given Result file.")

        return TransactionIO(filename).read()

    def __init__(self,
        env_rows: Dict[int           ,Dict[str,Any]] = {},
        lrn_rows: Dict[int           ,Dict[str,Any]] = {},
        int_rows: Dict[Tuple[int,int],Dict[str,Any]] = {},
        exp_dict: Dict[str, Any]                     = {}) -> None:
        """Instantiate a Result class.

        This constructor should never be called directly. Instead a Result file should be created
        from an Experiment and the result file should be loaded via Result.from_file(filename).
        """

        env_flat = [ { "environment_id":k,                       **v } for k,v in env_rows.items() ]
        lrn_flat = [ {                        "learner_id" :k,   **v } for k,v in lrn_rows.items() ]
        int_flat = [ { "environment_id":k[0], "learner_id":k[1], **v } for k,v in int_rows.items() ]

        self.experiment = exp_dict

        self._environments = Table            ("Environments", ['environment_id'              ], env_flat, ["source"])
        self._learners     = Table            ("Learners"    , ['learner_id'                  ], lrn_flat, ["family","shuffle","take"])
        self._interactions = InteractionsTable("Interactions", ['environment_id', 'learner_id'], int_flat, ["index","reward"])

        self._plotter = MatplotlibPlotter()

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
    def interactions(self) -> InteractionsTable:
        """The collection of interactions that learners chose actions for in the Experiment.

        The primary key of this Table is (index, environment_id, learner_id). It should be noted that the InteractionTable
        has an additional property, to_progressive_pandas() which calculates the expanding moving average or exponential
        moving average for interactions ordered by index and grouped by environment_id and learner_id.
        """
        return self._interactions

    def set_plotter(self, plotter: Plotter) -> None:
        """Manually set the underlying plotting tool. By default matplotlib is used though this can be changed."""
        self._plotter = plotter

    def copy(self) -> 'Result':
        """Create a copy of Result."""

        result = Result()

        result._environments = copy(self._environments)
        result._learners     = copy(self._learners)
        result._interactions = copy(self._interactions)

        return result

    def filter_fin(self, n_interactions: int = None) -> 'Result':
        """Filter the result to only contain data about environments with all learners and interactions.

        Args:
            n_interactions: The number of interactions at which an environment is considered complete.
        """

        def has_all(env_id):
            return all((env_id, lrn_id) in self.interactions for lrn_id in self.learners.keys)

        def has_min(env_id):
            if not n_interactions:
                return True
            else:
                return all(len(self.interactions[(env_id, lrn_id)].get('index',[])) >= n_interactions for lrn_id in self.learners.keys)

        complete_ids = [env_id for env_id in self.environments.keys if has_all(env_id) and has_min(env_id)]

        new_result               = copy(self)
        new_result._environments = self.environments.filter(environment_id=complete_ids)
        new_result._interactions = self.interactions.filter(environment_id=complete_ids)

        if n_interactions:
            new_inter = copy(new_result.interactions)
            new_inter._rows_pack = { rk: { pk: pv[0:n_interactions] for pk, pv in rv.items() } for rk,rv in new_inter._rows_pack.items() }
            new_result._interactions = new_inter

        if len(new_result.environments) == 0:
            CobaContext.logger.log(f"There was no environment which was finished for every learner.")

        return new_result

    def filter_env(self, pred:Callable[[Dict[str,Any]],bool] = None, **kwargs: Any) -> 'Result':
        """Filter the result to only contain data about specific environments.

        Args:
            pred: A predicate that returns true for learner dictionaries that should be kept.
            **kwargs: key-value pairs to filter on. To see filtering options see Table.filter.
        """

        new_result = copy(self)
        new_result._environments = new_result.environments.filter(pred, **kwargs)
        new_result._interactions = new_result.interactions.filter(lambda row: row['environment_id'] in new_result.environments)

        if len(new_result.environments) == 0:
            CobaContext.logger.log(f"No environments matched the given filter.")

        return new_result

    def filter_lrn(self, pred:Callable[[Dict[str,Any]],bool] = None, **kwargs: Any) -> 'Result':
        """Filter the result to only contain data about specific learners.

        Args:
            pred: A predicate that returns true for learner dictionaries that should be kept.
            **kwargs: key-value pairs to filter on. To see filtering options see Table.filter.
        """

        new_result = copy(self)
        new_result._learners     = new_result.learners.filter(pred, **kwargs)
        new_result._interactions = new_result.interactions.filter(learner_id=new_result.learners)

        if len(new_result.learners) == 0:
            CobaContext.logger.log(f"No learners matched the given filter.")

        return new_result

    def plot_contrast(self,
        learner_id1: int,
        learner_id2: int,
        x          : Union[str, Sequence[str]] = "environment_id",
        y          : str = "reward",
        mode       : Literal["diff","prob",'scat'] = "diff",
        span       : int = None,
        err        : Union[Literal['se','sd','bs'], None, PointAndInterval] = None,
        labels     : Sequence[str] = None,
        colors     : Sequence[str] = None,
        xlim       : Tuple[Optional[Number],Optional[Number]] = None,
        ylim       : Tuple[Optional[Number],Optional[Number]] = None,
        xticks     : bool = True,
        yticks     : bool = True,
        out        : Union[None,Literal['screen'],str] = 'screen',
        ax = None) -> None:
        """Plot a direct contrast of the performance for two learners.

        Args:
            learner_id1: The first learner to plot in the contrast.
            learner_id2: The second learner to plot in the contrast.
            x: The value to plot on the x-axis. This can either be index or environment columns to group by.
            y: The value to plot on the y-axis.
            mode: The kind of contrast plot to make: diff plots the pairwise difference, prob plots the the probability
                of learner_id1 beating learner_id2, and scatter plots learner_id1 on x-axis and learner_id2 on y axis.
            span: The number of y values to smooth together when reporting y. If this is None then the average of all y
                values up to current is shown otherwise a moving average with window size of span (the window will be
                smaller than span initially).
            err: This determines what kind of error bars to plot (if any). If `None` then no bars are plotted, if 'se'
                the standard error is shown, and if 'sd' the standard deviation is shown.
            labels: The legend labels to use in the plot. These should be in order of the actual legend labels.
            colors: The colors used to plot the learners plot.
            xlim: Define the x-axis limits to plot. If `None` the x-axis limits will be inferred.
            ylim: Define the y-axis limits to plot. If `None` the y-axis limits will be inferred.
            xticks: Whether the x-axis labels should be drawn.
            yticks: Whether the y-axis labels should be drawn.
            out: Indicate where the plot should be sent to after plotting is finished.
            ax: Provide an optional axes that the plot will be drawn to. If not provided a new figure/axes is created.
        """

        xlim = xlim or [None,None]
        ylim = ylim or [None,None]

        x = [x] if isinstance(x,str) else list(x)
        self._validate_parameters(x)

        rows = FilterPlottingData().filter(list(self.interactions), x, y, [learner_id1, learner_id2])
        rows = SmoothPlottingData().filter(rows, y, span)
        rows = ContrastPlottingData().filter(rows, y, mode, learner_id1)

        XYE = TransformToXYE().filter(rows, self.environments, x, y, err)

        if x != ['index']:
            XYE = sorted(XYE, key=lambda xye: xye[1])

        bound = .5 if mode == "prob" else 0

        win,tie,loss = [],[],[]
        for _x,_y,_xe,_ye in XYE:
            xl,xu = (0,0) if _xe is None else (_xe,_xe) if isinstance(_xe,Number) else _xe
            yl,yu = (0,0) if _ye is None else (_ye,_ye) if isinstance(_ye,Number) else _ye
            if mode != 'scat':
                (win if _y-yl > bound else loss if _y+yu < bound else tie).append((_x,_y,_xe,_ye))
            else:
                (win if _x-xl>_y+yu else loss if _y-yl>_x+xu else tie).append((_x,_y,_xe,_ye))

        colors = (colors or []) + [0,1,2]
        labels = (labels or []) + [self.learners[learner_id1]['full_name'], self.learners[learner_id2]['full_name']]

        fmt = "-" if x == ['index'] else "."

        plots = []

        if loss: plots.append(Points(*zip(*loss), colors[0], 1, labels[1] + " " + f"({len(loss)})", fmt))
        if tie : plots.append(Points(*zip(*tie) , colors[1], 1, 'Tie'     + " " + f"({len(tie )})", fmt))
        if win : plots.append(Points(*zip(*win) , colors[2], 1, labels[0] + " " + f"({len(win )})", fmt))

        if mode != 'scat':
            leftmost_x  = (loss+tie+win)[ 0][0]
            rightmost_x = (loss+tie+win)[-1][0]
            plots.append(Points((leftmost_x,rightmost_x),(bound,bound), None, None , "#888", 1, None, '-'))
        else:
            m    = max([p[0] for p in (loss+tie+win)]+[p[1] for p in (loss+tie+win)]+[1])
            plots.append(Points((0,m),(0,m), None, None , "#888", 1, None, '-'))

        xrotation = 90 if x != ['index'] and len(XYE)>5 else 0
        yrotation = 0

        if mode != "scat":
            xlabel = "Interaction" if x==['index'] else x[0] if len(x) == 1 else x
            ylabel = f"{labels[0]} - {labels[1]}" if mode=="diff" else f"P({labels[0]} > {labels[1]})"
        else:
            xlabel = y
            ylabel = y

        title  = f"{ylabel} ({len(rows) if x==['index'] else len(XYE)} Environments)"

        self._plotter.plot(ax, plots, title, xlabel, ylabel, xlim, ylim, xticks, yticks, xrotation, yrotation, out)

    def plot_learners(self,
        ids   : Union[int,Sequence[int]] = None, 
        x     : Union[str,Sequence[str]] = "index",
        y     : str = "reward",
        span  : int = None,
        err   : Union[Literal['se','sd','bs'], None, PointAndInterval] = None,
        labels: Sequence[str] = None,
        colors: Sequence[Union[str,int]] = None,
        xlim  : Tuple[Optional[Number],Optional[Number]] = None,
        ylim  : Tuple[Optional[Number],Optional[Number]] = None,
        xticks: bool = True,
        yticks: bool = True,
        top_n: int = None,
        out   : Union[None,Literal['screen'],str] = 'screen',
        ax = None) -> None:
        """Plot the performance of multiple learners on multiple environments. It gives a sense of the expected
            performance for different learners across independent environments. This plot is valuable in gaining
            insight into how various learners perform in comparison to one another.

        Args:
            ids: Sequence of learner ids to plot (if None we will plot all learnesr in the result).
            x: The value to plot on the x-axis. This can either be index or environment columns to group by.
            y: The value to plot on the y-axis.
            span: The number of y values to smooth together when reporting y. If this is None then the average of all y
                values up to current is shown otherwise a moving average with window size of span (the window will be
                smaller than span initially).
            err: This determines what kind of error bars to plot (if any). If `None` then no bars are plotted, if 'se'
                the standard error is shown, and if 'sd' the standard deviation is shown.
            labels: The legend labels to use in the plot. These should be in order of the actual legend labels.
            colors: The colors used to plot the learners plot.
            xlim: Define the x-axis limits to plot. If `None` the x-axis limits will be inferred.
            ylim: Define the y-axis limits to plot. If `None` the y-axis limits will be inferred.
            xticks: Whether the x-axis labels should be drawn.
            yticks: Whether the y-axis labels should be drawn.
            top_n: Only plot the top_n learners. If `None` all learners will be plotted.
            out: Indicate where the plot should be sent to after plotting is finished.
            ax: Provide an optional axes that the plot will be drawn to. If not provided a new figure/axes is created.
        """

        xlim = xlim or [None,None]
        ylim = ylim or [None,None]

        if isinstance(ids,int): ids = [ids]
        if isinstance(colors,int): colors = [colors]
        if isinstance(labels,str): labels = [labels]
        if isinstance(x,str): x = [x]

        self._validate_parameters(x)

        interactions = self.interactions if not ids else self.filter_lrn(learner_id=ids).interactions

        rows = FilterPlottingData().filter(list(interactions), x, y)
        rows = SmoothPlottingData().filter(rows, y, span)

        env_rows = self.environments
        get_key  = lambda row: row['learner_id']
        lines    = []

        style = "-" if x == ['index'] else "."

        def get_color(colors:Sequence[Union[str,int]], i:int):
            return i if not colors else i+max(colors) if isinstance(colors[0],int) else colors[i] if i < len(colors) else i
        
        def get_label(labels:Sequence[str], i:int):
            return labels[i] if labels and i < len(labels) else self.learners[lrn_id]['full_name']

        for i, (lrn_id, lrn_rows) in enumerate(groupby(sorted(rows, key=get_key),key=get_key)):
            lrn_rows = list(lrn_rows)
            XYE      = TransformToXYE().filter(lrn_rows, env_rows, x, y, err)
            color    = get_color(colors,i)
            label    = get_label(labels,i)
            lines.append(Points(*zip(*XYE), color, 1, label, style))

        lines  = sorted(lines, key=lambda line: line[1][-1], reverse=True)
        labels = [l.label for l in lines]
        xlabel = "Interaction" if x==['index'] else x[0] if len(x) == 1 else x
        ylabel = y.capitalize().replace("_pct"," Percent")

        title = ("Instantaneous" if span == 1 else f"Span {span}" if span else "Progressive") + f" {ylabel}"
        title = title + f" ({len(lrn_rows) if x==['index'] else len(XYE)} Environments)"

        if x != ['index']: title = f"Final {title}"
        if top_n         : lines = [l._replace(color=get_color(colors,i),label=get_label(labels,i)) for i,l in enumerate(lines[:top_n]) ]

        self._plotter.plot(ax, lines, title, xlabel, ylabel, xlim, ylim, xticks, yticks, 0, 0, out)

    def __str__(self) -> str:
        return str({"Learners": len(self._learners), "Environments": len(self._environments), "Interactions": len(self._interactions) })

    def _ipython_display_(self):
        #pretty print in jupyter notebook (https://ipython.readthedocs.io/en/stable/config/integrating.html)
        print(str(self))

    def _validate_parameters(self, x:Sequence[str]):
        if 'index' in x and len(x) > 1:
            raise CobaException('The x-axis cannot contain both interaction index and environment features. Please choose one or the other.')
