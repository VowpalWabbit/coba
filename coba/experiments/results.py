import re
import collections
import collections.abc

from copy import copy
from pathlib import Path
from numbers import Number
from operator import truediv, sub, gt, itemgetter
from abc import abstractmethod
from itertools import chain, repeat, accumulate, groupby, count, compress
from typing import Any, Mapping, Tuple, Optional, Sequence, Iterable, Iterator, Union, Callable, NamedTuple
from coba.backports import Literal

from coba.environments import Environment
from coba.statistics import mean, stdev, StandardErrorOfMean, BootstrapConfidenceInterval, BinomialConfidenceInterval, PointAndInterval
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

def old_to_new(
    env_rows: Mapping[int           ,Mapping[str,Any]] = {},
    lrn_rows: Mapping[int           ,Mapping[str,Any]] = {},
    int_rows: Mapping[Tuple[int,int],Mapping[str,Any]] = {}) -> Tuple[Sequence,Sequence,Sequence]:

    env_cols = set().union(*[v.keys()            for v in env_rows.values()]) | {'environment_id'}
    lrn_cols = set().union(*[v.keys()            for v in lrn_rows.values()]) | {'learner_id'}
    int_cols = set().union(*[v['_packed'].keys() for v in int_rows.values()]) | {'environment_id','learner_id','index'}

    pref_env_cols = {'environment_id':'a'                                                }
    pref_lrn_cols = {                     'learner_id': 'a'                              }
    pref_int_cols = {'environment_id':'a','learner_id':'aa','index':'aaa','reward':'aaaa'}

    env_cols = sorted(env_cols, key=lambda k: pref_env_cols.get(k,k))
    lrn_cols = sorted(lrn_cols, key=lambda k: pref_lrn_cols.get(k,k))
    int_cols = sorted(int_cols, key=lambda k: pref_int_cols.get(k,k))

    env_flat = [ { "environment_id":k, **v } for k,v in env_rows.items() ]
    env_flat = [ [ row.get(k) for k in env_cols] for row in env_flat ]

    lrn_flat = [ { "learner_id" :k,   **v } for k,v in lrn_rows.items() ]
    lrn_flat = [ [ row.get(k) for k in lrn_cols] for row in lrn_flat ]

    int_flat = []
    for (env_id, lrn_id), results in int_rows.items():
        names,cols = zip(*results['_packed'].items())
        N          = len(cols[0])
        names      = ['environment_id','learner_id','index'] + list(names)
        cols       = [[env_id]*N, [lrn_id]*N, list(range(1,N+1))] + list(cols)

        if set(int_cols) != set(names):
            new_names = list(set(int_cols)-set(names))
            names.extend(new_names)
            cols.extend([[None]*N]*len(new_names))

        col_index_order = sorted(range(len(names)), key=lambda i: int_cols.index(names[i]))
        cols = [cols[i] for i in col_index_order]

        int_flat.extend(zip(*cols))

    return [env_cols]+env_flat, [lrn_cols]+lrn_flat, [int_cols]+int_flat

class Table:
    """A container class for storing tabular data."""

    def __init__(self, cols: Sequence[str], rows: Sequence[Sequence[Any]]):
        """Instantiate a Table.

        Args:
            cols: The names assigned to each item in an element of rows.
            rows: The actual rows that should be stored in the table.
        """
        self._col_names = cols
        self.row_vals   = rows
        self.col_vals   = list(zip(*rows)) if rows else [[]]*len(cols)

    @property
    def columns(self) -> Sequence[str]:
        """The columns in the table."""
        return self._col_names

    def filter(self, row_pred:Callable[[Sequence[Any]],bool] = None, **kwargs) -> 'Table':
        """Filter to specific rows. Applied as an Or. To "And" call filter multiple times.

        Args:
            pred: A predicate that returns true for row dictionaries that should be kept.
            kwargs: key value pairs where the key is the column and the value indicates what
                value a row should have in that column to be kept. Keeping logic depends on
                the row value type and the kwargs value type. If kwarg value == row value keep
                the row. If kwarg value is callable pass the row value to the predicate. If
                kwarg value is a collection keep the row if the row value is in the collection.
                If kwarg value is a string apply a regular expression match to the row value.
        """

        any_keep = list(map(row_pred,self)) if row_pred else [False]*len(self) if kwargs else [True]*len(self)

        if kwargs: col_major_order = self.col_vals

        for filter_col, filter_val in kwargs.items():
            values = col_major_order[self._col_names.index(filter_col)]

            if callable(filter_val):
                any_keep = [a or filter_val(v) for a,v in zip(any_keep,values)]

            else:
                is_sequence = isinstance(filter_val, collections.abc.Sequence) and not isinstance(filter_val,str)
                filter_vals = filter_val if is_sequence else [filter_val]

                for filter_val in filter_vals:
                    if isinstance(filter_val,Number) and isinstance(values[0],Number):
                        any_keep = [a or v == filter_val for a,v in zip(any_keep,values)]
                    elif isinstance(filter_val,Number) and isinstance(values[0],str):
                        _re = re.compile(f'(\D|^){filter_val}(\D|$)')
                        any_keep = [a or bool(_re.search(v)) for a,v in zip(any_keep,values)]
                    elif isinstance(filter_val,str) and isinstance(values[0],str):
                        _re = re.compile(filter_val)
                        any_keep = [a or filter_val == v or bool(_re.search(v)) for a,v in zip(any_keep,values)]

        return Table(self._col_names, list(compress(self.row_vals, any_keep)))

    def to_pandas(self) -> Any:
        """Turn the Table into a Pandas data frame."""

        PackageChecker.pandas("Table.to_pandas")
        import pandas as pd #type: ignore
        return pd.DataFrame(self.row_vals, columns=self.columns)

    def to_dicts(self) -> Sequence[Mapping[str,Any]]:
        """Turn the Table into a sequence of tuples."""
        return [dict(zip(self.columns,row)) for row in self.row_vals]

    def __iter__(self) -> Iterator[Sequence[Any]]:
        return iter(self.row_vals)

    def __str__(self) -> str:
        return str({"Columns": self.columns, "Rows": len(self)})

    def __len__(self) -> int:
        return len(self.row_vals)

    def _ipython_display_(self):
        #pretty print in jupyter notebook (https://ipython.readthedocs.io/en/stable/config/integrating.html)
        print(str(self))

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
    
        return Result(*old_to_new(env_rows, lrn_rows, int_rows), exp_dict)

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
        env_rows = {}
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
                env_rows[trx[1]] = trx[2]

            if trx[0] == "L":
                lrn_rows[trx[1]] = trx[2]

            if trx[0] == "I":
                int_rows[tuple(trx[1])] = trx[2]

        return Result(*old_to_new(env_rows, lrn_rows, int_rows), {"n_learners":n_lrns,"n_environments":n_envs})

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
    zorder:int              = 1

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

            for X, Y, XE, YE, c, a, l, fmt,z in lines:

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
                        ax.plot(X, Y, fmt,  color=c, alpha=a, label=l,zorder=z)
                    else:
                        XE = None if not_err_bar(XE) else list(zip(*XE)) if isinstance(XE[0],tuple) else XE
                        YE = None if not_err_bar(YE) else list(zip(*YE)) if isinstance(YE[0],tuple) else YE
                        error_every = max(int(len(X)*0.05),1) if fmt == "-" else 1
                        elinewidth = 0.5 if 'elinewidth' not in CobaContext.store else CobaContext.store['elinewidth']
                        ax.errorbar(X, Y, YE, XE, fmt, elinewidth=elinewidth, errorevery=error_every, color=c, alpha=a, label=l,zorder=z)

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

    def filter(self, interactions:Table, x:Sequence[str], y:str, learner_ids:Sequence[int] = None) -> Table:

        if len(interactions) == 0: raise CobaException("This result doesn't contain any evaluation data to plot.")
        if y not in interactions.columns: raise CobaException(f"{y} is not available in the environment. Plotting has been stopped.")
        
        if learner_ids: interactions = interactions.filter(learner_id = set(learner_ids).__contains__)

        env_lrn_row_counts = collections.Counter(zip(*interactions.col_vals[:2]))
        env_lrn_counts     = collections.Counter([e for e,_ in env_lrn_row_counts.keys()])
        env_lengths        = env_lrn_row_counts.values()

        n_learners = len(set(l for _,l in env_lrn_row_counts.keys()))

        if max(env_lrn_counts.values()) != n_learners:
            raise CobaException("This result does not contain an environment which has been finished for every learner. Plotting has been stopped.")

        if min(env_lrn_counts.values()) != n_learners:
            CobaContext.logger.log("Environments not present for all learners have been excluded. To supress this call filter_fin() before plotting.")
            complete_environments = [e for e,v in env_lrn_counts.items() if v == n_learners]
            interactions = interactions.filter(environment_id=set(complete_environments).__contains__)

        if len(set(env_lengths)) > 1 and x == ['index']:
            CobaContext.logger.log("This result contains environments of different lengths. The plot only includes interactions up to the shortest environment. To supress this warning in the future call <result>.filter_fin(n_interactions) before plotting.")
            complete_indexes = list(range(1,min(env_lengths)+1))
            interactions = interactions.filter(index=set(complete_indexes).__contains__)

        return interactions

class SmoothPlottingData:

    def filter(self, interactions:Table, y:str, span:Optional[int]) -> Sequence[Mapping[str,Any]]:
        
        Y = interactions.col_vals[interactions.columns.index(y)]
        G = iter(zip(*interactions.col_vals[:2]))
        
        out = []

        for g, _Y in groupby(Y, lambda key, G=G: next(G)):
            out.append({"environment_id":g[0], "learner_id":g[1], y:moving_average(list(_Y),span)})

        return out

class ContrastPlottingData:

    def filter(self, rows:Sequence[Mapping[str,Any]], y:str, mode:Union[Literal["diff","prob",'scat'],Callable[[float,float],float]], learner_id1:int) -> Sequence[Mapping[str,Any]]:

        sort_key  = lambda row: (row['environment_id'], 0 if row['learner_id']==learner_id1 else 1)
        group_key = lambda row: row['environment_id']

        if mode == "prob":
            contraster = lambda Y1,Y2: list(map(int,map(gt,Y1,Y2))) 
        elif mode == "diff": 
            contraster = lambda Y1,Y2: list(map(sub,Y1,Y2))
        elif mode =="scat":
            contraster = lambda Y1,Y2: list(zip(Y1,Y2))
        else:
            contraster = lambda Y1,Y2: list(map(mode,Y1,Y2))

        contrast = lambda env_id, pair: {'environment_id': env_id, y: contraster(pair[0][y],pair[1][y]) }

        return [ contrast(env_id,list(pair)) for env_id, pair in groupby(sorted(rows,key=sort_key),key=group_key) ]

class TransformToXYE:

    def filter(self,
        rows: Sequence[Mapping[str,Any]], 
        envs: Mapping[int,Mapping[str,Any]], 
        x:Sequence[str], 
        y:str,
        err:Union[str,None,PointAndInterval]) -> Sequence[Tuple[Any,float,Union[None,float,Tuple[float,float]]]]:

        if err == 'sd':
            YE = lambda z: (mean(z), stdev(z))
        elif err == 'se':
            YE = StandardErrorOfMean(1.96).calculate #z-score for .975 area to the left and right
        elif err == "bs":
            YE = BootstrapConfidenceInterval(.95, mean).calculate
        elif err == "bi":
            YE = BinomialConfidenceInterval('wilson').calculate
        elif isinstance(err,PointAndInterval):
            YE = err.calculate
        else:
            YE = lambda z: (mean(z) if len(z) > 1 else z[0], None)

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
            
            make_x = lambda eid: eid if list(x) == ['environment_id'] else envs[eid].get(x[0]) if len(x) == 1 else tuple(envs[eid].get(k) for k in x)
            for row,I in zip(rows,iters):
                XZ[str(make_x(row["environment_id"]))].append(list(I)[-1])

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

    @staticmethod
    def from_logged_envs(environments: Sequence[Environment]):

        environments = [e for e in environments if e.params.get('logged')]

        envs_params = []
        lrns_params = []

        env_rows = {}
        lrn_rows = {}
        int_rows = {}

        my_mean = lambda x: sum(x)/len(x)

        for env in environments:
            
            is_batched = 'batched' in env.params
            env_params  = dict(env.params)
            lrn_params  = env_params.pop('learner')

            try:
                env_id = envs_params.index(env_params)
            except:
                env_id = len(envs_params)
                envs_params.append(env_params)
                env_rows[env_id] = env_params

            try:
                lrn_id = lrns_params.index(lrn_params)
            except:
                lrn_id = len(lrns_params)
                lrns_params.append(lrn_params)
                lrn_rows[lrn_id] = lrn_params

            if is_batched:
                results = {'_packed':{'reward': list(map(my_mean,map(itemgetter('reward'),env.read())))}}
            else:
                results = {'_packed':{'reward': list(map(itemgetter('reward'),env.read())) }}

            int_rows[(env_id,lrn_id)]= results

        return Result(*old_to_new(env_rows,lrn_rows,int_rows))

    def __init__(self,
        env_rows: Union[Sequence,Table] = Table([],[]),
        lrn_rows: Union[Sequence,Table] = Table([],[]),
        int_rows: Union[Sequence,Table] = Table([],[]),
        exp_dict: Mapping  = {}) -> None:
        """Instantiate a Result class.

        This constructor should never be called directly. Instead a Result file should be created
        from an Experiment and the result file should be loaded via Result.from_file(filename).
        """
        self.experiment = exp_dict

        self._environments = env_rows if isinstance(env_rows,Table) else Table(env_rows[0], env_rows[1:])
        self._learners     = lrn_rows if isinstance(lrn_rows,Table) else Table(lrn_rows[0], lrn_rows[1:])
        self._interactions = int_rows if isinstance(int_rows,Table) else Table(int_rows[0], int_rows[1:])

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
    def interactions(self) -> Table:
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

        counts = collections.Counter(zip(*self.interactions.col_vals[:2]))

        def has_all(env_id):
            return all((env_id, lrn_id) in counts for lrn_id in self.learners.col_vals[0])

        def has_min(env_id):
            return n_interactions == None or all(counts[(env_id, lrn_id)] >= n_interactions for lrn_id in self.learners.col_vals[0])

        complete_ids = [env_id for env_id in self.environments.col_vals[0] if has_all(env_id) and has_min(env_id)]

        environments = self.environments.filter(environment_id=complete_ids)
        interactions = self.interactions.filter(environment_id=complete_ids)

        if n_interactions:
            #in theory this could be a little faster but we'd have to break the API
            interactions = interactions.filter(index=set(range(1,n_interactions+1)).__contains__)

        if len(environments) == 0:
            CobaContext.logger.log(f"There was no environment which was finished for every learner.")

        return Result(environments, self.learners, interactions, self.experiment)

    def filter_env(self, pred:Callable[[Mapping[str,Any]],bool] = None, **kwargs: Any) -> 'Result':
        """Filter the result to only contain data about specific environments.

        Args:
            pred: A predicate that returns true for learner dictionaries that should be kept.
            **kwargs: key-value pairs to filter on. To see filtering options see Table.filter.
        """

        environments = self.environments.filter(pred, **kwargs)
        interactions = self.interactions.filter(environment_id=set(environments.col_vals[0]).__contains__)
        learners     = self.learners    .filter(learner_id=set(interactions.col_vals[1]).__contains__)

        if len(environments) == 0:
            CobaContext.logger.log(f"No environments matched the given filter.")

        return Result(environments,learners,interactions,self.experiment) 

    def filter_lrn(self, pred:Callable[[Mapping[str,Any]],bool] = None, **kwargs: Any) -> 'Result':
        """Filter the result to only contain data about specific learners.

        Args:
            pred: A predicate that returns true for learner dictionaries that should be kept.
            **kwargs: key-value pairs to filter on. To see filtering options see Table.filter.
        """

        
        learners     = self.learners.filter(pred, **kwargs)
        interactions = self.interactions.filter(learner_id=set(learners.col_vals[0]).__contains__)
        environments = self.environments.filter(environment_id=set(interactions.col_vals[0]).__contains__)

        if len(learners) == 0:
            CobaContext.logger.log(f"No learners matched the given filter.")

        return Result(environments,learners,interactions)

    def plot_contrast(self,
        learner_id1: int,
        learner_id2: int,
        x          : Union[str, Sequence[str]] = "environment_id",
        y          : str = "reward",
        mode       : Union[Literal["diff","prob",'scat'],Callable[[float,float],float]] = "diff",
        span       : int = None,
        err        : Union[Literal['se','sd','bs'], None, PointAndInterval] = None,
        labels     : Sequence[str] = None,
        colors     : Sequence[str] = None,
        xlim       : Tuple[Optional[Number],Optional[Number]] = None,
        ylim       : Tuple[Optional[Number],Optional[Number]] = None,
        xticks     : bool = True,
        yticks     : bool = True,
        reverse    : bool = False,
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
            reverse: Whether to reverse order when counting winners and losers.
            out: Indicate where the plot should be sent to after plotting is finished.
            ax: Provide an optional axes that the plot will be drawn to. If not provided a new figure/axes is created.
        """
        try:
            xlim = xlim or [None,None]
            ylim = ylim or [None,None]

            x = [x] if isinstance(x,str) else list(x)
            self._validate_parameters(x)

            if x == ['index']:
                raise CobaException("plot_contrast does not currently support contrasting by `index`.")

            rows = FilterPlottingData().filter(self.interactions, x, y, [learner_id1, learner_id2])
            rows = SmoothPlottingData().filter(rows, y, span)
            rows = ContrastPlottingData().filter(rows, y, mode, learner_id1)

            envs = self.environments
            XYE = TransformToXYE().filter(rows, { row[0]:dict(zip(envs.columns,row)) for row in envs }, x, y, err)

            if x != ['index']:
                XYE = sorted(XYE, key=lambda xye: xye[1], reverse=reverse)

            bound = .5 if mode == "prob" else 0

            win,tie,loss = [],[],[]
            for _x,_y,_xe,_ye in XYE:
                o = 1 if not reverse else -1

                xl,xu = (0,0) if _xe is None else (_xe,_xe) if isinstance(_xe,Number) else _xe
                yl,yu = (0,0) if _ye is None else (_ye,_ye) if isinstance(_ye,Number) else _ye

                if mode != 'scat':
                    (win if _y-yl > o*bound else loss if _y+yu < o*bound else tie).append((_x,_y,_xe,_ye))
                else:
                    (win if _x-xl>o*(_y+yu) else loss if _y-yl> o*(_x+xu) else tie).append((_x,_y,_xe,_ye))

            colors = (colors or []) + [0,1,2]
            labels = (labels or []) + [self._full_name(learner_id1), self._full_name(learner_id2)]

            fmt = "-" if x == ['index'] else "."

            plots = []

            if not reverse:
                if loss: plots.append(Points(*zip(*loss), colors[0], 1, labels[1] + " " + f"({len(loss)})", fmt))
                if tie : plots.append(Points(*zip(*tie) , colors[1], 1, 'Tie'     + " " + f"({len(tie )})", fmt))
                if win : plots.append(Points(*zip(*win) , colors[2], 1, labels[0] + " " + f"({len(win )})", fmt))
            else:
                if win : plots.append(Points(*zip(*win) , colors[2], 1, labels[0] + " " + f"({len(win )})", fmt))
                if tie : plots.append(Points(*zip(*tie) , colors[1], 1, 'Tie'     + " " + f"({len(tie )})", fmt))
                if loss: plots.append(Points(*zip(*loss), colors[0], 1, labels[1] + " " + f"({len(loss)})", fmt))

            if mode != 'scat':
                leftmost_x  = (loss+tie+win)[ 0][0] if not reverse else (win+tie+loss)[ 0][0]
                rightmost_x = (loss+tie+win)[-1][0] if not reverse else (win+tie+loss)[-1][0]
                plots.append(Points((leftmost_x,rightmost_x),(bound,bound), None, None , "#888", 1, None, '-',.5))
            else:
                m = max([p[0] for p in (loss+tie+win)]+[p[1] for p in (loss+tie+win)]+[1])
                plots.append(Points((0,m),(0,m), None, None , "#888", 1, None, '-',.5))

            xrotation = 90 if x != ['index'] and len(XYE)>5 else 0
            yrotation = 0

            if mode != "scat":
                xlabel = "Interaction" if x==['index'] else x[0] if len(x) == 1 else x
                ylabel = f"{labels[0]} - {labels[1]}" if mode=="diff" else f"P({labels[0]} > {labels[1]})"
            else:
                xlabel = y
                ylabel = y

            title = f"{ylabel} ({len(rows) if x==['index'] else len(XYE)} Environments)"

            self._plotter.plot(ax, plots, title, xlabel, ylabel, xlim, ylim, xticks, yticks, xrotation, yrotation, out)
        except CobaException as e:
            CobaContext.logger.log(str(e)) 

    def plot_learners(self,
        x     : Union[str,Sequence[str]] = "index",
        y     : str = "reward",
        span  : int = None,
        err   : Union[Literal['se','sd','bs'], None, PointAndInterval] = None,
        labels: Sequence[str] = None,
        colors: Union[int,Sequence[Union[str,int]]] = None,
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
            top_n: Only plot the top_n learners. If `None` all learners will be plotted. If negative the bottom will be plotted.
            out: Indicate where the plot should be sent to after plotting is finished.
            ax: Provide an optional axes that the plot will be drawn to. If not provided a new figure/axes is created.
        """
        try:
            xlim = xlim or [None,None]
            ylim = ylim or [None,None]

            if isinstance(labels,str): labels = [labels]
            if isinstance(x,str): x = [x]

            self._validate_parameters(x)

            rows = FilterPlottingData().filter(self.interactions, x, y)
            rows = SmoothPlottingData().filter(rows, y, span)

            envs     = self.environments
            env_rows = { row[0]:dict(zip(envs.columns,row)) for row in envs }
            get_key  = lambda row: row['learner_id']
            lines    = []

            style = "-" if x == ['index'] else "."

            def get_color(colors:Union[None,Sequence[Union[str,int]]], i:int):
                try:
                    return colors[i] if colors else i
                except IndexError:
                    return i+max(colors) if isinstance(colors[0],(int,float)) else i
                except TypeError:
                    return i+colors

            def get_label(labels:Sequence[str], i:int, lrn_id:int=None):
                try:
                    return labels[i] if labels else self._full_name(lrn_id)
                except:
                    return self._full_name(lrn_id)

            for i, (lrn_id, lrn_rows) in enumerate(groupby(sorted(rows, key=get_key),key=get_key)):
                lrn_rows = list(lrn_rows)
                XYE      = TransformToXYE().filter(lrn_rows, env_rows, x, y, err)
                color    = get_color(colors,i)
                label    = get_label(labels,i,lrn_id)
                lines.append(Points(*zip(*XYE), color, 1, label, style))

            lines  = sorted(lines, key=lambda line: line[1][-1], reverse=True)
            labels = [l.label for l in lines]
            colors = [l.color for l in lines]
            xlabel = "Interaction" if x==['index'] else x[0] if len(x) == 1 else x
            ylabel = y.capitalize().replace("_pct"," Percent")

            title = ("Instantaneous" if span == 1 else f"Span {span}" if span else "Progressive") + f" {ylabel}"
            title = title + f" ({len(lrn_rows) if x==['index'] else len(XYE)} Environments)"

            if x != ['index']: title = f"Final {title}"

            xrotation = 90 if x != ['index'] and len(XYE)>5 else 0
            yrotation = 0

            if top_n:
                if abs(top_n) > len(lines): top_n = len(lines)*abs(top_n)/top_n
                if top_n > 0: lines = [l._replace(color=get_color(colors,i),label=get_label(labels,i)) for i,l in enumerate(lines[:top_n],0    ) ]
                if top_n < 0: lines = [l._replace(color=get_color(colors,i),label=get_label(labels,i)) for i,l in enumerate(lines[top_n:],top_n) ]

            self._plotter.plot(ax, lines, title, xlabel, ylabel, xlim, ylim, xticks, yticks, xrotation, yrotation, out)
        except CobaException as e:
            CobaContext.logger.log(str(e)) 

    def _full_name(self,lrn_id:int) -> str:
        """A user-friendly name created from a learner's params for reporting purposes."""

        cols = self.learners.columns
        vals = self.learners.row_vals[self.learners.col_vals[0].index(lrn_id)]

        values = dict((k,v) for k,v in zip(cols,vals) if v is not None)
        family = values.get('family',values['learner_id'])
        params = f"({','.join(f'{k}={v}' for k,v in values.items() if k not in ['family','learner_id'])})"

        return family if params == '()' else family+params

    def __str__(self) -> str:
        return str({"Learners": len(self._learners), "Environments": len(self._environments), "Interactions": len(self._interactions) })

    def _ipython_display_(self):
        #pretty print in jupyter notebook (https://ipython.readthedocs.io/en/stable/config/integrating.html)
        print(str(self))

    def _validate_parameters(self, x:Sequence[str]):
        if 'index' in x and len(x) > 1:
            raise CobaException('The x-axis cannot contain both interaction index and environment features. Please choose one or the other.')
