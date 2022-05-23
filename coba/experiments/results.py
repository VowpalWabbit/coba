import re
import collections
import collections.abc

from copy import copy
from pathlib import Path
from numbers import Number
from statistics import mean
from operator import truediv
from abc import abstractmethod
from itertools import chain, repeat, accumulate
from typing import Any, Dict, List, Tuple, Optional, Sequence, Hashable, Iterator, Union, Type, Set, Callable
from coba.backports import Literal

from coba.contexts import CobaContext
from coba.exceptions import CobaException
from coba.utilities import PackageChecker
from coba.pipes import Pipes, Sink, Source, JsonEncode, JsonDecode, DiskSource, DiskSink, ListSource, ListSink, Foreach

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

    def to_progressive_dicts(self,
        span: int = None,
        each: bool = False,
        order: str = "index",
        value: Literal['reward','reward_pct','rank','rank_pct','regret','regret_pct'] = "reward"):
        """Return expanding or exponential averages for col grouped by learner and possibly environment.

        Args:
            span: If span is None return an expanding average (i.e., progressive validation). If span is not none
                calculate a simple moving average with window size of span (window will be smaller than span initially).
            each: If true the group by learner and environment (as in each environment). If each is false
                then only group by learner.
            ord_col: The column which indicates the order in which averaging is calculated on the val_col.
            val_col: The column we wish to calculate the progressive average values for.

        Returns:
            Either [[learner_id, col progressive means...],...] or [[learner_id, environment_id, col progressive means...],...].
        """
        lrn_env_rows = []

        value_functions = {
            'reward_pct': lambda interactions: [r/m for r,m in zip(interactions['reward'],interactions['max_reward'])],
            'rank_pct'  : lambda interactions: [(r-1)/(m-1) for r,m in zip(interactions['rank'],interactions['max_rank'])],
            'regret'    : lambda interactions: [m-r for r,m in zip(interactions['reward'],interactions['max_reward'])],
            'regret_pct': lambda interactions: [(u-r)/(u-l) for r,l,u in zip(interactions['reward'],interactions['min_reward'],interactions['max_reward'])],
        }

        for interactions in self:

            values = value_functions.get(value, lambda interactions: interactions[value])(interactions)
            orders = interactions[order]
            values = [ v[1] for v in sorted(zip(orders,values)) ]

            if span is None or span >= len(values):
                cumwindow  = list(accumulate(values))
                cumdivisor = list(range(1,len(cumwindow)+1))

            elif span == 1:
                cumwindow  = list(values)
                cumdivisor = [1]*len(cumwindow)

            else:
                moving_sum = 0
                cumwindow  = []
                cumdivisor = []
                for index in range(len(values)):
                    sub_i = index-span
                    add_i = index

                    moving_sum += values[add_i]
                    moving_sum -= values[sub_i] if sub_i >= 0 else 0

                    cumwindow.append(moving_sum)
                    cumdivisor.append(min(span,add_i+1))

                #efficient way to calucate exponential moving average identical to Pandas df.ewm(span=span).mean()
                #alpha = 2/(1+span)
                #cumwindow  = list(accumulate(values          , lambda a,c: c + (1-alpha)*a))
                #cumdivisor = list(accumulate([1.]*len(values), lambda a,c: c + (1-alpha)*a)) #type: ignore

            lrn_env_rows.append( {
                "learner_id"    : interactions["learner_id"], 
                "environment_id": interactions["environment_id"], 
                "values"        : list(map(truediv, cumwindow, cumdivisor))
            })

        if each:
            return lrn_env_rows

        else:
            grouped_lrn_sim_rows = collections.defaultdict(list)

            for row in lrn_env_rows:
                grouped_lrn_sim_rows[row["learner_id"]].append(row["values"])

            lrn_rows = []

            for learner_id in grouped_lrn_sim_rows.keys():

                Z = list(zip(*grouped_lrn_sim_rows[learner_id]))

                if not Z: continue

                Y = [ sum(z)/len(z) for z in Z ]

                lrn_rows.append({"learner_id":learner_id, "values": Y})

            return lrn_rows

    def to_progressive_pandas(self, span: int = None, each: bool = False, ord_col="index", val_col: str = "reward"):
        """Return expanding or exponential averages for yaxis grouped by learner and possibly environment.

        Args:
            span: If span is None return an expanding average (i.e., progressive validation). If span is not none
                calculate a simple moving average with window size of span (window will be smaller than span initially).
            each: If true the group by learner and environment (as in each environment). If each is false
                then only group by learner.
            ord_col: The column which indicates the order in which averaging is calculated on the val_col.
            val_col: The column we wish to calculate the progressive average values for.

        Returns:
            A data frame whose columns are (learner_id, [environment_id], interaction indexes...).
        """

        PackageChecker.pandas("Result.to_pandas")

        import pandas as pd

        data = []

        for d in self.to_progressive_dicts(span, each, ord_col, val_col):
            if each:
                data.append([d["learner_id"], d["environment_id"], *d["values"]])
            if not each:
                data.append([d["learner_id"], *d["values"]])

        if each:
            n_index = len(data[0][2:])
            return pd.DataFrame(data, columns=["learner_id", "environment_id", *range(1,n_index+1)])

        else:
            n_index = len(data[0][1:])
            return pd.DataFrame(data, columns=["learner_id", *range(1,n_index+1)])

class TransactionIO_V3(Source['Result'], Sink[Any]):

    def __init__(self, log_file: Optional[str] = None, minify:bool = True) -> None:

        self._log_file = log_file
        self._minify   = minify
        self._source   = DiskSource(log_file) if log_file else ListSource()
        self._sink     = DiskSink(log_file) if log_file else ListSink(self._source.items)

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

        if isinstance(self._source, ListSource):
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
        self._source   = DiskSource(log_file) if log_file else ListSource()
        self._sink     = DiskSink(log_file)   if log_file else ListSink(self._source.items)

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

        if isinstance(self._source, ListSource):
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

    def write(self, transaction: Any) -> None:
        self._transactionIO.write(transaction)

    def read(self) -> 'Result':
        return self._transactionIO.read()

class Plotter:
    @abstractmethod
    def plot(self,
        ax,
        lines: Sequence[Tuple[Sequence[float], Sequence[float], Optional[Sequence[float]], str, float, Optional[str]]],
        title: str,
        xlabel: str,
        ylabel: str,
        xlim: Optional[Tuple[Number,Number]],
        ylim: Optional[Tuple[Number,Number]],
        filename: Optional[str]) -> None:
        pass

class MatplotlibPlotter(Plotter):
    
    def plot(self,
        ax,
        lines: Sequence[Tuple[Sequence[float], Sequence[float], Optional[Sequence[float]], str, float, Optional[str]]],
        title: str,
        xlabel: str,
        ylabel: str,
        xlim: Optional[Tuple[Number,Number]],
        ylim: Optional[Tuple[Number,Number]],
        filename: Optional[str]
    ) -> None:

        PackageChecker.matplotlib('Result.plot_learners')
        import matplotlib.pyplot as plt #type: ignore

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        ax: plt.Axes
        show = ax is None

        bad_xlim  = xlim and xlim[0] is not None and xlim[1] is not None and xlim[0] >= xlim[1]
        bad_ylim  = ylim and ylim[0] is not None and ylim[1] is not None and ylim[0] >= ylim[1]
        bad_lines = not lines

        if bad_xlim or bad_ylim or bad_lines:
            if bad_xlim:
                CobaContext.logger.log("The xlim end is less than the xlim start. Plotting is impossible.")
            if bad_ylim:
                CobaContext.logger.log("The ylim end is less than the ylim start. Plotting is impossible.")
            if bad_lines:
                CobaContext.logger.log(f"No data was found for plotting in the given results.")
        else:
            ax = ax or plt.figure(figsize=(10,4)).add_subplot(111) #type: ignore

            in_lim = lambda v,lim: lim is None or ((lim[0] or -float('inf')) <= v and v <= (lim[1] or float('inf')))

            for X, Y, E, c, a, l in lines:

                # we remove values outside of the given lims because matplotlib won't correctly scale otherwise
                if not E:
                    XY = [(x,y) for x,y in zip(X,Y) if in_lim(x,xlim) and in_lim(y,ylim)]
                    X,Y = map(list,zip(*XY)) if XY else ([],[])
                else:
                    XYE = [(x,y,e) for x,y,e in zip(X,Y,E) if in_lim(x,xlim) and in_lim(y,ylim)]
                    X,Y,E = map(list,zip(*XYE)) if XYE else ([],[],[])

                if isinstance(c,int): c = color_cycle[c]

                if X and Y:
                    if E is None:
                        ax.plot(X, Y, color=c, alpha=a, label=l)
                    else:
                        ax.errorbar(X, Y, yerr=E, elinewidth=0.5, errorevery=(0,max(int(len(X)*0.05),1)), color=c, alpha=a, label=l)

            padding = .05
            ax.margins(0)
            ax.set_xticks([min(ax.get_xlim()[1], max(ax.get_xlim()[0],x)) for x in ax.get_xticks()])
            ax.margins(padding)

            if xlim:
                x_pad = padding*(xlim[1]-xlim[0])
                ax.set_xlim(xlim[0]-x_pad, xlim[1]+x_pad)

            if ylim:
                y_pad = padding*(ylim[1]-ylim[0])
                ax.set_ylim(ylim[0]-y_pad, ylim[1]+y_pad)

            ax.set_title(title, loc='left', pad=15)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)

            if ax.get_legend() is None: #pragma: no cover
                scale = 0.65
                box1 = ax.get_position()
                ax.set_position([box1.x0, box1.y0 + box1.height * (1-scale), box1.width, box1.height * scale])
            else:
                ax.get_legend().remove()

            ax.legend(*ax.get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(-.01, -.25), ncol=1, fontsize='medium') #type: ignore

            if show:
                plt.show()
                plt.close()

            if filename:
                plt.tight_layout()
                plt.savefig(filename, dpi=300)
                plt.close()

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

    def plot_learners(self,
        y    : Literal['reward','reward_pct','rank','rank_pct','regret','regret_pct'] = "reward",
        xlim : Optional[Tuple[Number,Number]] = None,
        ylim : Optional[Tuple[Number,Number]] = None,
        span : int = None,
        err  : Optional[Literal['se','sd']] = None,
        each : bool = False,
        filename: str = None,
        sort : Literal["name","id","y"] = "y",
        labels: Sequence[str] = [],
        ax = None) -> None:
        """Plot the performance of multiple learners on multiple environments. It gives a sense of the expected
            performance for different learners across independent environments. This plot is valuable in gaining
            insight into how various learners perform in comparison to one another.

        Args:
            y: The value to plot on the y-axis: `reward` plots the reward obtained on each interaction, `pct_reward`
                plots the percentage of the optimal possible reward obtained on each interaction.
            xlim: Define the x-axis limits to plot. If `None` the x-axis limits will be inferred.
            ylim: Define the y-axis limits to plot. If `None` the y-axis limits will be inferred.
            span: If span is None return an expanding average (i.e., progressive validation). If span is not none
                calculate a simple moving average with window size of span (window will be smaller than span initially).
            err: This determines what kind of error bars to plot (if any). Valid types are `None`, 'se', and 'sd'. If `None`
                then no bars are plotted, if 'se' the standard error is shown, and if 'sd' the standard deviation is shown.
            each: This determines whether each evaluated environment used to estimate mean performance is also plotted.
            filename: Provide a filename to write plot image to disk.
            sort: Indicate how learner names should be sorted in the legend.
            labels: The legend labels to use in the plot. These should be in order of the actual legend labels.
            ax: Provide an optional axes that the plot will be drawn to. If not provided a new figure/axes is created.
        """

        n_envs = float('inf')
        given_labels = labels

        lines = []

        for i, (label, X, Y, E, Z) in enumerate(self._plot_learners_data(y,span,err,sort)):

            n_envs = min(n_envs, len(Z[0]))
            color  = i
            label  = given_labels[i] if i < len(given_labels) else label

            lines.append( (X, Y, E, color, 1, label) )

            if each: 
                lines.extend(zip(repeat(X), map(list,zip(*Z)), repeat(None), repeat(color), repeat(0.15), repeat(None)))

        xlabel = "Interactions"
        ylabel = y.capitalize().replace("_pct"," Percent")
        title  = ("Instantaneous" if span == 1 else f"Span {span}" if span else "Progressive") + f" {ylabel} ({n_envs} Environments)"

        self._plotter.plot(ax, lines, title, xlabel, ylabel, xlim, ylim, filename)

    def _plot_learners_data(self,
        y   : Literal['reward','reward_pct','rank','rank_pct','regret','regret_pct'] = "reward",
        span: int = None,
        err : Literal['se','sd'] = None,
        sort: Literal['name',"id","y"] = "y"):

        prog_by_lrn: Dict[int,List[Sequence[float]]] = collections.defaultdict(list)
        prog_by_lrn_env = self.interactions.to_progressive_dicts(span=span, each=True, value=y)

        #TODO: only include env_ids that are present for all learners
        lrn_id_count   = len(set([ d['learner_id'] for d in prog_by_lrn_env ]))
        env_id_counts  = collections.Counter([ d['environment_id'] for d in prog_by_lrn_env ])
        val_len_counts = collections.Counter([ len(d['values'])    for d in prog_by_lrn_env ])

        if len(set(env_id_counts.values())) > 1:
            CobaContext.logger.log("This result contains environments not present for all learners. Environments not present for all learners have been excluded. To supress this warning in the future call <result>.filter_fin() before plotting.") 

        if len(val_len_counts) > 1:
            CobaContext.logger.log("The result contains environments of different lengths. The plot only includes data which is present in all environments. To only plot environments with a minimum number of interactions call <result>.filter_fin(n_interactions).")

        for progressive in prog_by_lrn_env:
            if env_id_counts[progressive['environment_id']] == lrn_id_count:
                prog_by_lrn[progressive["learner_id"]].append(progressive["values"])

        if prog_by_lrn:
            if sort == "name":
                sort_func = lambda id: self.learners[id]["full_name"]

            if sort == "y":
                sort_func = lambda id: -sum(list(zip(*prog_by_lrn[id]))[-1])

            if sort == "id":
                sort_func = lambda id: id

            for learner_id in sorted(self.learners.keys, key=sort_func):

                label = self._learners[learner_id]["full_name"]
                Z     = list(zip(*prog_by_lrn[learner_id]))

                if not Z or not Z[0]: continue

                Y = [ sum(z)/len(z) for z in Z ]
                X = list(range(1,len(Y)+1))
                E = self._yerr(Z, err)

                yield label, X, Y, E, Z

    def _yerr(self, Z: Sequence[Sequence[float]], err: Literal['sd','se']):
        
        err = err.lower() if err else err
        
        N = [ len(z) for z in Z        ]
        Y = [ sum(z)/len(z) for z in Z ]

        #this is much faster than python's native stdev
        #and more or less free computationally so we always
        #calculate it regardless of if they are showing them
        #we are using the identity Var[Y] = E[Y^2]-E[Y]^2
        
        Y2 = [ sum([zz**2 for zz in z])/len(z) for z in Z            ]
        SD = [ (round(y2-y**2,8))**(1/2)       for y2,y in zip(Y2,Y) ]
        SE = [ sd/(n**(1/2))                   for sd,n in zip(SD,N) ]

        return SE if err == 'se' else SD if err == 'sd' else None

    def __str__(self) -> str:
        return str({ "Learners": len(self._learners), "Environments": len(self._environments), "Interactions": len(self._interactions) })

    def _ipython_display_(self):
        #pretty print in jupyter notebook (https://ipython.readthedocs.io/en/stable/config/integrating.html)
        print(str(self))
