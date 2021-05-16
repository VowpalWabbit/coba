import math
import collections

from pathlib import Path
from itertools import chain, groupby, count
from typing import Any, Iterable, Dict, List, Tuple, Optional, Sequence, cast, Hashable

from coba.tools import PackageChecker
from coba.statistics import OnlineMean, OnlineVariance
from coba.data.filters import Filter, Cartesian, JsonEncode, JsonDecode
from coba.data.pipes import StopPipe, Pipe
from coba.data.sinks import DiskSink
from coba.data.sources import DiskSource

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

        PackageChecker.pandas('Table.to_pandas')
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

class Result:
    """A class for creating and returning the result of a Benchmark evaluation."""

    @staticmethod
    def from_file(filename: Optional[str]) -> 'Result':
        """Create a Result from a transaction file."""
        
        if filename is None or not Path(filename).exists(): return Result()

        json_encode = Cartesian(JsonEncode())
        json_decode = Cartesian(JsonDecode())

        Pipe.join(DiskSource(filename), [json_decode, ResultPromote(), json_encode], DiskSink(filename, 'w')).run()
        
        return Result.from_transactions(Pipe.join(DiskSource(filename), [json_decode]).read())

    @staticmethod
    def from_transactions(transactions: Iterable[Any]) -> 'Result':

        result = Result()

        for transaction in transactions:
            result.add_transaction(transaction)

        return result

    def add_transaction(self, transaction: Any) -> None:
        if transaction[0] == "version"  : self.version = transaction[1]
        if transaction[0] == "benchmark": self.benchmark = transaction[1]
        if transaction[0] == "L"        : self.learners.add_row(transaction[1], **transaction[2])
        if transaction[0] == "S"        : self.simulations.add_row(transaction[1], **transaction[2])
        if transaction[0] == "B"        :
            for col in ["C", "A", "N"]:
                if "reward" in transaction[2] and col in transaction[2] and not isinstance(transaction[2][col], collections.Sequence):
                    transaction[2][col] = [transaction[2][col]] * len(transaction[2]["reward"])

            self.batches.add_row(*transaction[1], **transaction[2])

    def __init__(self) -> None:
        """Instantiate a Result class."""

        self.version     = None
        self.benchmark   = cast(Dict[str,Any],{})
        self.learners    = Table("Learners"   , ['learner_id'])
        self.simulations = Table("Simulations", ['simulation_id'])

        #Warning, if you change the order of the columns for batches then:
        # 1. TransactionLogPromote.current_version() will need to be bumped to version 3
        # 2. TransactionLogPromote.to_next_version() will need to promote version 2 to 3
        # 3. TransactionLog.write_batch will need to be modified to write in new order
        self.batches     = Table("Batches"    , ['simulation_id', 'learner_id'])

    def to_tuples(self) -> Tuple[Sequence[Any], Sequence[Any], Sequence[Any]]:
        return (
            self.learners.to_tuples(),
            self.simulations.to_tuples(),
            self.batches.to_tuples()
        )

    def to_indexed_tuples(self) -> Tuple[Dict[int,Any], Dict[int,Any], Dict[Tuple[int,int,Optional[int],int],Any]]:
        return (
            cast(Dict[int,Any], self.learners.to_indexed_tuples()),
            cast(Dict[int,Any], self.simulations.to_indexed_tuples()),
            cast(Dict[Tuple[int,int,Optional[int],int],Any], self.batches.to_indexed_tuples())
        )

    def to_pandas(self) -> Tuple[Any,Any,Any]:

        PackageChecker.pandas("Result.to_pandas")

        l = self.learners.to_pandas()
        s = self.simulations.to_pandas()
        b = self.batches.to_pandas()

        return (l,s,b)

    def standard_plot(self, select_learners: Sequence[int] = None,  show_err: bool = False, show_sd: bool = False, figsize=(12,4)) -> None:

        PackageChecker.matplotlib('Plots.standard_plot')

        def _plot(axes, label, xs, ys, vs, ns):
            axes.plot(xs, ys, label=label)

            if show_sd:
                ls = [ y-math.sqrt(v) for y,v in zip(ys,vs) ]
                us = [ y+math.sqrt(v) for y,v in zip(ys,vs) ]
                axes.fill_between(xs, ls, us, alpha = 0.1)

            if show_err:
                # I don't really understand what this is... For each x our distribution
                # is changing so its VAR is also changing. What does it mean to calculate
                # sample variance from a deterministic collection of random variables with
                # different distributions? For example sample variance of 10 random variables
                # from dist1 and 10 random variables from dist2... This is not the same as 20
                # random variables with 50% chance drawing from dist1 and 50% chance of drawing
                # from dist2. So the distribution can only be defined over the whole space (i.e.,
                # all 20 random variables) and not for a specific random variable. Oh well, for
                # now I'm leaving this as it is since I don't have any better ideas. I think what
                # I've done is ok, but I need to more some more thought into it.
                ls = [ y-math.sqrt(v/n) for y,v,n in zip(ys,vs,ns) ]
                us = [ y+math.sqrt(v/n) for y,v,n in zip(ys,vs,ns) ]
                axes.fill_between(xs, ls, us, alpha = 0.1)

        learners, _, batches = self.to_indexed_tuples()

        learners = {key:value for key,value in learners.items() if select_learners is None or key in select_learners}
        batches  = {key:value for key,value in batches.items() if select_learners is None or value.learner_id in select_learners}

        sorted_batches  = sorted(batches.values(), key=lambda batch: batch.learner_id)
        grouped_batches = groupby(sorted_batches , key=lambda batch: batch.learner_id)

        max_batch_N = 0

        indexes     = cast(Dict[int,List[int  ]], collections.defaultdict(list))
        incounts    = cast(Dict[int,List[int  ]], collections.defaultdict(list))
        inmeans     = cast(Dict[int,List[float]], collections.defaultdict(list))
        invariances = cast(Dict[int,List[float]], collections.defaultdict(list))
        cucounts    = cast(Dict[int,List[int  ]], collections.defaultdict(list))
        cumeans     = cast(Dict[int,List[float]], collections.defaultdict(list))
        cuvariances = cast(Dict[int,List[float]], collections.defaultdict(list))

        for learner_id, learner_batches in grouped_batches:

            cucount    = 0
            cumean     = OnlineMean()
            cuvariance = OnlineVariance()

            Ns, Rs = list(zip(*[ (b.N, b.reward) for b in learner_batches ]))

            Ns = list(zip(*Ns))
            Rs = list(zip(*Rs))

            for batch_index, batch_Ns, batch_Rs in zip(count(), Ns,Rs):

                incount    = 0
                inmean     = OnlineMean()
                invariance = OnlineVariance()

                for N, reward in zip(batch_Ns, batch_Rs):
                    
                    max_batch_N = max(N, max_batch_N)
                    
                    incount     = incount + 1
                    inmean      .update(reward)
                    invariance  .update(reward)
                    cucount     = cucount + 1
                    cumean      .update(reward)
                    cuvariance  .update(reward)

                #sanity check, sorting above (in theory) should take care of this...
                #if this isn't the case then the cu* values will be incorrect...
                assert indexes[learner_id] == [] or batch_index > indexes[learner_id][-1]

                incounts[learner_id].append(incount)
                indexes[learner_id].append(batch_index)
                inmeans[learner_id].append(inmean.mean)
                invariances[learner_id].append(invariance.variance)
                cucounts[learner_id].append(cucount)
                cumeans[learner_id].append(cumean.mean)
                cuvariances[learner_id].append(cuvariance.variance)

        import matplotlib.pyplot as plt #type: ignore

        fig = plt.figure(figsize=figsize)

        index_unit = "Interactions" if max_batch_N ==1 else "Batches"
        
        ax1 = fig.add_subplot(1,2,1) #type: ignore
        ax2 = fig.add_subplot(1,2,2) #type: ignore

        for learner_id in learners:
            _plot(ax1, learners[learner_id].full_name, indexes[learner_id], inmeans[learner_id], invariances[learner_id], incounts[learner_id])

        ax1.set_title(f"Instantaneous Reward")
        ax1.set_ylabel("Reward")
        ax1.set_xlabel(f"{index_unit}")

        for learner_id in learners:
            _plot(ax2, learners[learner_id].full_name, indexes[learner_id], cumeans[learner_id], cuvariances[learner_id], cucounts[learner_id])

        ax2.set_title("Progressive Reward")
        #ax2.set_ylabel("Reward")
        ax2.set_xlabel(f"{index_unit}")

        (bot1, top1) = ax1.get_ylim()
        (bot2, top2) = ax2.get_ylim()

        ax1.set_ylim(min(bot1,bot2), max(top1,top2))
        ax2.set_ylim(min(bot1,bot2), max(top1,top2))

        scale = 0.5
        box1 = ax1.get_position()
        box2 = ax2.get_position()
        ax1.set_position([box1.x0, box1.y0 + box1.height * (1-scale), box1.width, box1.height * scale])
        ax2.set_position([box2.x0, box2.y0 + box2.height * (1-scale), box2.width, box2.height * scale])

        # Put a legend below current axis
        fig.legend(*ax1.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .3), ncol=2, fontsize='small') #type: ignore

        plt.show()

    def __str__(self) -> str:
        return str({ "Learners": len(self.learners), "Simulations": len(self.simulations), "Interactions": sum([len(b.reward) for b in self.batches.to_tuples()]) })

    def __repr__(self) -> str:
        return str(self)

class ResultPromote(Filter):

    CurrentVersion = 2

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        items_iter = iter(items)
        items_peek = next(items_iter)
        items_iter = chain([items_peek], items_iter)

        version = 0 if items_peek[0] != 'version' else items_peek[1]

        if version == ResultPromote.CurrentVersion:
            raise StopPipe()

        while version != ResultPromote.CurrentVersion:
            if version == 0:
                promoted_items = [["version",1]]

                for transaction in items:

                    if transaction[0] == "L":

                        index  = transaction[1][1]['learner_id']
                        values = transaction[1][1]

                        del values['learner_id']

                        promoted_items.append([transaction[0], index, values])

                    if transaction[0] == "S":

                        index  = transaction[1][1]['simulation_id']
                        values = transaction[1][1]

                        del values['simulation_id']

                        promoted_items.append([transaction[0], index, values])

                    if transaction[0] == "B":
                        key_columns = ['learner_id', 'simulation_id', 'seed', 'batch_index']
                        
                        index  = [ transaction[1][1][k] for k in key_columns ]
                        values = transaction[1][1]
                        
                        for key_column in key_columns: del values[key_column]
                        
                        if 'reward' in values:
                            values['reward'] = values['reward'].estimate
                        
                        if 'mean_reward' in values:
                            values['reward'] = values['mean_reward'].estimate
                            del values['mean_reward']

                        values['reward'] = round(values['reward', 5])

                        promoted_items.append([transaction[0], index, values])

                items   = promoted_items
                version = 1

            if version == 1:

                n_seeds       : Optional[int]                  = None
                S_transactions: Dict[int, Any]                 = {}
                S_seeds       : Dict[int, List[Optional[int]]] = collections.defaultdict(list)

                B_rows: Dict[Tuple[int,int], Dict[str, List[float]] ] = {}
                B_cnts: Dict[int, int                               ] = {}

                promoted_items = [["version",2]]

                for transaction in items:

                    if transaction[0] == "benchmark":
                        n_seeds = transaction[1].get('n_seeds', None)

                        del transaction[1]['n_seeds']
                        del transaction[1]['batcher']
                        del transaction[1]['ignore_first']

                        promoted_items.append(transaction)

                    if transaction[0] == "L":
                        promoted_items.append(transaction)

                    if transaction[0] == "S":
                        S_transactions[transaction[1]] == transaction

                    if transaction[0] == "B":
                        S_id = transaction[1][1]
                        seed = transaction[1][2]
                        L_id = transaction[1][0]
                        B_id = transaction[1][3]
                        
                        if n_seeds is None:
                            raise StopPipe("We are unable to promote logs from version 1 to version 2")

                        if seed not in S_seeds[S_id]:
                            S_seeds[S_id].append(seed)
                            
                            new_S_id = n_seeds * S_id + S_seeds[S_id].index(seed)
                            new_dict = S_transactions[S_id][2].clone()
                            
                            new_dict["source"]  = str(S_id)
                            new_dict["filters"] = f'[{{"Shuffle":{seed}}}]'

                            B_cnts[S_id] = new_dict['batch_count']

                            promoted_items.append(["S", new_S_id, new_dict])

                        if B_id == 0: B_rows[(S_id, L_id)] = {"N":[], "reward":[]}

                        B_rows[(S_id, L_id)]["N"     ].append(transaction[2]["N"])
                        B_rows[(S_id, L_id)]["reward"].append(transaction[2]["reward"])

                        if len(B_rows[(S_id, L_id)]["N"]) == B_cnts[S_id]:
                            promoted_items.append(["B", [S_id, L_id], B_rows[(S_id, L_id)]])
                            del B_rows[(S_id, L_id)]

                items   = promoted_items
                version = 2

        return items