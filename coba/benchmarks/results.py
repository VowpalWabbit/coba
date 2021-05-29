import math
import collections

from operator import truediv
from pathlib import Path
from statistics import mean, stdev
from itertools import chain, repeat, product, accumulate
from typing import Any, Iterable, Dict, List, Tuple, Optional, Sequence, Hashable, Iterator, Union, Callable

from coba.utilities import PackageChecker
from coba.pipes import Filter, Cartesian, JsonEncode, JsonDecode, StopPipe, Pipe, DiskSink, DiskSource 

class Table:
    """A container class for storing tabular data."""

    def __init__(self, name:str, primary: Sequence[str], types: Dict[str,Any]={}):
        """Instantiate a Table.
        
        Args:
            name: The name of the table.
            default: The default values to fill in missing values with
        """
        self._name    = name
        self._primary = primary
        self._cols    = collections.OrderedDict(zip(primary,repeat(None)))
        self._types   = types

        self._rows: Dict[Tuple[Hashable,...], Dict[str,Any]] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def columns(self) -> Sequence[str]:
        return list(self._cols.keys())

    def to_pandas(self) -> Any:
        PackageChecker.pandas("Table.to_pandas")
        import pandas as pd
        return pd.DataFrame(self.to_tuples(), columns=self.columns)

    def to_tuples(self) -> Iterable[Tuple[Any,...]]:
        return [ tuple( row.get(col,float('nan')) for col in self.columns) for row in self ]

    def _keyify(self, key: Union[Hashable, Sequence[Hashable]]) -> Tuple[Hashable,...]:
        key_len = len(key) if isinstance(key,(list,tuple)) else 1
        
        assert key_len == len(self._primary), "Incorrect primary key length given"
        
        return tuple(key) if isinstance(key,(list,tuple)) else key

    def __iter__(self) -> Iterator[Dict[str,Any]]:
        for row in self._rows.values(): yield row

    def __contains__(self, key: Union[Hashable, Sequence[Hashable]]) -> bool:
        return self._keyify(key) in self._rows

    def __str__(self) -> str:
        return str({"Table": self.name, "Columns": self.columns, "Rows": len(self)})

    def __len__(self) -> int:
        return len(self._rows)

    def __setitem__(self, key: Union[Hashable, Sequence[Hashable]], values: Dict[str,Any]):

        key = self._keyify(key)

        row = values.copy()        
        row.update(zip(self._primary, key if isinstance(key,tuple) else [key]))

        self._cols.update(zip(values.keys(), repeat(None)))
        self._rows[key] = row

    def __getitem__(self, key: Union[Hashable, Sequence[Hashable]]) -> Dict[str,Any]:
        return self._rows[self._keyify(key)]

    def __delitem__(self, key: Union[Hashable, Sequence[Hashable]]) -> None:
        del self._rows[self._keyify(key)]

class PackedTable:
    def __init__(self, name:str, primary: Sequence[str], types: Dict[str,Any] = {}) -> None:
        self._packed_rows = Table(name, primary)
        self._unpacked_sizes: Dict[Any,int] = {}
        self._types = types

        #make sure the index column is the 3rd column
        self._packed_rows._cols['index'] = None
        self._types['index'] = int

    @property
    def name(self) -> str:
        self._packed_rows.name

    @property
    def columns(self) -> Sequence[str]:
        return self._packed_rows.columns

    def to_pandas(self) -> Any:
        PackageChecker.pandas("Table.to_pandas")
        import pandas as pd
        import numpy as np

        col_values = {col:np.empty(len(self),dtype=self._types.get(col,object)) for col in self.columns}
        index = 0

        for key in self._packed_rows._rows:

            unpacked_size = self._unpacked_sizes[key]

            for col in self.columns:
                val = self._packed_rows[key].get(col,float('nan'))
                col_values[col][index:(index+unpacked_size)] = val

            index += unpacked_size

        return pd.DataFrame(col_values, columns=self.columns)

    def to_tuples(self) -> Sequence[Tuple[Any,...]]:
        items = []
        for key,packed in zip(self._packed_rows._rows.keys(), self._packed_rows.to_tuples()):
            unpacked_size = self._unpacked_sizes[key]
            for unpacked in zip(*(v if isinstance(v,list) else repeat(v,unpacked_size) for v in packed)):
                items.append(unpacked)
        return items

    def __len__(self) -> int:
        return sum(self._unpacked_sizes.values())

    def __str__(self) -> str:
        return str({"Table": self.name, "Columns": list(self.columns), "Rows": len(self)})

    def __iter__(self) -> Iterator[Dict[str,Any]]:
        return self._packed_rows

    def __contains__(self, key: Union[Hashable, Sequence[Hashable]]) -> bool:
        return key in self._packed_rows

    def __setitem__(self, key: Union[Hashable, Sequence[Hashable]], values: Dict[str,Any]):
        
        unpacked_size = 1
        
        for value in values.values():
            if not isinstance(value,list): continue
            unpacked_size = len(value) if unpacked_size == 1 else unpacked_size 
            assert len(value) == unpacked_size, "When using a packed format all columns must be equal length or 1."

        row = values.copy()
        row['index'] = 1 if unpacked_size == 1 else list(range(1,unpacked_size+1))

        self._packed_rows[key]    = row
        self._unpacked_sizes[key] = unpacked_size
    
    def __getitem__(self, key: Union[Hashable, Sequence[Hashable]]) -> Dict[str,Any]:
        return self._packed_rows[key]

    def __delitem__(self, key: Union[Hashable, Sequence[Hashable]]) -> None:
        del self._packed_rows[key]

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

        for trx in transactions:
            if trx[0] == "version"  : result.version   = trx[1]
            if trx[0] == "benchmark": result.benchmark = trx[1]
            if trx[0] == "L"        : result.learners    [trx[1]       ] = trx[2]
            if trx[0] == "S"        : result.simulations [trx[1]       ] = trx[2]
            if trx[0] == "B"        : result.interactions[tuple(trx[1])] = trx[2]

        return result

    def __init__(self) -> None:
        """Instantiate a Result class."""

        self.version    : int            = None
        self.benchmark  : Dict[str, Any] = {}

        #providing the types in advance makes to_pandas about 10 times faster since we can preallocate space
        self.interactions = PackedTable("Interactions", ['simulation_id', 'learner_id'], types={'simulation_id':int, 'learner_id':int, 'C':int,'A':int,'N':int,'reward':float})
        self.learners     = Table      ("Learners"    , ['learner_id'])
        self.simulations  = Table      ("Simulations" , ['simulation_id'])

    def plot_learners(self, 
        sim_pipes: Sequence[str] = [""], 
        lrn_names: Sequence[str] = [""], 
        span=None,
        show_sd=True,
        show_se=True,
        figsize=(8,6)) -> None:

        PackageChecker.matplotlib('Result.standard_plot')

        learner_ids    = [ l["learner_id"] for l in self.learners if any(name in l["full_name"] for name in lrn_names ) ]
        simulation_ids = [ s["simulation_id"] for s in self.simulations if any(pipe in s['pipe'] for pipe in sim_pipes) ]
        
        max_batch_N = 1
        progressives: Dict[int,List[Sequence[float]]] = collections.defaultdict(list)

        for simulation_id, learner_id in product(simulation_ids,learner_ids):
            
            if (simulation_id,learner_id) not in self.interactions: continue

            rewards = self.interactions._packed_rows[(simulation_id,learner_id)]["reward"]

            if span is None or span >= len(rewards):
                cumwindow  = list(accumulate(rewards))
                cumdivisor = list(range(1,len(cumwindow)+1))
            
            elif span == 1:
                cumwindow  = list(rewards)
                cumdivisor = [1]*len(cumwindow)

            else:
                cumwindow  = list(accumulate(rewards))
                cumwindow  = cumwindow + [0] * span
                cumwindow  = [ cumwindow[i] - cumwindow[i-span] for i in range(len(cumwindow)-span) ]
                cumdivisor = list(range(1, span)) + [span]*(len(cumwindow)-span+1)

            progressives[learner_id].append(list(map(truediv, cumwindow, cumdivisor)))

        import matplotlib.pyplot as plt #type: ignore

        fig = plt.figure(figsize=figsize)
        
        ax = fig.add_subplot(1,1,1) #type: ignore

        stder = lambda values: stdev(values)/math.sqrt(len(values))

        for learner_id in learner_ids:
            label = self.learners[learner_id]["full_name"] 
            Y     = list(map(mean ,zip(*progressives[learner_id])))
            X     = list(range(1,len(Y)+1))

            ax.plot(X, Y,label=label)

            if show_sd and len(progressives[learner_id]) > 1:                
                sd = list(map(stdev,zip(*progressives[learner_id])))
                ls = [ y-s for y,s in zip(Y,sd) ]
                us = [ y+s for y,s in zip(Y,sd) ]
                ax.fill_between(X, ls, us, alpha = 0.1)

            if show_se and len(progressives[learner_id]) > 1:
                se = list(map(stder,zip(*progressives[learner_id])))
                ls = [ y-s for y,s in zip(Y,se) ]
                us = [ y+s for y,s in zip(Y,se) ]
                ax.fill_between(X, ls, us, alpha = 0.1)

        ax.set_title ("Instantaneous" if span == 1 else "Progressive" if span is None else f"Span {span}" + " Reward")
        ax.set_ylabel("Reward")
        ax.set_xlabel("Interactions" if max_batch_N ==1 else "Batches")

        #make room for the legend
        scale = 0.65
        box1 = ax.get_position()
        ax.set_position([box1.x0, box1.y0 + box1.height * (1-scale), box1.width, box1.height * scale])

        # Put a legend below current axis
        fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .3), ncol=1, fontsize='small') #type: ignore

        plt.show()

    def __str__(self) -> str:
        return str({ "Learners": len(self.learners), "Simulations": len(self.simulations), "Interactions": len(self.interactions) })

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