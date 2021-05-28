import math
import collections

from pathlib import Path
from itertools import chain, groupby, count, repeat
from typing import Any, Iterable, Dict, List, Tuple, Optional, Sequence, cast, Hashable, Iterator, Union

from coba.utilities import PackageChecker
from coba.statistics import OnlineMean, OnlineVariance
from coba.pipes import Filter, Cartesian, JsonEncode, JsonDecode, StopPipe, Pipe, DiskSink, DiskSource 

class Table:
    """A container class for storing tabular data."""

    def __init__(self, name:str, primary: Sequence[str]):
        """Instantiate a Table.
        
        Args:
            name: The name of the table.
            default: The default values to fill in missing values with
        """
        self._name    = name
        self._primary = primary
        self._cols    = collections.OrderedDict(zip(primary,repeat(None)))

        self._rows: Dict[Tuple[Hashable,...], Dict[str,Any]] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def columns(self) -> Sequence[str]:
        return self._cols.keys()

    def to_pandas(self) -> Any:
        PackageChecker.pandas("Table.to_pandas")
        import pandas as pd
        return pd.DataFrame(self.to_tuples(), columns=self.columns)

    def to_tuples(self) -> Sequence[Tuple[Any,...]]:
        return [ tuple([ row.get(col,float('nan')) for col in self.columns ]) for row in self]

    def _keyify(self, key: Union[Hashable, Sequence[Hashable]]) -> Tuple[Hashable,...]:
        seq_key = tuple([key] if isinstance(key,str) or not isinstance(key, collections.Sequence) else key)
        assert len(seq_key) == len(self._primary), "Incorrect primary key length given"
        return seq_key

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
        row.update(zip(self._primary, key))

        self._cols.update(zip(values.keys(), repeat(None)))
        self._rows[key] = row

    def __getitem__(self, key: Union[Hashable, Sequence[Hashable]]) -> Dict[str,Any]:
        return self._rows[self._keyify(key)]

    def __delitem__(self, key: Union[Hashable, Sequence[Hashable]]) -> None:
        del self._rows[self._keyify(key)]

class PackedTable:
    def __init__(self, name:str, primary: Sequence[str]) -> None:
        self._packed_rows = Table(name, primary)
        self._unpacked_sizes: Dict[Any,int] = {}

        #make sure the index column is the 3rd column
        self._packed_rows._cols['index'] = None

    @property
    def name(self) -> str:
        self._packed_rows.name

    @property
    def columns(self) -> Sequence[str]:
        return self._packed_rows.columns

    def to_pandas(self) -> Any:
        PackageChecker.pandas("Table.to_pandas")
        import pandas as pd
        return pd.DataFrame(self.to_tuples(), columns=self.columns)

    def to_tuples(self) -> Sequence[Tuple[Any,...]]:
        return [ tuple([ row.get(col,float('nan')) for col in self.columns ]) for row in self]

    def __len__(self) -> int:
        return sum(self._unpacked_sizes.values())

    def __str__(self) -> str:
        return str({"Table": self.name, "Columns": list(self.columns), "Rows": len(self)})

    def __iter__(self) -> Iterator[Dict[str,Any]]:
        for key in self._packed_rows._rows.keys():
            if len(key) == 1: 
                key=key[0]
            
            for row in self[key]:
                yield row

    def __contains__(self, key: Union[Hashable, Sequence[Hashable]]) -> bool:
        return key in self._packed_rows

    def __setitem__(self, key: Union[Hashable, Sequence[Hashable]], values: Dict[str,Any]):
        
        unpacked_size = 1
        
        for col in values:
            value       = values.get(col, float('nan'))
            is_singular = not isinstance(value,collections.Sequence) or isinstance(value,str)
            
            if is_singular: continue

            unpacked_size = len(value) if unpacked_size == 1 else unpacked_size 
            assert len(value) == unpacked_size, "When using a packed format all columns must be equal length or 1."

        row = values.copy()

        row['index'] = 1 if unpacked_size == 1 else list(range(1,unpacked_size+1))

        self._packed_rows[key] = row
        self._unpacked_sizes[key] = unpacked_size
    
    def __getitem__(self, key: Union[Hashable, Sequence[Hashable]]) -> Sequence[Dict[str,Any]]:
        unpacked_size = self._unpacked_sizes[key]

        if unpacked_size == 1: return [ self._packed_rows[key] ]

        unpacked_rows = [ {} for _ in range(unpacked_size) ]
        
        for col,value in self._packed_rows[key].items():

            if isinstance(value, collections.Sequence) and not isinstance(value,str):
                unpacked_value = value
            else:
                unpacked_value = [ value ] * unpacked_size

            for i,unpacked_row in enumerate(unpacked_rows):
                unpacked_row[col] = unpacked_value[i]

        return unpacked_rows

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

        #Warning, if you change the order of the primary keys old transaction files will break:
        #ResultPromote can be used to mitigate backwards compatability problems if this is necessary.
        self.interactions = PackedTable("Interactions", ['simulation_id', 'learner_id'])
        self.learners     = Table      ("Learners"    , ['learner_id'])
        self.simulations  = Table      ("Simulations" , ['simulation_id'])

    def plot_learners(self, select_learners: Sequence[int] = None, figsize=(12,4)) -> None:

        PackageChecker.matplotlib('Result.standard_plot')

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
        fig.legend(*ax1.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .3), ncol=1, fontsize='small') #type: ignore

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