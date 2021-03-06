import math
import collections

from pathlib import Path
from itertools import chain, groupby, count
from typing import Any, Iterable, Dict, List, Tuple, Optional, Sequence, cast

from coba.learners import Learner
from coba.tools import PackageChecker
from coba.statistics import OnlineMean, OnlineVariance
from coba.data.structures import Table
from coba.data.filters import Filter, Cartesian, JsonEncode, JsonDecode
from coba.data.pipes import StopPipe, Pipe
from coba.data.sinks import Sink, DiskSink, MemorySink
from coba.data.sources import DiskSource

class Result:
    """A class for creating and returning the result of a Benchmark evaluation."""

    @staticmethod
    def from_file(filename: Optional[str]) -> 'Result':
        """Create a Result from a transaction file."""
        
        if filename is None or not Path(filename).exists(): return Result()

        json_encode = Cartesian(JsonEncode())
        json_decode = Cartesian(JsonDecode())

        Pipe.join(DiskSource(filename), [json_decode, TransactionPromote(), json_encode], DiskSink(filename, 'w')).run()
        
        return Result.from_transactions(Pipe.join(DiskSource(filename), [json_decode]).read())

    @staticmethod
    def from_transactions(transactions: Iterable[Any]) -> 'Result':

        result = Result()

        for transaction in transactions:
            if transaction[0] == "version"  : result.version = transaction[1]
            if transaction[0] == "benchmark": result.benchmark = transaction[1]
            if transaction[0] == "L"        : result.learners.add_row(transaction[1], **transaction[2])
            if transaction[0] == "S"        : result.simulations.add_row(transaction[1], **transaction[2])
            if transaction[0] == "B"        : result.batches.add_row(*transaction[1], **transaction[2])

        return result

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
        return str({ "Learners": len(self.learners), "Simulations": len(self.simulations), "Interactions": sum([len(b.N) for b in self.batches.to_tuples()]) })

    def __repr__(self) -> str:
        return str(self)

class Transaction:

    @staticmethod
    def version(version = None) -> Any:
        return ['version', version or TransactionPromote.CurrentVersion]

    @staticmethod
    def benchmark(n_learners, n_simulations) -> Any:
        data = {
            "n_learners"   : n_learners,
            "n_simulations": n_simulations,
        }

        return ['benchmark',data]

    @staticmethod
    def learner(learner_id:int, **kwargs) -> Any:
        """Write learner metadata row to Result.
        
        Args:
            learner_id: The primary key for the given learner.
            kwargs: The metadata to store about the learner.
        """
        return ["L", learner_id, kwargs]

    @staticmethod
    def learners(learners: Sequence[Learner]) -> Iterable[Any]:
        for index, learner in enumerate(learners):

            try:
                params = learner.params
            except:
                params = {}

            try:
                family = learner.family
            except:
                family = type(learner).__name__

            if len(learner.params) > 0:
                full_name = f"{learner.family}({','.join(f'{k}={v}' for k,v in learner.params.items())})"
            else:
                full_name = learner.family

            yield Transaction.learner(index, full_name=full_name, family=family, **params)

    @staticmethod
    def simulation(simulation_id: int, **kwargs) -> Any:
        """Write simulation metadata row to Result.
        
        Args:
            simulation_index: The index of the simulation in the benchmark's simulations.
            kwargs: The metadata to store about the learner.
        """
        return ["S", simulation_id, kwargs]

    @staticmethod
    def batch(simulation_id:int, learner_id:int, **kwargs) -> Any:
        """Write batch metadata row to Result.

        Args:
            learner_id: The primary key for the learner we observed the batch for.
            simulation_id: The primary key for the simulation the batch came from.
            batch_index: The index of the batch within the simulation.
            kwargs: The metadata to store about the batch.
        """
        return ["B", (simulation_id, learner_id), kwargs]

class TransactionPromote(Filter):

    CurrentVersion = 2

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        items_iter = iter(items)
        items_peek = next(items_iter)
        items_iter = chain([items_peek], items_iter)

        version = 0 if items_peek[0] != 'version' else items_peek[1]

        if version == TransactionPromote.CurrentVersion:
            raise StopPipe()

        while version != TransactionPromote.CurrentVersion:
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

class TransactionIsNew(Filter):

    def __init__(self, existing: Result):

        self._existing = existing

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        for item in items:

            tipe  = item[0]

            if tipe == "version" and self._existing.version is not None:
                continue

            if tipe == "benchmark" and len(self._existing.benchmark) != 0:
                continue

            if tipe == "B" and item[1] in self._existing.batches:
                continue

            if tipe == "S" and item[1] in self._existing.simulations:
                continue

            if tipe == "L" and item[1] in self._existing.learners:
                continue

            yield item

class TransactionSink(Sink):

    def __init__(self, transaction_log: Optional[str], restored: Result) -> None:

        json_encode = Cartesian(JsonEncode())

        self._sink = Pipe.join([json_encode], DiskSink(transaction_log)) if transaction_log else MemorySink()
        self._sink = Pipe.join([TransactionIsNew(restored)], self._sink)

    def write(self, items: Sequence[Any]) -> None:
        self._sink.write(items)

    @property
    def result(self) -> Result:
        if isinstance(self._sink, Pipe.FiltersSink):
            final_sink = self._sink.final_sink()
        else:
            final_sink = self._sink

        if isinstance(final_sink, MemorySink):
            return Result.from_transactions(cast(Iterable[Any], final_sink.items))

        if isinstance(final_sink, DiskSink):
            return Result.from_file(final_sink.filename)

        raise Exception("Transactions were written to an unrecognized sink.")