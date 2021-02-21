"""The benchmarks module contains core benchmark functionality and protocols.

This module contains the abstract interface expected for Benchmark implementations. This 
module also contains several Benchmark implementations and Result data transfer class.
"""

import math
import collections

from copy import deepcopy
from statistics import mean
from itertools import product, groupby, chain, count, repeat
from statistics import median
from pathlib import Path
from typing import (
    Iterable, Tuple, Sequence, Dict, Any, cast, Optional,
    overload, List, Mapping, MutableMapping, Union
)
from coba.random import CobaRandom
from coba.learners import Learner, Key
from coba.simulations import BatchedSimulation, OpenmlSimulation, Take, Shuffle, Batch, Simulation, Interaction, Choice, Context, Action, Reward, PCA, Sort
from coba.statistics import OnlineMean, OnlineVariance
from coba.tools import PackageChecker, CobaRegistry, CobaConfig

from coba.data.structures import Table
from coba.data.filters import Filter, IdentityFilter, JsonEncode, JsonDecode, Cartesian, StringJoin
from coba.data.sources import HttpSource, Source, MemorySource, DiskSource
from coba.data.sinks import Sink, MemorySink, DiskSink
from coba.data.pipes import Pipe, StopPipe

class Result:
    """A class for creating and returning the result of a Benchmark evaluation."""

    @staticmethod
    def from_transaction_log(filename: Optional[str]) -> 'Result':
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
        ax2.set_xlabel(f"{index_unit} Index")

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
        return str({ "Learners": len(self.learners), "Simulations": len(self.simulations), "Batches": len(self.batches) })

    def __repr__(self) -> str:
        return str(self)

class Transaction:

    @staticmethod
    def version(version) -> Any:
        return ['version', version]

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
    def learners(learners: 'Sequence[BenchmarkLearner]') -> Iterable[Any]:
        for index, learner in enumerate(learners):
            yield Transaction.learner(index, family=learner.family, full_name=learner.full_name, **learner.params)

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

class TaskSource(Source):
    
    def __init__(self, simulations: 'Sequence[BenchmarkSimulation]', learners: Sequence['BenchmarkLearner'], restored: Result) -> None:
        self._simulations = simulations
        self._learners    = learners
        self._restored    = restored

    def read(self) -> Iterable:

        no_batch_sim_rows = self._restored.simulations.get_where(batch_count=0)
        no_batch_sim_ids  = [ row['simulation_id'] for row in no_batch_sim_rows ]

        sim_ids   = list(range(len(self._simulations)))
        learn_ids = list(range(len(self._learners)))

        simulations = dict(enumerate(self._simulations))
        learners    = dict(enumerate(self._learners))

        def is_not_complete(sim_id: int, learn_id: int):
            return (sim_id,learn_id) not in self._restored.batches and sim_id not in no_batch_sim_ids

        not_complete_pairs = [ k for k in product(sim_ids, learn_ids) if is_not_complete(*k) ]
        
        not_complete_pairs_by_source: Dict[object,List[Tuple[int,int]]] = collections.defaultdict(list)

        for ncp in not_complete_pairs:
            not_complete_pairs_by_source[simulations[ncp[0]].source].append(ncp)

        for source, not_complete_pairs in not_complete_pairs_by_source.items():

            not_completed_sims = {ncp[0]: simulations[ncp[0]] for ncp in not_complete_pairs}
            not_completed_lrns = {ncp[1]:    learners[ncp[1]] for ncp in not_complete_pairs}

            yield source, not_complete_pairs, not_completed_sims, not_completed_lrns
    
class TaskToTransactions(Filter):

    def __init__(self, ignore_raise: bool) -> None:
        self._ignore_raise = ignore_raise

    def filter(self, tasks: Iterable[Any]) -> Iterable[Any]:
        tasks = list(tasks)
        print(len(tasks))
        for task in tasks:
            for transaction in self._process_task(task):
                yield transaction

    def _process_task(self, task) -> Iterable[Any]:

        source     = cast(Source[Any]                   , task[0])
        todo_pairs = cast(Sequence[Tuple[int,int]]      , task[1])
        pipes      = cast(Dict[int, BenchmarkSimulation], task[2])
        learners   = cast(Dict[int, Learner            ], task[3])

        def batchify(simulation: Simulation) -> Sequence[Sequence[Interaction]]:
            if isinstance(simulation, BatchedSimulation):
                return simulation.interaction_batches
            else:
                return [ [interaction] for interaction in simulation.interactions ]

        def sim_transaction(sim_id, pipe, interactions, batches):
            return Transaction.simulation(sim_id,
                pipe              = str(pipe),
                interaction_count = len(interactions),
                batch_count       = len(batches),
                context_size      = int(median(self._context_sizes(interactions))),
                action_count      = int(median(self._action_counts(interactions))))

        try:
            with CobaConfig.Logger.log(f"Processing {source}..."):

                with CobaConfig.Logger.time(f"Loading shared data once for {source}..."):
                    loaded_source = source.read()

                for sim_id, pipe in pipes.items():

                    with CobaConfig.Logger.time(f"Creating simulation {sim_id} from {source} shared data..."):            
                        simulation = pipe.filter.filter(loaded_source)

                        interactions = simulation.interactions
                        batches      = batchify(simulation)

                        yield sim_transaction(sim_id, pipe, interactions, batches)
        
                        if not batches:
                            CobaConfig.Logger.log(f"After creation simulation {sim_id} has nothing to evaluate.")
                            CobaConfig.Logger.log("This often happens because `Take` was larger than source data set.")
                            continue

                    with CobaConfig.Logger.log(f"Evaluating learners on simulation {sim_id}..."):
                        for lrn_id, learner in learners.items():

                            if (sim_id,lrn_id) not in todo_pairs: continue

                            learner = deepcopy(learner)
                            learner.init()

                            with CobaConfig.Logger.time(f"Evaluating learner {lrn_id}..."):
                                batch_sizes  = [ len(batch)                                             for batch in batches ]
                                mean_rewards = [ self._process_batch(batch, learner, simulation.reward) for batch in batches ]

                                yield Transaction.batch(sim_id, lrn_id, N=batch_sizes, reward=mean_rewards)
        
        except KeyboardInterrupt:
            raise
        except Exception as e:
            CobaConfig.Logger.log_exception("unhandled exception:", e)
            if not self._ignore_raise: raise e

    def _process_batch(self, batch, learner, reward) -> float:
        
        keys     = []
        contexts = []
        choices  = []
        actions  = []
        probs    = []

        for interaction in batch:

            choice, prob = learner.choose(interaction.key, interaction.context, interaction.actions)

            assert choice in range(len(interaction.actions)), "An invalid action was chosen by the learner"

            keys    .append(interaction.key)
            contexts.append(interaction.context)
            choices .append(choice)
            probs   .append(prob)
            actions .append(interaction.actions[choice])

        rewards = reward(list(zip(keys, choices))) 

        for (key,context,action,reward,prob) in zip(keys,contexts,actions,rewards, probs):
            learner.learn(key,context,action,reward,prob)

        return round(mean(rewards),5)

    def _context_sizes(self, interactions) -> Iterable[int]:
        if len(interactions) == 0:
            yield 0

        for context in [i.context for i in interactions]:
            yield 0 if context is None else len(context) if isinstance(context,tuple) else 1

    def _action_counts(self, interactions) -> Iterable[int]:
        if len(interactions) == 0:
            yield 0

        for actions in [i.actions for i in interactions]:
            yield len(actions)

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
            return Result.from_transaction_log(final_sink.filename)

        raise Exception("Transactions were written to an unrecognized sink.")

class BenchmarkFileFmtV1(Filter[Dict[str,Any], 'Benchmark']):

    def filter(self, config: Dict[str,Any]) -> 'Benchmark':

        config = self.materialize_templates(config)

        if not isinstance(config["simulations"], collections.Sequence):
            config["simulations"] = [ config["simulations"] ]

        batch_config = config.get('batches', {})

        shuffle_filters: List[Shuffle] = []
        take_filters   : List[Take]    = []
        batch_filters  : List[Batch]   = []

        if 'shuffle' in config and config['shuffle'] != None:
            shuffle_filters = [ Shuffle(seed) for seed in config['shuffle'] ]

        if 'min' in batch_config:
            take_filters = [ Take(batch_config['min']) ]
        elif 'max' in batch_config:
            take_filters = [ Take(batch_config['max']) ]

        if 'count' in batch_config:
            batch_filters = [ Batch(count=batch_config['count']) ]
        elif 'size' in batch_config and batch_config['size'] != 1: 
            batch_filters = [ Batch(size=batch_config['size']) ]
        elif 'sizes' in batch_config:
            batch_filters = [ Batch(sizes=batch_config['sizes']) ]

        simulations: List[Source[Simulation]] = []

        for sim_config in config["simulations"]:

            if sim_config["type"] != "classification":
                raise Exception("We were unable to recognize the provided simulation type.")

            if sim_config["from"]["format"] != "openml":
                raise Exception("We were unable to recognize the provided data format.")

            source = OpenmlSimulation(sim_config["from"]["id"], sim_config["from"].get("md5_checksum", None))

            pca_filters  : List[PCA ] = []
            sort_filters : List[Sort] = []
            final_filters: Sequence[Sequence[Any]]

            if sim_config.get("pca",False):
                pca_filters = [PCA()]

            if "sort" in sim_config:
                sort_filters = [Sort(sim_config["sort"])]

            if len(sort_filters) > 0:
                final_filters = [pca_filters, sort_filters, take_filters, batch_filters ]
            else:
                final_filters = [pca_filters, shuffle_filters, take_filters, batch_filters ]

            final_filters = list(filter(None, final_filters))

            if len(final_filters) > 0:
                simulations.extend([ Pipe.join(source, f) for f in product(*final_filters) ])
            else:
                simulations.append(source)

        return Benchmark(simulations)

    def materialize_templates(self, root: Any):
        """This class materializes templates within benchmark json files.
    
        The templating engine works as follows: 
            1. Look in the root object for a "templates" key. Templates should be objects themselves
            with constants and variables. Variable values are indicated by beginning with a $.
            2. Recursively walk through the remainder of the children from the root. For every object found
            check to see if it has a "template" value. 
            3. For every object found with a "template" value defined materialize all of that template's static
            values into this object. If an object has a static variable defined that is also in the template give
            preference to the local object's value.
            4. Assign any defined variables to the template as well (i.e., those values that start with a $). 
            5. Keep defined variables in context while walking child objects in case they are needed as well.
        """

        if "templates" in root:

            templates: Dict[str,Dict[str,Any]] = root.pop("templates")
            nodes    : List[Any]               = [root]
            scopes   : List[Dict[str,Any]]     = [{}]

            def materialize_template(document: MutableMapping[str,Any], template: Mapping[str,Any]):

                for key in template:
                    if key in document:
                        if isinstance(template[key], collections.Mapping) and isinstance(template[key], collections.Mapping):
                            materialize_template(document[key],template[key])
                    else:
                        document[key] = template[key]

            def materialize_variables(document: MutableMapping[str,Any], variables: Mapping[str,Any]):
                for key in document:
                    if isinstance(document[key],str) and document[key] in variables:
                        document[key] = variables[document[key]]

            while len(nodes) > 0:
                node  = nodes.pop()
                scope = scopes.pop().copy()  #this could absolutely be made more memory-efficient if needed

                if isinstance(node, collections.MutableMapping):

                    if "template" in node and node["template"] not in templates:
                        raise Exception(f"We were unable to find template '{node['template']}'.")

                    keys      = list(node.keys())
                    template  = templates[node.pop("template")] if "template" in node else cast(Dict[str,Any], {})
                    variables = { key:node.pop(key) for key in keys if key.startswith("$") }

                    template = deepcopy(template)
                    scope.update(variables)

                    materialize_template(node, template)
                    materialize_variables(node, scope)

                    for child_node, child_scope in zip(node.values(), repeat(scope)):
                        nodes.append(child_node)
                        scopes.append(child_scope)

                if isinstance(node, collections.Sequence) and not isinstance(node, str):
                    for child_node, child_scope in zip(node, repeat(scope)):
                        nodes.append(child_node)
                        scopes.append(child_scope)

        return root

class BenchmarkFileFmtV2(Filter[Dict[str,Any], 'Benchmark']):

    def filter(self, config: Dict[str,Any]) -> 'Benchmark':

        variables = { k: CobaRegistry.construct(v) for k,v in config.get("variables",{}).items() }

        def _construct(item:Any) -> Sequence[Any]:
            result = None

            if isinstance(item, str) and item in variables:
                result = variables[item]

            if isinstance(item, str) and item not in variables:
                result = CobaRegistry.construct(item)

            if isinstance(item, dict):
                result = CobaRegistry.construct(item)

            if isinstance(item, list):
                if any([ isinstance(i,list) for i in item ]):
                    raise Exception("Recursive structures are not supported in benchmark simulation configs.")
                pieces = list(map(_construct, item))
                result = [ Pipe.join(s, f) for s in pieces[0] for f in product(*pieces[1:])]

            if result is None:
                raise Exception(f"We were unable to construct {item} in the given benchmark file.")

            return result if isinstance(result, collections.Sequence) else [result]

        if not isinstance(config['simulations'], list): config['simulations'] = [config['simulations']]

        simulations = [ simulation for recipe in config['simulations'] for simulation in _construct(recipe)]

        return Benchmark(simulations)

class BenchmarkLearner:

    @property
    def family(self) -> str:
        try:
            return self._learner.family
        except AttributeError:
            return self._learner.__class__.__name__

    @property
    def params(self) -> Dict[str, Any]:
        try:
            return self._learner.params
        except AttributeError:
            return {}

    @property
    def full_name(self) -> str:
        if len(self.params) > 0:
            return f"{self.family}({','.join(f'{k}={v}' for k,v in self.params.items())})"
        else:
            return self.family

    def __init__(self, learner: Learner, seed: Optional[int]) -> None:
        self._learner = learner
        self._random  = CobaRandom(seed)

    def init(self) -> None:
        try:
            self._learner.init()
        except AttributeError:
            pass

    def choose(self, key: Key, context: Context, actions: Sequence[Action]) -> Tuple[Choice, float]:
        p = self._learner.predict(key, context, actions)
        c = self._random.choice(list(range(len(actions))), p)

        return c, p[c]
    
    def learn(self, key: Key, context: Context, action: Action, reward: Reward, probability: float) -> None:
        self._learner.learn(key, context, action, reward, probability)

class BenchmarkSimulation(Source[Simulation]):

    def __init__(self, source: Source[Simulation], filters: Sequence[Filter[Simulation,Simulation]] = None) -> None:
        self._pipe = source if filters is None else Pipe.join(source, filters)

    @property
    def source(self) -> Source[Simulation]:
        return self._pipe._source if isinstance(self._pipe, (Pipe.SourceFilters)) else self._pipe

    @property
    def filter(self) -> Filter[Simulation,Simulation]:
        return self._pipe._filter if isinstance(self._pipe, Pipe.SourceFilters) else IdentityFilter()

    def read(self) -> Simulation:
        return self._pipe.read()

    def __repr__(self) -> str:
        return self._pipe.__repr__()

class Benchmark:
    """An on-policy Benchmark using samples drawn from simulations to estimate performance statistics."""
    
    @overload
    @staticmethod
    def from_file(filesource:Union[Source[str], Source[Iterable[str]]]) -> 'Benchmark': ...

    @overload
    @staticmethod
    def from_file(filename:str) -> 'Benchmark': ...
    
    @staticmethod #type: ignore #(this apppears to be a mypy bug https://github.com/python/mypy/issues/7781)
    def from_file(arg) -> 'Benchmark': #type: ignore
        """Instantiate a Benchmark from a config file."""

        source:Any = None

        if isinstance(arg,str) and arg.startswith('http'):
            source = HttpSource(arg, cache=False)
        
        if isinstance(arg,str) and not arg.startswith('http'):
            source = DiskSource(arg)

        if source is None:
            source = arg
        
        content = source.read()
        joined  = content if isinstance(content,str) else StringJoin('\n').filter(content)
        decoded = JsonDecode().filter(joined)

        return CobaRegistry.construct(CobaConfig.Benchmark['file_fmt']).filter(decoded)

    @overload
    def __init__(self, 
        simulations : Sequence[Source[Simulation]],
        *,
        batch_size      : int = 1,
        take            : int = None,
        shuffle         : Sequence[Optional[int]] = [None],
        ignore_raise    : bool = True,
        processes       : int = None,
        maxtasksperchild: int = None) -> None: ...

    @overload
    def __init__(self,
        simulations : Sequence[Source[Simulation]],
        *,
        batch_count     : int,
        take            : int = None,
        shuffle         : Sequence[Optional[int]] = [None],
        ignore_raise    : bool = True,
        processes       : int = None,
        maxtasksperchild: int = None) -> None: ...

    @overload
    def __init__(self, 
        simulations : Sequence[Source[Simulation]],
        *,
        batch_sizes     : Sequence[int],
        shuffle         : Sequence[Optional[int]] = [None],
        ignore_raise    : bool = True,
        processes       : int = None,
        maxtasksperchild: int = None) -> None: ...

    def __init__(self,*args, **kwargs) -> None:
        """Instantiate a UniversalBenchmark.

        Args:
            simulations: The sequence of simulations to benchmark against.
            batcher: How each simulation is broken into evaluation batches.
            ignore_raise: Should exceptions be raised or logged during evaluation.
            shuffle: A sequence of seeds for simulation shuffling. None means no shuffle.
            processes: The number of process to spawn during evalution (overrides coba config).
            maxtasksperchild: The number of tasks each process will perform before a refresh.
        
        See the overloads for more information.
        """

        sources = cast(Sequence[Source[Simulation]], args[0])
        filters: List[Sequence[Filter[Simulation,Simulation]]] = []

        if 'shuffle' in kwargs and kwargs['shuffle'] != [None]:
            filters.append([ Shuffle(seed) for seed in kwargs['shuffle'] ])

        if 'take' in kwargs:
            filters.append([ Take(kwargs['take']) ])

        if 'batch_count' in kwargs:
            filters.append([ Batch(count=kwargs['batch_count']) ])
        elif 'batch_size' in kwargs:
            filters.append([ Batch(size=kwargs['batch_size']) ])
        elif 'batch_sizes' in kwargs:
            filters.append([ Batch(sizes=kwargs['batch_sizes']) ])

        if len(filters) > 0:
            benchmark_simulations = [BenchmarkSimulation(s,f) for s,f in product(sources, product(*filters))]
        else:
            benchmark_simulations = list(map(BenchmarkSimulation, sources))

        self._simulations      = benchmark_simulations
        self._ignore_raise     = cast(bool         , kwargs.get('ignore_raise'    , True))
        self._processes        = cast(Optional[int], kwargs.get('processes'       , None))
        self._maxtasksperchild = cast(Optional[int], kwargs.get('maxtasksperchild', None))

    def ignore_raise(self, value:bool=True) -> 'Benchmark':
        self._ignore_raise = value
        return self

    def processes(self, value:int) -> 'Benchmark':
        self._processes = value
        return self

    def maxtasksperchild(self, value:int) -> 'Benchmark':
        self._maxtasksperchild = value
        return self

    def evaluate(self, learners: Sequence[Learner], transaction_log:str = None, seed:int = None) -> Result:
        """Collect observations of a Learner playing the benchmark's simulations to calculate Results.

        Args:
            factories: See the base class for more information.

        Returns:
            See the base class for more information.
        """
        benchmark_learners   = [ BenchmarkLearner(learner, seed) for learner in learners ] #type: ignore
        restored             = Result.from_transaction_log(transaction_log)
        task_source          = TaskSource(self._simulations, benchmark_learners, restored)
        task_to_transactions = TaskToTransactions(self._ignore_raise)
        transaction_sink     = TransactionSink(transaction_log, restored)

        n_given_learners    = len(benchmark_learners)
        n_given_simulations = len(self._simulations)
 
        if len(restored.benchmark) != 0:
            assert n_given_learners    == restored.benchmark['n_learners'   ], "The currently evaluating benchmark doesn't match the given transaction log"
            assert n_given_simulations == restored.benchmark['n_simulations'], "The currently evaluating benchmark doesn't match the given transaction log"

        preamble_transactions = []
        preamble_transactions.append(Transaction.version(TransactionPromote.CurrentVersion))
        preamble_transactions.append(Transaction.benchmark(n_given_learners, n_given_simulations))
        preamble_transactions.extend(Transaction.learners(benchmark_learners))

        mp = self._processes        if self._processes        else CobaConfig.Benchmark['processes']
        mt = self._maxtasksperchild if self._maxtasksperchild else CobaConfig.Benchmark['maxtasksperchild']
        
        Pipe.join(MemorySource(preamble_transactions), []                    , transaction_sink).run(1,None)
        Pipe.join(task_source                        , [task_to_transactions], transaction_sink).run(mp,mt)

        return transaction_sink.result