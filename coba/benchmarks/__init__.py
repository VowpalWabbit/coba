"""The benchmarks module contains core benchmark functionality and protocols.

This module contains the abstract interface expected for Benchmark implementations. This 
module also contains several Benchmark implementations and Result data transfer class.
"""

import math
import itertools
import json
import collections

from copy import deepcopy
from statistics import mean
from itertools import product, groupby, chain
from statistics import median
from pathlib import Path
from typing import (
    Iterable, Tuple, Union, Sequence, Generic,
    TypeVar, Dict, Any, cast, Optional,
    overload, List, Mapping, MutableMapping
)

from coba.random import CobaRandom
from coba.learners import Learner, Key
from coba.simulations import BatchedSimulation, OpenmlSimulation, Take, Shuffle, Batch, Simulation, Choice, Context, Action, Reward, PCA, Sort
from coba.statistics import OnlineMean, OnlineVariance
from coba.tools import check_matplotlib_support, check_pandas_support, create_class, CobaConfig

from coba.data.structures import Table
from coba.data.filters import Filter, JsonEncode, JsonDecode, ForeachFilter
from coba.data.sources import Source, MemorySource, DiskSource
from coba.data.sinks import Sink, MemorySink, DiskSink
from coba.data.pipes import Pipe, StopPipe

_C = TypeVar('_C', bound=Context)
_A = TypeVar('_A', bound=Action)

class Result:
    """A class for creating and returning the result of a Benchmark evaluation."""

    @staticmethod
    def from_transaction_log(filename: Optional[str]) -> 'Result':
        """Create a Result from a transaction file."""
        
        if filename is None or not Path(filename).exists(): return Result()

        Pipe.join(DiskSource(filename), [JsonDecode(), TransactionPromote(), JsonEncode()], DiskSink(filename, 'w')).run()
        
        return Result.from_transactions(Pipe.join(DiskSource(filename), [JsonDecode()]).read())

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

        check_pandas_support("Result.to_pandas")

        l = self.learners.to_pandas()
        s = self.simulations.to_pandas()
        b = self.batches.to_pandas()

        return (l,s,b)

    def standard_plot(self, select_learners: Sequence[int] = None,  show_err: bool = False, show_sd: bool = False, figsize=(12,4)) -> None:

        check_matplotlib_support('Plots.standard_plot')

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

            for batch_index, batch_Ns, batch_Rs in zip(itertools.count(), Ns,Rs):

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

        index_unit = "Interaction" if max_batch_N ==1 else "Batch"
        
        ax1 = fig.add_subplot(1,2,1) #type: ignore
        ax2 = fig.add_subplot(1,2,2) #type: ignore

        for learner_id in learners:
            _plot(ax1, learners[learner_id].full_name, indexes[learner_id], inmeans[learner_id], invariances[learner_id], incounts[learner_id])

        ax1.set_title(f"Instantaneous Reward")
        ax1.set_ylabel("Reward")
        ax1.set_xlabel(f"{index_unit} Index")

        for learner_id in learners:
            _plot(ax2, learners[learner_id].full_name, indexes[learner_id], cumeans[learner_id], cuvariances[learner_id], cucounts[learner_id])

        ax2.set_title("Progressive Validation")
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
    
    def __init__(self, simulations: Sequence[Source[BatchedSimulation]], learners: Sequence['BenchmarkLearner'], restored: Result) -> None:
        self._simulations = simulations
        self._learners    = learners
        self._restored    = restored

    def read(self) -> Iterable:

        #first turn every prod(simulation,seed) into its own pipe
        #second remove pipes that are already finished
        #group pipes into a single pipe based on source

        simulation_sources = [s._source if isinstance(s, (Pipe.SourceFilters, BenchmarkSimulation)) else s for s in self._simulations]
        sources_set        = list(set(simulation_sources))

        simulations = dict(enumerate(self._simulations))
        learners    = dict(enumerate(self._learners))
        restored    = self._restored
        source_idxs = { sim_key:sources_set.index(sim_source) for sim_key,sim_source in zip(simulations.keys(), simulation_sources) }

        zero_batch_sims = [row['simulation_id'] for row in restored.simulations.get_where(batch_count=0)]

        is_not_complete = lambda t: t not in restored.batches and t[0] not in zero_batch_sims
        task_keys       = filter(is_not_complete, product(simulations.keys(), learners.keys()))

        source_grouped_tasks: Dict[int, Tuple[List[int], List[int], List[BenchmarkLearner], List[Source[BatchedSimulation]]]] = {}

        for simulation_key, learner_key in task_keys:
            
            if source_idxs[simulation_key] not in source_grouped_tasks:
                source_grouped_tasks[source_idxs[simulation_key]] = ([],[],[],[])
            
            source_grouped_task = source_grouped_tasks[source_idxs[simulation_key]]

            source_grouped_task[0].append(simulation_key)
            source_grouped_task[1].append(learner_key)
            source_grouped_task[2].append(learners[learner_key])
            source_grouped_task[3].append(simulations[simulation_key])

        return list(source_grouped_tasks.values())
    
class TaskToTransactions(Filter):

    def __init__(self, ignore_raise: bool) -> None:
        self._ignore_raise = ignore_raise

    def filter(self, tasks: Iterable[Any]) -> Iterable[Any]:
        for task in tasks:
            for transaction in self._process_task(task):
                yield transaction

    def _process_task(self, task) -> Iterable[Any]:

        simulation_ids   = task[0]
        learner_ids      = task[1]
        learners         = task[2]
        
        simulation_pipes   = task[3]
        simulation_source  = simulation_pipes[0]._source #we only need one source since we group by sources when making tasks
        simulation_filters = [ simulation_pipe._filter for simulation_pipe in simulation_pipes ]

        collapsed_pipe = Pipe.join(simulation_source, [ForeachFilter(simulation_filters)])

        written_simulations = []

        try:
            for simulation_id, learner_id, learner, pipe, simulation in zip(simulation_ids, learner_ids, learners, simulation_pipes, collapsed_pipe.read()):
                batches      = simulation.interaction_batches
                interactions = list(chain.from_iterable(batches))

                if simulation_id not in written_simulations:
                    written_simulations.append(simulation_id)
                    yield Transaction.simulation(simulation_id,
                        source            = pipe.source_description,
                        filters           = pipe.filter_descriptions,
                        interaction_count = len(interactions),
                        batch_count       = len(batches),
                        context_size      = int(median(self._context_sizes(interactions))),
                        action_count      = int(median(self._action_counts(interactions))))

                learner = deepcopy(learner)
                learner.init()

                if len(batches) > 0:
                    Ns, Rs = zip(*[ self._process_batch(batch, simulation.reward, learner) for batch in batches ])
                    yield Transaction.batch(simulation_id, learner_id, N=list(Ns), reward=list(Rs))

        except KeyboardInterrupt:
            raise
        except Exception as e:
            CobaConfig.Logger.log_exception(e, "unhandled exception:")
            if not self._ignore_raise: raise e

    def _process_batch(self, batch, reward, learner) -> Tuple[int, float]:
        
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

        return len(rewards), round(mean(rewards),5)

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
        items_iter = itertools.chain([items_peek], items_iter)

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
        self._sink = Pipe.join([JsonEncode()], DiskSink(transaction_log)) if transaction_log else MemorySink()
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

class BenchmarkFileFmtV1:

    def parse(self, json_txt: str) -> 'Benchmark':

        config = cast(Dict[str,Any], json.loads(json_txt))
        config = self.materialize_templates(config)

        if not isinstance(config["simulations"], collections.Sequence):
            config["simulations"] = [ config["simulations"] ]

        batch_config = config.get('batches', {'size':1})

        kwargs: Dict[str, Any] = {}
        kwargs['ignore_raise'] = config.get("ignore_raise", True)
        kwargs['seeds']        = config.get('shuffle', [None])

        if 'min' in batch_config:
            kwargs['take'] = batch_config['min']

        if 'max' in batch_config:
            kwargs['take'] = batch_config['min']

        for batch_rule in ['size', 'count', 'sizes']:
            if batch_rule in batch_config:
                kwargs[f"batch_{batch_rule}"] = batch_config[batch_rule]

        simulations: List[BenchmarkSimulation] = []

        for sim_config in config["simulations"]:
            if sim_config["type"] != "classification":
                raise Exception("We were unable to recognize the provided simulation type.")

            if sim_config["from"]["format"] != "openml":
                raise Exception("We were unable to recognize the provided data format.")

            source             = OpenmlSimulation(sim_config["from"]["id"], sim_config["from"].get("md5_checksum", None))
            source_description = f'{{"OpenmlSimulation":{sim_config["from"]["id"]}}}'

            filters: List[Filter[Simulation,Simulation]] = []

            filter_descriptions = []

            if "pca" in sim_config and sim_config["pca"] == True:
                filters.append(PCA())
                filter_descriptions.append("PCA")

            if "sort" in sim_config:
                filters.append(Sort(sim_config["sort"]))
                filter_descriptions.append(f'{{"Sort":{sim_config["sort"]}}}')

            simulations.append(BenchmarkSimulation(source, filters, source_description, filter_descriptions))

        return Benchmark(simulations, **kwargs)

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

                    for child_node, child_scope in zip(node.values(), itertools.repeat(scope)):
                        nodes.append(child_node)
                        scopes.append(child_scope)

                if isinstance(node, collections.Sequence) and not isinstance(node, str):
                    for child_node, child_scope in zip(node, itertools.repeat(scope)):
                        nodes.append(child_node)
                        scopes.append(child_scope)

        return root

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

    def __init__(self, learner: Learner[Context,Action], seed: Optional[int]) -> None:
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

class BenchmarkSimulation(Source[Simulation[_C,_A]]):

    def __init__(self, 
        source             : Source[Simulation], 
        filters            : Sequence[Filter[Simulation,Union[Simulation, BatchedSimulation]]],
        source_description :str = "", 
        filter_descriptions:Sequence[str] = []) -> None:
        
        if isinstance(source, BenchmarkSimulation):
            self._source             = source._source #type: ignore
            self._filter             = Pipe.FiltersFilter(list(source._filter._filters)+list(filters)) #type: ignore
            self.source_description  = source.source_description or source_description #type: ignore
            self.filter_descriptions = list(source.filter_descriptions) + list(filter_descriptions) #type: ignore
        else:
            self._source             = source
            self._filter             = Pipe.FiltersFilter(filters)
            self.source_description  = source_description
            self.filter_descriptions = list(filter_descriptions)
    
    def read(self) -> Simulation[_C,_A]:
        return self._filter.filter(self._source.read())

class Benchmark(Generic[_C,_A]):
    """An on-policy Benchmark using samples drawn from simulations to estimate performance statistics."""

    @staticmethod
    def from_file(filename:str) -> 'Benchmark[Context,Action]':
        """Instantiate a Benchmark from a config file."""

        return create_class(CobaConfig.Benchmark['file_fmt']).parse(Path(filename).read_text())

    @overload
    def __init__(self, 
        simulations : Sequence[Source[Simulation[_C,_A]]],
        *,
        batch_size      : int = 1,
        take            : int = None,
        seeds           : Sequence[Optional[int]] = [None],
        ignore_raise    : bool = True,
        processes       : int = None,
        maxtasksperchild: int = None) -> None: ...

    @overload
    def __init__(self,
        simulations : Sequence[Source[Simulation[_C,_A]]],
        *,
        batch_count     : int,
        take            : int = None,
        seeds           : Sequence[Optional[int]] = [None],
        ignore_raise    : bool = True,
        processes       : int = None,
        maxtasksperchild: int = None) -> None: ...

    @overload
    def __init__(self, 
        simulations : Sequence[Source[Simulation[_C,_A]]],
        *,
        batch_sizes     : Sequence[int],
        seeds           : Sequence[Optional[int]] = [None],
        ignore_raise    : bool = True,
        processes       : int = None,
        maxtasksperchild: int = None) -> None: ...

    def __init__(self,*args, **kwargs) -> None:
        """Instantiate a UniversalBenchmark.

        Args:
            simulations: The sequence of simulations to benchmark against.
            batcher: How each simulation is broken into evaluation batches.
            ignore_raise: Should exceptions be raised or logged during evaluation.
            shuffle_seeds: A sequence of seeds for interaction shuffling. None means no shuffle.
            processes: The number of process to spawn during evalution (overrides coba config).
            maxtasksperchild: The number of tasks each process will perform before a refresh.
        
        See the overloads for more information.
        """

        simulations = cast(Sequence[Source[Simulation[_C,_A]]], args[0])
        shufflers   = [ Shuffle(seed) for seed in kwargs.get('seeds', [None]) ]
        taker       = Take(kwargs.get('take', None))

        if 'batch_count' in kwargs:
            batcher = Batch(count=kwargs['batch_count'])
        elif 'batch_sizes' in kwargs:
            batcher = Batch(sizes=kwargs['batch_sizes'])
        else:
            batcher = Batch(size=kwargs.get('batch_size',1))

        pipes: List[BenchmarkSimulation] = []

        for source_id, source in enumerate(simulations):
            for shuffler in shufflers:

                source_description = str(source_id)
                filters_description = []

                if shuffler._seed is not None:
                    filters_description.append(f'{{"Shuffle":{shuffler._seed}}}')

                if taker._count is not None:
                    filters_description.append(f'{{"Take":{taker._count}}}')

                if batcher._size != 1:
                    filters_description.append(f'{{"Batch":{[batcher._size,batcher._count,batcher._sizes]}}}')

                filters: List[Union[Filter[Simulation, Simulation],Filter[Simulation,BatchedSimulation]]] = [shuffler, taker, batcher]

                pipes.append(BenchmarkSimulation(source, filters, source_description, filters_description))

        self._simulation_pipes = cast(Sequence[Source[BatchedSimulation[Context,Action]]], pipes)
        self._ignore_raise     = cast(bool                                               ,kwargs.get('ignore_raise', True))
        self._processes        = cast(Optional[int]                                      ,kwargs.get('processes', None))
        self._maxtasksperchild = cast(Optional[int]                                      ,kwargs.get('maxtasksperchild', None))

    def ignore_raise(self, value:bool=True) -> 'Benchmark[_C,_A]':
        self._ignore_raise = value
        return self

    def processes(self, value:int) -> 'Benchmark[_C,_A]':
        self._processes = value
        return self

    def maxtasksperchild(self, value:int) -> 'Benchmark[_C,_A]':
        self._maxtasksperchild = value
        return self

    def evaluate(self, learners: Sequence[Learner[_C,_A]], transaction_log:str = None, seed:int = None) -> Result:
        """Collect observations of a Learner playing the benchmark's simulations to calculate Results.

        Args:
            factories: See the base class for more information.

        Returns:
            See the base class for more information.
        """
        benchmark_learners   = [ BenchmarkLearner(learner, seed) for learner in learners ] #type: ignore
        restored             = Result.from_transaction_log(transaction_log)
        task_source          = TaskSource(self._simulation_pipes, benchmark_learners, restored)
        task_to_transactions = TaskToTransactions(self._ignore_raise)
        transaction_sink     = TransactionSink(transaction_log, restored)

        n_given_learners    = len(benchmark_learners)
        n_given_simulations = len(self._simulation_pipes)
 
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