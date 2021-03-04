
import collections

from copy import deepcopy
from itertools import product, repeat
from typing import Sequence, List, Dict, Any, MutableMapping, Mapping, cast

from coba.simulations import Simulation, OpenmlSimulation, Shuffle, Take, Batch, PCA, Sort
from coba.tools import CobaRegistry

from coba.benchmarks.core import Benchmark

from coba.data.pipes   import Pipe
from coba.data.sources import Source
from coba.data.filters import Filter

class BenchmarkFileFmtV1(Filter[Dict[str,Any], Benchmark]):

    def filter(self, config: Dict[str,Any]) -> Benchmark:

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
