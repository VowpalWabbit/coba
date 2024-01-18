import json
import collections.abc

from urllib import request
from pathlib import Path
from zipfile import ZipFile, BadZipFile
from typing import Sequence, overload, Union, Iterable, Iterator, Optional, Tuple, Callable, Mapping, Type, Literal, Any

from coba                 import pipes
from coba.context         import CobaContext, DiskCacher, DecoratedLogger, ExceptLog, NameLog, StampLog
from coba.primitives      import Context, Action, Source, Learner, Environment, EnvironmentFilter
from coba.random          import CobaRandom
from coba.primitives      import Dense, Sparse
from coba.pipes           import Pipes, HttpSource, IterableSource, DataFrameSource, DiskSource, NextSource
from coba.exceptions      import CobaException
from coba.multiprocessing import CobaMultiprocessor

from coba.environments.templates  import EnvironmentsTemplateV1, EnvironmentsTemplateV2
from coba.environments.openml     import OpenmlSimulation
from coba.environments.synthetics import LinearSyntheticSimulation, NeighborsSyntheticSimulation
from coba.environments.synthetics import KernelSyntheticSimulation, MLPSyntheticSimulation, LambdaSimulation
from coba.environments.supervised import SupervisedSimulation
from coba.environments.results    import ResultEnvironment

from coba.environments.filters   import Repr, Batch, Chunk, Logged, Finalize
from coba.environments.filters   import Binary, Shuffle, Take, Sparsify, Densify, Reservoir, Cycle, Scale, Unbatch
from coba.environments.filters   import Slice, Impute, Where, Riffle, Sort, Flatten, Cache, Params, Grounded
from coba.environments.filters   import OpeRewards, Noise, BatchSafe

from coba.environments.serialized import EnvironmentFromObjects, EnvironmentsToObjects, ZipMemberToObjects, ObjectsToZipMember

class Environments(collections.abc.Sequence, Sequence[Environment]):
    """A friendly API for common environment functionality."""

    @staticmethod
    def cache_dir(path:Union[str,Path]='~/.cache/coba') -> Type['Environments']:
        """Set the cache directory for openml sources.

        Args:
            path: A path to a directory to cache openml sources.

        Returns:
            The Environments class.
        """
        CobaContext.cacher = DiskCacher(path)
        return Environments

    @staticmethod
    def from_template(source: Union[str,Source[Iterable[str]]], **user_vars) -> 'Environments':
        """Create Environments from a template file.

        Args:
            **user_vars: overrideable template variables

        Returns:
            An Environments object.
        """
        try:
            return Environments(*EnvironmentsTemplateV2(source, **user_vars).read())
        except Exception as e: #pragma: no cover
            try:
                #try previous version of definition files. If previous versions also fail
                #then we raise the exception given by the most up-to-date version so that
                #changes can be made to conform to the latest version.
                return Environments(*EnvironmentsTemplateV1(source).read())
            except:
                raise e

    @staticmethod
    def from_prebuilt(name:str) -> 'Environments':
        """Create Environments from a pre-built definition.

        Args:
            name: The desired pre-built environment.

        Returns:
            An Environments object.
        """

        repo_url       = "https://github.com/mrucker/coba_prebuilds/blob/main"
        definition_url = f"{repo_url}/{name}/index.json?raw=True"

        try:
            definition_txt = HttpSource(definition_url).read()
        except request.HTTPError as e:
            if e.code != 404:
                raise
            else:
                root_dir_text = HttpSource("https://api.github.com/repos/mrucker/coba_prebuilds/contents/").read()
                root_dir_json = json.loads(root_dir_text)
                known_names   = [ obj['name'] for obj in root_dir_json if obj['name'] != "README.md" ]
                raise CobaException(f"The given prebuilt name, {name}, couldn't be found. Known names are: {known_names}")

        definition_txt = definition_txt.replace('"./', f'"{repo_url}/{name}/')
        definition_txt = definition_txt.replace('.json"', '.json?raw=True"')
        return Environments.from_template(IterableSource([definition_txt]))

    @staticmethod
    def from_linear_synthetic(
        n_interactions: int,
        n_actions:int = 5,
        n_context_features: int = 5,
        n_action_features: int = 5,
        n_coefficients:Optional[int] = 5,
        reward_features: Sequence[str] = ["a","xa"],
        seed: Union[int,Sequence[int]] = 1) -> 'Environments':
        """Create Environments from linear reward functions.

        Args:
            n_interactions: The number of interactions the simulation should have.
            n_actions: The number of actions each interaction should have.
            n_context_features: The number of features each context should have.
            n_action_features: The number of features each action should have.
            n_coefficients The number of non-zero weights in the final reward function.
            reward_features: The features in the simulation's linear reward function.
            seed: The seed used to generate all random values. If seed is a list then
                an separate environment will be created for each seed.

        Returns:
            An Environments object.
        """

        seed = [seed] if not isinstance(seed,collections.abc.Sequence) else seed
        args = (n_interactions, n_actions, n_context_features, n_action_features, n_coefficients, reward_features)

        return Environments([LinearSyntheticSimulation(*args, s) for s in seed])

    @staticmethod
    def from_neighbors_synthetic(
        n_interactions: int,
        n_actions: int = 2,
        n_context_features: int = 5,
        n_action_features: int = 5,
        n_neighborhoods: int = 30,
        seed: Union[int,Sequence[int]] = 1) -> 'Environments':
        """Create Environments from nearest neighbors reward functions.

        Args:
            n_interactions: The number of interactions the simulation should have.
            n_actions: The number of actions each interaction should have.
            n_context_features: The number of features each context should have.
            n_action_features: The number of features each action should have.
            n_neighborhoods: The number of distinct reward value neighborhoods.
            seed: The seed used to generate all random values. If seed is a list then
                an separate environment will be created for each seed.

        Returns:
            An Environments object.
        """

        seed = [seed] if not isinstance(seed,collections.abc.Sequence) else seed
        args = (n_interactions, n_actions, n_context_features, n_action_features, n_neighborhoods)

        return Environments([NeighborsSyntheticSimulation(*args, s) for s in seed])

    @staticmethod
    def from_kernel_synthetic(
        n_interactions:int,
        n_actions:int = 10,
        n_context_features:int = 10,
        n_action_features:int = 10,
        n_exemplars:int = 10,
        kernel: Literal['linear','polynomial','exponential','gaussian'] = 'gaussian',
        degree: int = 3,
        gamma: float = 1,
        seed: Union[int,Sequence[int]] = 1) -> 'Environments':
        """Create Environments from kernel-based reward functions.

        Args:
            n_interactions: The number of interactions the simulation should have.
            n_actions: The number of actions each interaction should have.
            n_context_features: The number of features each context should have.
            n_action_features: The number of features each action should have.
            n_exemplars: The number of exemplar context-action pairs.
            kernel: The family of the kernel basis functions.
            degree: This argument is only relevant when using polynomial kernels.
            gamma: This argument is only relevant when using exponential kernels.
            seed: The seed used to generate all random values. If seed is a list then
                an separate environment will be created for each seed.

        Returns:
            An Environments object.
        """

        seed = [seed] if not isinstance(seed,collections.abc.Sequence) else seed
        args = (n_interactions, n_actions, n_context_features, n_action_features, n_exemplars, kernel, degree, gamma)

        return Environments([KernelSyntheticSimulation(*args, s) for s in seed])

    @staticmethod
    def from_mlp_synthetic(
        n_interactions:int,
        n_actions:int = 10,
        n_context_features:int = 10,
        n_action_features:int = 10,
        seed: Union[int,Sequence[int]] = 1) -> 'Environments':
        """Create Environments from kernel-based reward functions.

        Args:
            n_interactions: The number of interactions the simulation should have.
            n_actions: The number of actions each interaction should have.
            n_context_features: The number of features each context should have.
            n_action_features: The number of features each action should have.
            seed: The seed used to generate all random values. If seed is a list then
                an separate environment will be created for each seed.

        Remarks:
            The MLP architecture has a single hidden layer with sigmoid
            activation and one output value calculated from a random
            linear combination of the hidden layer's output.

        Returns:
            An Environments object.
        """

        seed = [seed] if not isinstance(seed,collections.abc.Sequence) else seed
        args = (n_interactions, n_actions, n_context_features, n_action_features)

        return Environments([MLPSyntheticSimulation(*args, s) for s in seed])

    @overload
    @staticmethod
    def from_openml(
        data_id: Union[int,Sequence[int]],
        drop_missing: bool = True,
        take: int = None,
        *,
        target:str = None,
        label_type:Literal['c','r','m'] = None) -> 'Environments':
        ...
        """Create Environments from openml datasets.

        Args:
            data_id: The data id for a dataset on openml (i.e., openml.org/d/{id}).
                If data_id is a list then an environment will be created for each id.
            drop_missing: Drop data rows with missing values.
            take: The interaction count for the simulation (selected at random).
            target: The column that should be marked as the label in the source.
            label_type: Is the label a classification, regression or multilabel type.

        Returns:
            An Environments object.
        """

    @overload
    @staticmethod
    def from_openml(
        *,
        task_id: Union[int,Sequence[int]],
        drop_missing: bool = True,
        take: int = None,
        target:str = None,
        label_type:Literal['m','c','r'] = None) -> 'Environments':
        ...
        """Create Environments from openml datasets.

        Args:
            task_id: The task id for a task on openml (i.e., openml.org/t/{id}).
                If task_id is a list then an environment will be created for each id.
            drop_missing: Drop data rows with missing values.
            take: The interaction count for the simulation (selected at random).
            target: The column that should be marked as the label in the source.
            label_type: Is the label a classification, regression or multilabel type.

        Returns:
            An Environments object.
        """

    @staticmethod
    def from_openml(*args,**kwargs) -> 'Environments':
        """Create Environments from openml datasets.

        Args:
            data_id: The data id for a dataset on openml (i.e., openml.org/d/{id}).
                If data_id is a list then an environment will be created for each id.
            task_id: The task id for a task on openml (i.e., openml.org/t/{id}).
                If task_id is a list then an environment will be created for each id.
            drop_missing: Drop data rows with missing values.
            take: The interaction count for the simulation (selected at random).
            target: The column that should be marked as the label in the source.
            label_type: Is the label a classification, regression or multilabel type.

        Returns:
            An Environments object.

        """

        kwargs.update(zip(['data_id','drop_missing','take'], args))

        if 'data_id' in kwargs and isinstance(kwargs['data_id'],int):
            kwargs['data_id'] = [kwargs['data_id']]

        if 'task_id' in kwargs and isinstance(kwargs['task_id'],int):
            kwargs['task_id'] = [kwargs['task_id']]

        if 'data_id' in kwargs:
            return Environments(*[OpenmlSimulation(data_id=id, **kwargs) for id in kwargs.pop('data_id')])
        else:
            return Environments(*[OpenmlSimulation(task_id=id, **kwargs) for id in kwargs.pop('task_id')])

    @overload
    @staticmethod
    def from_supervised(
        source: Source[Union[Iterable[Dense], Iterable[Sparse],Iterable[Tuple[Any,Any]]]],
        label_col: Union[int,str] = None,
        label_type: Literal["c","r","m"] = None,
        take: int = None) -> 'Environments':
        """Create Environments from supervised datasets.

        Args:
            source: A source that reads the supervised data. Coba natively
                provides support for csv, arff, libsvm, and manik data sources.
            label_col: The header or index of the label in each example. If None
                the source must return an iterable of tuple pairs where the first
                item are features and the second item is a label.
            label_type: Is the label a classification, regression or multilabel type.
                If None the label type will be inferred based on the data source.
            take: The interaction count for the simulation (selected at random).

        Returns:
            An Environments object.
        """
        ...

    @overload
    @staticmethod
    def from_supervised(
        X = Sequence[Any],
        Y = Sequence[Any],
        label_type: Literal["c","r","m"] = None) -> 'Environments':
        """Create Environments using a supervised dataset.

        Args:
            X: The features to use when creating contexts.
            Y: The labels to use when creating actions and rewards.
            label_type: Is the label a classification, regression or multilabel type.
                If None the label type will be inferred based on the data source.

        Returns:
            An Environments object.
        """
        ...

    @staticmethod
    def from_supervised(*args, **kwargs) -> 'Environments':
        """Create Environments using a supervised dataset.

        Args:
            source: A source that reads the supervised data. Coba natively
                provides support for csv, arff, libsvm, and manik data sources.
            label_col: The header or index of the label in each example. If None
                the source must return an iterable of tuple pairs where the first
                item are features and the second item is a label.
            X: The features to use when creating contexts.
            Y: The labels to use when creating actions and rewards.
            label_type: Is the label a classification, regression or multilabel type.
                If None the label type will be inferred based on the data source.
            take: The interaction count for the simulation (selected at random).

        Returns:
            An Environments object.
        """
        return Environments(SupervisedSimulation(*args, **kwargs))

    @staticmethod
    def from_save(path:str) -> 'Environments':
        """Create Environments from an Environments save file.

        Args:
            path: Path to an Environments save file.

        Returns:
            An Environments object.
        """
        envs = []
        for name in ZipFile(path).namelist():
            envs.append(EnvironmentFromObjects(ZipMemberToObjects(path,name)))
        return Environments(envs)

    @overload
    def from_lambda(self,
        n_interactions: Optional[int],
        context       : Callable[[int               ],Context         ],
        actions       : Callable[[int,Context       ],Sequence[Action]],
        reward        : Callable[[int,Context,Action],float           ]) -> 'Environments':
        """Create a SimulatedEnvironment from lambda functions."""

    @overload
    def from_lambda(self,
        n_interactions: Optional[int],
        context       : Callable[[int               ,CobaRandom],Context         ],
        actions       : Callable[[int,Context       ,CobaRandom],Sequence[Action]],
        reward        : Callable[[int,Context,Action,CobaRandom],float           ],
        seed          : int) -> 'Environments':
        """Create a SimulatedEnvironment from lambda functions."""

    @staticmethod
    def from_lambda(*args,**kwargs) -> 'Environments':
        """Create a SimulatedEnvironment from lambda functions."""
        return Environments(LambdaSimulation(*args,**kwargs))

    @staticmethod
    def from_dataframe(dataframe) -> 'Environments':
        """Create Environments from a dataframe.

        Args:
            dataframe: Create Environments from this. There will
                only be one environment whose interactions will
                be the rows of the dataframe.

        Returns:
            An Environments object.
        """
        return Environments(DataFrameSource(dataframe))

    @staticmethod
    def from_result(path:str) -> 'Environments':
        """Create Environments from a given Result file.

        Args:
            path: The path to results of an experiment. One environment will be
            created for every environment in the Experiment that produced the results.

        Remarks:
            We assume that 'context', 'action', probability', and 'reward' was
            recorded during the experiment.

        Returns:
            An Environments object.
        """

        env_rows = collections.defaultdict(dict)
        lrn_rows = collections.defaultdict(dict)
        val_rows = collections.defaultdict(dict)
        interactions = []

        for (loc,line) in DiskSource(path,include_loc=True).read():
            if line.strip():
                trx = json.loads(line)
                if trx[0] == "E": env_rows[trx[1]].update(trx[2])
                if trx[0] == "L": lrn_rows[trx[1]].update(trx[2])
                if trx[0] == "V": val_rows[trx[1]].update(trx[2])
                if trx[0] == "I": interactions.append((loc,*trx[1],0)[:4] )

        envs = []
        for loc, env_id, lrn_id, val_id in interactions:
            env_params = env_rows.get(env_id)
            lrn_params = lrn_rows.get(lrn_id)
            val_params = val_rows.get(val_id)
            int_source = NextSource(DiskSource(path,start_loc=loc))
            envs.append(ResultEnvironment(int_source,env_params,lrn_params,val_params))

        return Environments(envs)

    @staticmethod
    def from_custom(*environments: Union[Environment, Sequence[Environment]]):
        """Create Environments from Environment.

        Args:
            *environments: Create an Environments from the environments.

        Returns:
            An Environments object.
        """
        return Environments(*environments)

    @staticmethod
    def from_feurer(drop_missing: bool = True) -> 'Environments':
        """Create Environments from the Feurer benchmark.

        Args:
            drop_missing: Exclude interactions with missing context features.

        Remarks:
            The description of the benchmark is provided at https://arxiv.org/abs/2007.04074.
            For Task ids 232, 3044, 75105, and 211723 every row has a missing feature. These
            environments will be empty when drop_missing is True. Task id 189866 has been
            updated to 361282, a new version of the original dataset that fixes api issues
            with the old dataset.

        Returns:
            An Environments object.
        """

        task_ids = [232,236,241,245,253,254,256,258,260,262,267,271,273,275,279,288,336,340,
                    2119,2120,2121,2122,2123,2125,2356,3044,3047,3048,3049,3053,3054,3055,
                    75089,75092,75093,75097,75098,75100,75105,75108,75109,75112,75114,75115,
                    75116,75118,75120,75121,75125,75126,75127,75129,75131,75133,75134,75136,
                    75139,75141,75142,75143,75146,75147,75148,75149,75153,75154,75156,75157,
                    75159,75161,75163,75166,75169,75171,75173,75174,75176,75178,75179,75180,
                    75184,75185,75187,75192,75193,75195,75196,75199,75210,75212,75213,75215,
                    75217,75219,75221,75223,75225,75232,75233,75234,75235,75236,75237,75239,
                    75250,126021,126024,126025,126026,126028,126029,126030,126031,146574,146575,
                    146576,146577,146578,146583,146586,146592,146593,146594,146596,146597,146600,
                    146601,146602,146603,146679,166859,166866,166872,166875,166882,166897,166905,
                    166906,166913,166915,166931,166932,166944,166950,166951,166953,166956,166957,
                    166958,166959,166970,166996,167083,167085,167086,167087,167088,167089,167090,
                    167094,167096,167097,167099,167100,167101,167103,167104,167105,167106,167149,
                    167152,167161,167168,167181,167184,167185,167190,167200,167201,167202,167203,
                    167204,167205,168785,168791,168792,168793,168794,168795,168796,168797,168798,
                    189779,189786,189828,189829,189836,189840,189841,189843,189844,189845,189846,
                    189858,189859,189860,189861,189862,189863,189864,189865,361282,189869,189870,
                    189871,189872,189873,189874,189875,189878,189880,189881,189882,189883,189884,
                    189887,189890,189893,189894,189899,189900,189902,189905,189906,189908,189909,
                    190154,190155,190156,190157,190158,190159,211720,211721,211722,211723,211724]

        return Environments.from_openml(task_id=task_ids,drop_missing=drop_missing)

    def __init__(self, *environments: Union[Environment, Sequence[Environment]]):
        """Instantiate an Environments class.

        Args:
            *environments: Base environments.
        """
        is_collection = lambda env: isinstance(env, (collections.abc.Sequence,collections.abc.Generator))
        self._envs = []
        for env in environments:
            if is_collection(env):
                self._envs.extend(env)
            else:
                self._envs.append(env)
        self._envs = list(map(Pipes.join,self._envs))

    def binary(self) -> 'Environments':
        """Transform reward values to 1 or 0.

        Remarks:
            Reward values are either 1 (if max reward) or 0 (not max reward).

        Returns:
            An Environments object.
        """
        return self.filter(Binary())

    def sparse(self, context:bool = True, action:bool = False) -> 'Environments':
        """Ensure that features are sparse.

        Args:
            context: Sparsify context features.
            action: Sparsify action features.

        Returns:
            An Environments object.
        """
        return self.filter(Sparsify(context,action))

    def dense(self, n_feats:int, method:Literal['lookup','hashing'], context:bool = True, action:bool = False) -> 'Environments':
        """Ensure that features are dense.

        Args:
            n_feats: The number of features densified environment should have.
            method: How sparse features are turned into dense features. The hashing
                trick is more memory efficient but may have collisions. The lookup
                method is less memory efficient but guaranteed to have no collisions.
            context: Densify context features.
            action: Densify action features.

        Returns:
            An Environments object.
        """


        make_dense = lambda: Densify(n_feats=n_feats, method=method, context=context, action=action)
        return Environments([Pipes.join(env,make_dense()) for env in self._envs])

    @overload
    def shuffle(self, seed: int = 1) -> 'Environments':
        ...

    @overload
    def shuffle(self, seeds: Iterable[int]) -> 'Environments':
        ...

    @overload
    def shuffle(self, *, n:int) -> 'Environments':
        ...

    def shuffle(self, *args,**kwargs) -> 'Environments':
        """Shuffle interaction order.

        Args:
            seed: The seed determining shuffle order.
            seeds: Sequence of seeds determining shuffle order.
                A new environment is made for every seed where
                the only difference is the order of interactions.
            n: The number of shuffling orders to produce. Equivalent
                to `shuffle(seeds=range(n))`.

        Returns:
            An Environments object.
        """

        flat = lambda a: next(pipes.Flatten().filter([a]))

        if kwargs and 'n' in kwargs:
            seeds = range(kwargs['n'])
        else:
            args = kwargs.get('seed',kwargs.get('seeds',args))
            seeds = flat(args) or [1]

        if isinstance(seeds,int): seeds = [seeds]

        shuffled = self.filter([Shuffle(seed) for seed in seeds])

        #Experience has shown that most of the time we want to sort.
        #This doesn't change the experiment results. It simply makes it
        #easier monitor an experiment while it runs in the background.
        ordered = sorted(shuffled, key=lambda env: env.params.get('shuffle',0))

        return Environments(ordered)

    def sort(self, *keys: Union[str,int,Sequence[Union[str,int]]]) -> 'Environments':
        """Sort interactions by features.

        Args:
            *keys: The index or keys for context features.

        Returns:
            An Environments object.
        """
        return self.filter(Sort(*keys))

    def riffle(self, spacing: int, seed: int = 1) -> 'Environments':
        """Riffle shuffle interactions.

        Args:
            spacing: The number of interactions from the beginning
                between each interaction shuffled in from the end.
            seed: The seed used to determine the location of each
                ending interaction when placed within its beginning
                space.

        Returns:
            An Environments object.
        """
        return self.filter(Riffle(spacing, seed))

    def cycle(self, after: int) -> 'Environments':
        """Cycle reward values.

        Useful for testing a learner's response to a non-stationary noise.

        Args:
            after: The number of interactions to wait before cycling reward values.

        Returns:
            An Environments object.
        """
        if isinstance(after,(int,float)): after = [after]
        return self.filter([Cycle(a) for a in after])

    def params(self, params: Mapping[str,Any]) -> 'Environments':
        """Add params to environments.

        Args:
            params: Parameter values to add to each Environment in Environments.

        Returns:
            An Environments object.
        """
        return self.filter(Params(params))

    def take(self, n_interactions: int, strict: bool = False) -> 'Environments':
        """Take the first n interactions.

        Args:
            n_interactions: The maximum number of interactions to take.
            strict: Do not take any interactions if there are not at least n_interactions.

        Returns:
            An Environments object.
        """
        return self.filter(Take(n_interactions, strict))

    def slice(self, start: Optional[int], stop: Optional[int]=None, step:int = 1) -> 'Environments':
        """Take a slice of interactions.

        Args:
            start: The starting index for the slice.
            stop: The finishing index for the slice (exclusive).
            step: The step size between each item in the slice.

        Returns:
            An Environments object.
        """
        return self.filter(Slice(start,stop,step))

    def reservoir(self, n_interactions: int, seeds: Union[int,Sequence[int]]=1, strict:bool = False) -> 'Environments':
        """Take n random interactions.

        Args:
            n_interactions: The maximum number of interactions to sample.
            seeds: The seed to use to randomly generate the sample. If a
                sequence of seeds is provided then multiple samples are drawn.
            strict: Do not draw a sample if there are not at least n_interactions.

        Returns:
            An Environments object.
        """
        if isinstance(seeds,int): seeds = [seeds]
        return self.filter([Reservoir(n_interactions,strict=strict,seed=seed) for seed in seeds])

    def scale(self,
        shift: Union[float,Literal["min","mean","med"]] = "min",
        scale: Union[float,Literal["minmax","std","iqr","maxabs"]] = "minmax",
        targets:Literal["context"] = "context",
        using: Optional[int] = None) -> 'Environments':
        """Scale and shift features.

        Args:
            shift: The statistic to use to shift each context feature.
            scale: The statistic to use to scale each context feature.
            target: The target data we wish to scale in the environment.
            using: The number of interactions to use when calculating statistics.

        Remarks:
            For example, `scale('mean', 'std')` would standardize all context features
            while `scale('med', 'iqr')` would apply what sklearn calls a RobustScaler
            to all context features.

        Returns:
            An Environments object.
        """
        if isinstance(targets,str): targets = [targets]
        for t in targets: self = self.filter(Scale(shift, scale, t, using))
        return self

    def impute(self,
        stats: Union[Literal["mean","median","mode"],Sequence[Literal["mean","median","mode"]]] = "mean",
        indicator:bool = True,
        using: Optional[int] = None) -> 'Environments':
        """Impute missing data.

        Args:
            stats: The statistic to use for imputation. If a sequence is provided they
                will be applied in order. This is useful if different statistics should
                be applied for different types.
            indicator: Indicates whether a binary feature should be added for missingness.
            using: The number of interactions to use to calculate imputation statistics.

        Returns:
            An Environments object.
        """
        if isinstance(stats,str): stats = [stats]
        envs = self
        for stat in stats:
            envs = self.filter(Impute(stat, indicator, using))
        return envs

    def where(self,*,
            n_interactions: Union[int,Tuple[Optional[int],Optional[int]]] = None,
            n_actions: Union[int,Tuple[Optional[int],Optional[int]]] = None) -> 'Environments':
        """Select for characteristics.

        Args:
            n_interactions: The min, max or exact number of interactions an Environment must have.
            n_actions: The min, max or exact number of actions an interaction must have.

        Returns:
            An Environments object.
        """
        return self.filter(Where(n_interactions=n_interactions,n_actions=n_actions))

    def noise(self,
        context: Union[Tuple[str,float,float],Callable[[float,CobaRandom], float]] = None,
        action : Union[Tuple[str,float,float],Callable[[float,CobaRandom], float]] = None,
        reward : Union[Tuple[str,float,float],Callable[[float,CobaRandom], float]] = None,
        seed   : Union[int,Sequence[int]] = 1) -> 'Environments':
        """Add noise to values.

        Args:
            context: A distribution with shape parameters or a callable that returns a noisy value.
            action: A distribution with shape parameters or a callable that returns a noisy value.
            reward: A distribution with shape parameters or a callable that returns a noisy value.
            seed: The seed for all random values. If a sequence then multiple environments will be
                created using separate noise values.

        Remarks:
            Supported distribution with shapes are:
                * random integer: ('i',`inclusive min`,`inclusive max`)
                * random gaussian: ('g',`mean`,`std`)

        Returns:
            An Environments object.
        """
        if isinstance(seed,(int,float)): seed = [seed]
        return self.filter([Noise(context,action,reward,s) for s in seed])

    def flatten(self) -> 'Environments':
        """Flatten contexts and actions.

        Examples:
            An interaction {'context': [[1,2],3]} would become {'context':[1,2,3]}.

        Returns:
            An Environments object.
        """
        return self.filter(Flatten())

    def materialize(self) -> 'Environments':
        """Materialize and cache all environments.

        Remarks:
            Ideal for stateful environments such as Jupyter Notebook where
            environments can be saved in memory and re-used between experiments.

        Returns:
            An Environments object.
        """
        #we use pipes.cache directly because the environment cache will copy
        #which we don't need when we materialize an environment in memory

        nocache = lambda p: not isinstance(p,pipes.Cache) or p.protected
        materialized = []
        for env in map(self._finalize,self._envs):
            if not isinstance(env[-1],pipes.Cache):
                env = Pipes.join(*filter(nocache,env), pipes.Cache(None,True))
                list(env.read()) #force read to pre-load cache
            materialized.append(env)

        return Environments(materialized)

    def grounded(self, n_users: int, n_normal:int, n_words:int, n_good:int, seed:int=1) -> 'Environments':
        """Transform simulated interactions to IGL interactions.

        Args:
            n_users: The number of users in the grounded environment.
            n_normal: The number of users with normal grounded behavior.
            n_words: The number of potential feedback words for users.
            n_good: The number of words that mean good out of the n_words.
            seed: Seed for all random values.

        Remarks:
            See `here`__ for more on interaction grounded learning.

        Returns:
            An Environments object.

        __ https://proceedings.mlr.press/v139/xie21e.html
        """
        return self.filter(Grounded(n_users, n_normal, n_words, n_good, seed))

    def repr(self,
        cat_context:Literal["onehot","onehot_tuple","string"] = "onehot",
        cat_actions:Literal["onehot","onehot_tuple","string"] = "onehot") -> 'Environments':
        """Change representation of categorical data.

        Args:
            cat_context: How to represent categorical data in contexts.
            cat_actions: How to represent categorical data in actions.

        Returns:
            An Environments object.
        """
        return self.filter(Repr(cat_context,cat_actions))

    def batch(self, batch_size:int, batch_type: Literal['list','torch'] = 'list') -> 'Environments':
        """Batch interactions.

        Args:
            batch_size: The number of interactions in a batched interaction.
            batch_type: The type of batch for interaction values.

        Returns:
            An Environments object.
        """
        return self.filter(Batch(batch_size, batch_type))

    def chunk(self, cache:bool = True) -> 'Environments':
        """Add a chunk pipe.

        Args:
            cache: output before the chunk should be cached.

        Remarks:
            This is useful if an early part of an Environments pipeline
            takes considerable time to evaluate (e.g., `logged`). Placing
            a chunk after the long running filter means that the filter
            will only be executed one time and the results will be resued.

        Returns:
            An Environments object.
        """
        envs = Environments([Pipes.join(env,Chunk()) for env in self._envs])
        return envs.cache() if cache else envs

    def logged(self, learners: Union[Learner,Sequence[Learner]], seed:Optional[float] = 1.23) -> 'Environments':
        """Transform simulated interactions to logged interactions.

        Args:
            learners: The learners that will be used as the logging policy.
                An environment will be created for every learner provided.
            seed: The seed for used for all random number generation.

        Remarks:
            Adds 'action', 'reward', and 'probability' to interactions with 'context',
            'actions', and 'rewards'.

        Returns:
            An Environments object.
        """
        if not isinstance(learners, collections.abc.Sequence): learners = [learners]
        return self.filter([Logged(learner, seed) for learner in learners ])

    def unbatch(self):
        """Unbatch interactions.

        Remarks:
            The `unbatch` command is the inverse of `batch`.

        Returns:
            An Environments object.
        """

        return self.filter(Unbatch())

    def ope_rewards(self, rewards_type:Literal['IPS','DM','DR'] = None):
        """Transform logged interactions to simulated interactions.

        Args:
            rewards_type: How to estimate the rewards function from the logged
            data (i.e., inverse propensity score, direct method, or doubly robust).

        Remarks:
            Adds 'rewards' to interactions with 'context', 'action', 'reward',
            and 'probability'.

        Returns:
            An Environments object.
        """
        if isinstance(rewards_type,str): rewards_type = [rewards_type]
        return self.filter([OpeRewards(r) for r in rewards_type])

    def save(self, path: str, processes:int=1, overwrite:bool=False) -> 'Environments':
        """Save Environments to disk.

        Args:
            path: The location to save Environments (the file will be a zip archive).
            processes: The number of process to use when generating environments.
            overwrite: Indicate if an existing file at Path should be overwritten.

        Returns:
            An Environments object.
        """

        self_envs = list(self)
        if Path(path).exists():
            try:
                path_envs   = Environments.from_save(path)
                path_params = [e.params for e in path_envs]
                self_params = [e.params for e in self_envs]

                try:
                    while path_params:
                        param_index_in_self = self_params.index(path_params.pop())
                        self_params.pop(param_index_in_self)
                        self_envs.pop(param_index_in_self)
                except ValueError:
                    #there is a param in the file that isn't in self
                    is_equal = False
                else:
                    is_equal = True

                if is_equal and not self_envs:
                    return path_envs
                if not is_equal and overwrite:
                    Path(path).unlink()
                if not is_equal and not overwrite:
                    raise CobaException("The Environments save file does not match the actual Environments and overwite is False.")

            except BadZipFile:
                if overwrite:
                    Path(path).unlink()
                else:
                    raise CobaException("The given save file appears to be corrupted. Please check it and delete if it is unusable.")

        objs  = CobaMultiprocessor(EnvironmentsToObjects(),processes)
        disk  = ObjectsToZipMember(path)
        decor = [StampLog()] if processes ==1 else [NameLog(), StampLog()]

        CobaContext.logger = DecoratedLogger([ExceptLog()], CobaContext.logger, decor)
        disk.write(objs.filter(self_envs))
        CobaContext.logger = CobaContext.logger.undecorate()

        return Environments.from_save(path)

    def cache(self) -> 'Environments':
        """Add a caching pipe.

        Remarks:
            Results from earlier steps in the pipeline
            will re-used during later pipeline steps.

        Returns:
            An Environments object.
        """
        return Environments([Pipes.join(env,Cache(25)) for env in self._envs])

    def filter(self, filter: Union[EnvironmentFilter,Sequence[EnvironmentFilter]]) -> 'Environments':
        """Apply custom filter to Environments.

        Args:
            filter: The filters to apply to self. If a list of filters is
                provided then a new pipeline is created for each filter.

        Returns:
            An Environments object.
        """
        filters = filter if isinstance(filter, collections.abc.Sequence) else [filter]
        return Environments([Pipes.join(env,f) for env in self._envs for f in filters])

    def _finalize(self, env: Environment) -> Environment:
        is_finalize = lambda e: isinstance(e,BatchSafe) and isinstance(e._filter,Finalize)
        return env if any(map(is_finalize,env)) else Pipes.join(env,BatchSafe(Finalize()))

    def __getitem__(self, index:Any) -> Union[Environment,'Environments']:
        if isinstance(index,slice):
            return Environments(self._envs[index])
        else:
            return self._finalize(self._envs[index])

    def __iter__(self) -> Iterator[Environment]:
        return iter(map(self._finalize,self._envs))

    def __len__(self) -> int:
        return len(self._envs)

    def __add__(self, other: 'Environments') -> 'Environments':
        return Environments(self._envs+other._envs)

    def __str__(self) -> str:
        return "\n".join([f"{i+1}. {e}" for i,e in enumerate(self)])

    def _ipython_display_(self):
        #pretty print in jupyter notebook (https://ipython.readthedocs.io/en/stable/config/integrating.html)
        print(str(self))
