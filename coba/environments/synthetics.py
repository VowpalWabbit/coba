import math

from statistics import mean
from itertools import count, islice, cycle
from typing import Sequence, Dict, Tuple, Any, Callable, Optional, overload, Iterable, Literal

from coba.random import CobaRandom
from coba.exceptions import CobaException
from coba.primitives import DiscreteReward, Context, Action
from coba.encodings import InteractionsEncoder, OneHotEncoder

from coba.environments.primitives import Environment, SimulatedInteraction

class LambdaSimulation(Environment):
    """A simulation created from generative lambda functions."""

    @overload
    def __init__(self,
        n_interactions: Optional[int],
        context       : Callable[[int               ],Context         ],
        actions       : Callable[[int,Context       ],Sequence[Action]],
        reward        : Callable[[int,Context,Action],float           ]) -> None:
        """Instantiate a LambdaSimulation.

        Args:
            n_interactions: An optional integer indicating the number of interactions in the simulation.
            context: A function that should return a context given an index in `range(n_interactions)`.
            actions: A function that should return all valid actions for a given index and context.
            reward: A function that should return the reward for the index, context and action.
        """

    @overload
    def __init__(self,
        n_interactions: Optional[int],
        context       : Callable[[int               ,CobaRandom],Context         ],
        actions       : Callable[[int,Context       ,CobaRandom],Sequence[Action]],
        reward        : Callable[[int,Context,Action,CobaRandom],float           ],
        seed          : int) -> None:
        """Instantiate a LambdaSimulation.

        Args:
            n_interactions: An optional integer indicating the number of interactions in the simulation.
            context: A function that should return a context given an index and random state.
            actions: A function that should return all valid actions for a given index, context and random state.
            reward: A function that should return the reward for the index, context, action and random state.
            seed: An integer used to seed the random state in order to guarantee repeatability.
        """

    def __init__(self,n_interactions,context,actions,reward,seed=None) -> None:
        """Instantiate a LambdaSimulation."""

        self._n_interactions = n_interactions
        self._context        = context
        self._actions        = actions
        self._reward         = reward
        self._make_rng       = seed is not None
        if seed is not None: self._seed = seed

    @property
    def params(self) -> Dict[str, Any]:
        params = { "env_type": "LambdaSimulation" }

        if hasattr(self, '_seed'):
            params["seed"] = self._seed

        return params

    def read(self) -> Iterable[SimulatedInteraction]:
        rng = None if not self._make_rng else CobaRandom(self._seed)

        _context = lambda i    : self._context(i    ,rng) if rng else self._context(i   )
        _actions = lambda i,c  : self._actions(i,c  ,rng) if rng else self._actions(i,c )
        _reward  = lambda i,c,a: self._reward (i,c,a,rng) if rng else self._reward (i,c,a)

        for i in islice(count(), self._n_interactions):
            context  = _context(i)
            actions  = _actions(i, context)
            rewards  = [ _reward(i, context, action) for action in actions]

            yield {'context':context,'actions':actions,'rewards':DiscreteReward(actions,rewards) }

    def __str__(self) -> str:
        return "LambdaSimulation"

    class Spoof:
        def __init__(self, interactions, _params, _str, _name) -> None:
            self._str           = _str
            self._interactions  = interactions
            self._params        = _params
            type(self).__name__ = _name

        @property
        def params(self) -> Dict[str, Any]:
            return self._params

        def read(self):
            return self._interactions

        def __str__(self) -> str:
            return self._str

    def __reduce__(self) -> Tuple[object, ...]:
        if self._n_interactions is not None and self._n_interactions < 5000:
            #This is an interesting idea but maybe too wink-wink nudge-nudge in practice. It causes a weird flow
            #in the logs that look like bugs. It also causes unexpected lags because IO is happening at strange
            #places and in a manner that can cause thread locks.
            return (LambdaSimulation.Spoof, (list(self.read()), self.params, str(self), type(self).__name__ ))
        else:
            message = (
                "It is not possible to pickle a LambdaSimulation due to its use of lambda methods in the constructor. "
                "This error occured because an experiment containing a LambdaSimulation tried to execute on multiple processes. "
                "If this is neccesary there are three options to get around this limitation: (1) run your experiment "
                "on a single process rather than multiple, (2) re-design your LambdaSimulation as a class that inherits "
                "from LambdaSimulation and implements __reduce__ (see coba.environments.simulations.LinearSyntheticSimulation "
                "for an example), or (3) specify a finite number for n_interactions in the LambdaSimulation constructor (this "
                "allows us to create the interactions in memory ahead of time and convert to an in-memory simulation to pickle).")
            raise CobaException(message)

class NeighborsSyntheticSimulation(LambdaSimulation):
    """A synthetic simulation whose reward values are determined by neighborhoods.

    The simulation's rewards are determined by the location of given context and action pairs. These locations
    indicate which neighborhood the context action pair belongs to. Neighborhood rewards are determined by
    random assignment.
    """

    def __init__(self,
        n_interactions:int,
        n_actions:int = 10,
        n_context_features:int = 10,
        n_action_features:int = 10,
        n_neighborhoods:int = 10,
        seed: int = 1) -> None:
        """Instantiate a NeighborsSyntheticSimulation.

        Args:
            n_interactions: The number of interactions the simulation should have.
            n_actions: The number of actions each interaction should have.
            n_context_features: The number of features each context should have.
            n_action_features: The number of features each action should have.
            n_neighborhoods: The number of neighborhoods the simulation should have.
            seed: The random number seed used to generate all contexts and action rewards.
        """

        self._args = (n_interactions, n_actions, n_context_features, n_action_features, n_neighborhoods, seed)

        self._n_interactions  = n_interactions
        self._n_actions       = n_actions
        self._n_context_feats = n_context_features
        self._n_action_feats  = n_action_features
        self._n_neighborhoods = n_neighborhoods
        self._seed            = seed

        rng = CobaRandom(self._seed)

        def context_gen():
            return tuple(rng.gausses(n_context_features,0,1)) if n_context_features else None

        def actions_gen():
            if not n_action_features:
                return OneHotEncoder().fit_encodes(range(n_actions))
            else:
                return [ tuple(rng.gausses(n_action_features,0,1)) for _ in range(n_actions) ]

        contexts               = list(set([ context_gen() for _ in range(self._n_neighborhoods) ]))
        context_actions        = { c: actions_gen() for c in contexts }
        context_action_rewards = { (c,a):rng.random() for c in contexts for a in context_actions[c] }

        context_iter = iter(cycle(contexts))

        def context(index:int):
            return next(context_iter)

        def actions(index:int, context:Tuple[float,...]):
            return context_actions[context]

        def reward(index:int, context:Tuple[float,...], action:Tuple[int,...]):
            return context_action_rewards[(context,action)]

        return super().__init__(self._n_interactions, context, actions, reward)

    @property
    def params(self) -> Dict[str, Any]:
        return {**super().params, "env_type": "NeighborsSynthetic", "n_neighborhoods": self._n_neighborhoods }

    def __str__(self) -> str:
        return f"NeighborsSynth(A={self._n_actions},c={self._n_context_feats},a={self._n_action_feats},N={self._n_neighborhoods},seed={self._seed})"

    def __reduce__(self) -> Tuple[object, ...]:
        return (NeighborsSyntheticSimulation, self._args)

class LinearSyntheticSimulation(LambdaSimulation):
    """A synthetic simulation whose rewards are linear with respect to the given reward features.

    The simulation's rewards are linear with respect to the requrested reward features. When no context
    or action features are requested these terms are removed from the requested reward features.
    """

    def __init__(self,
        n_interactions:int,
        n_actions:int = 10,
        n_context_features:int = 10,
        n_action_features:int = 10,
        reward_features:Sequence[str] = ["a","xa"],
        seed:int = 1) -> None:
        """Instantiate a LinearSyntheticSimulation.

        Args:
            n_interactions: The number of interactions the simulation should have.
            n_actions: The number of actions each interaction should have.
            n_context_features: The number of features each context should have.
            n_action_features: The number of features each action should have.
            reward_features: The features in the simulation's linear reward function.
            seed: The random number seed used to generate all features, weights and noise in the simulation.
        """

        if n_actions < 2:
            raise CobaException("Linear synthetic environments must have at least two actions")

        self._args = (n_interactions, n_actions, n_context_features, n_action_features, reward_features, seed)

        self._n_actions          = n_actions
        self._n_context_features = n_context_features
        self._n_action_features  = n_action_features
        self._reward_features    = reward_features
        self._seed               = seed

        if not self._n_context_features:
            reward_features = list(set(filter(None,[ f.replace('x','') for f in reward_features])))

        if not self._n_action_features:
            reward_features = list(set(filter(None,[ f.replace('a','') for f in reward_features])))

        rng           = CobaRandom(seed)
        feats_encoder = InteractionsEncoder(reward_features)
        one_hot_acts  = OneHotEncoder().fit_encodes(range(n_actions))

        feature_count = len(feats_encoder.encode(x=[1]*n_context_features,a=[1]*n_action_features))
        weight_parts  = n_actions if n_context_features and not n_action_features else 1
        weight_count  = feature_count or n_actions

        self._weights = [ rng.gausses(weight_count,0,1) for _ in range(weight_parts) ]
        self._biases  = [ 0 ] * weight_parts

        def context(index:int, rng: CobaRandom) -> Context:
            return rng.randoms(n_context_features,-1.5,1.5) or None

        def actions(index:int,context: Context, rng: CobaRandom) -> Sequence[Action]:
            return [ rng.randoms(n_action_features,-1.5,1.5) for _ in range(n_actions) ] if n_action_features else one_hot_acts

        def reward(index:int,context:Context, action:Action, rng: CobaRandom) -> float:

            F = feats_encoder.encode(x=context,a=action) if feature_count else action
            W = self._weights[0 if weight_parts==1 else action.index(1)]
            b = self._biases [0 if weight_parts==1 else action.index(1)]

            return b+sum([w*f for w,f in zip(W,F)])

        interaction_rewards = [ [reward(i,c,a,None) for a in actions(i,c,rng)] for i in range(100) for c in [ context(i,rng)] ]

        parts_rewards  = [sum(interaction_rewards,[])] if weight_parts == 1 else list(zip(*interaction_rewards))
        global_rewards = sum(interaction_rewards,[])
        global_scale   = max(global_rewards)-min(global_rewards)

        is_global_scale = weight_count == 1

        for i, part_rewards in enumerate(parts_rewards):
            scale            = global_scale if is_global_scale else (max(part_rewards)-min(part_rewards)) or 1
            self._weights[i] = [w/scale for w in self._weights[i]]
            self._biases[i]  = 0.5-mean(part_rewards)/scale

        super().__init__(n_interactions, context, actions, reward, seed=self._seed)

    @property
    def params(self) -> Dict[str, Any]:
        return {**super().params, "env_type":"LinearSynthetic", "reward_features": self._reward_features }

    def __reduce__(self) -> Tuple[object, ...]:
        return (LinearSyntheticSimulation, self._args)

    def __str__(self) -> str:
        return f"LinearSynth(A={self._n_actions},c={self._n_context_features},a={self._n_action_features},R={self._reward_features},seed={self._seed})"

class KernelSyntheticSimulation(LambdaSimulation):
    """A synthetic simulation whose reward function is created from kernel basis functions.

    Kernel functions are created using random exemplar points generated at initialization and fixed for all time.
    """

    def __init__(self,
        n_interactions:int,
        n_actions:int = 10,
        n_context_features:int = 10,
        n_action_features:int = 10,
        n_exemplars:int = 10,
        kernel: Literal['linear','polynomial','exponential','gaussian'] = 'gaussian',
        degree: int = 2,
        gamma: float = 1,
        seed: int = 1) -> None:
        """Instantiate a KernelSyntheticSimulation.

        Args:
            n_interactions: The number of interactions the simulation should have.
            n_actions: The number of actions each interaction should have.
            n_context_features: The number of features each context should have.
            n_action_features: The number of features each action should have.
            n_exemplars: The number of exemplar context-action pairs.
            kernel: The family of the kernel basis functions.
            degree: This argument is only relevant when using polynomial kernels.
            gamma: This argument is only relevant when using exponential kernels.
            seed: The random number seed used to generate all features, weights and noise in the simulation.
        """

        if n_actions < 2:
            raise CobaException("Kernel synthetic environments must have at least two actions.")

        if n_exemplars < 1:
            raise CobaException("Kernel synthetic environments must have at least one exemplar.")

        self._args = (n_interactions, n_actions, n_context_features, n_action_features, n_exemplars, kernel, degree, gamma, seed)

        self._n_actions          = n_actions
        self._n_context_features = n_context_features
        self._n_action_features  = n_action_features
        self._n_exemplars        = n_exemplars
        self._seed               = seed
        self._kernel             = kernel
        self._degree             = degree
        self._gamma              = gamma

        rng = CobaRandom(seed)

        exemplar_parts = n_actions if not n_action_features and n_context_features else 1
        exemplar_count = n_exemplars
        exemplar_feats = (n_action_features+n_context_features) or n_actions

        one_hot_acts = OneHotEncoder().fit_encodes(range(n_actions))
        exemplar_gen = lambda n: [  tuple(rng.randoms(exemplar_feats,-1.5,1.5)) for _ in range(n)]

        self._exemplars = [ exemplar_gen(exemplar_count) for _ in range(exemplar_parts) ]
        self._weights   = [ rng.gausses(exemplar_count, 0, 1) for _ in range(exemplar_parts) ]
        self._biases    = [ 0 ] * exemplar_parts

        def context(index:int, rng: CobaRandom) -> Context:
            return tuple(rng.randoms(n_context_features,-1.5,1.5)) if n_context_features else None

        def actions(index:int, context: Context, rng: CobaRandom) -> Sequence[Action]:
            return [ tuple(rng.randoms(n_action_features,-1.5,1.5)) for _ in range(n_actions) ] if n_action_features else one_hot_acts

        def reward(index:int, context:Context, action:Action, rng: CobaRandom) -> float:
            part_index = action.index(1) if exemplar_parts > 1                           else 0
            context    = context         if n_context_features                           else []
            action     = action          if n_action_features or not n_context_features  else []

            f = list(context)+list(action)
            E = self._exemplars[part_index]
            W = self._weights  [part_index]
            b = self._biases   [part_index]

            if kernel == "linear":
                K = lambda x1,x2: self._linear_kernel(x1,x2)
            if kernel == "polynomial":
                K = lambda x1,x2: self._polynomial_kernel(x1,x2,self._degree)
            if kernel == "exponential":
                K = lambda x1,x2: self._exponential_kernel(x1,x2,self._gamma)
            if kernel == "gaussian":
                K = lambda x1,x2: self._gaussian_kernel(x1,x2,self._gamma)

            return b + sum([w*K(e,f) for w,e in zip(W, E)])

        interaction_rewards = [ [reward(i,c,a,rng) for a in actions(i,c,rng)] for i in range(200) for c in [ context(i,rng)] ]

        parts_rewards  = [sum(interaction_rewards,[])] if exemplar_parts == 1 else list(zip(*interaction_rewards))
        global_rewards = sum(interaction_rewards,[])
        global_scale   = max(global_rewards)-min(global_rewards)

        is_global_scale = (kernel == 'linear' or kernel=='polynomial' and degree==1) and exemplar_feats==1

        for i, part_rewards in enumerate(parts_rewards):
            scale            = global_scale if is_global_scale else (max(part_rewards)-min(part_rewards)) or 1
            self._weights[i] = [w/scale for w in self._weights[i]]
            self._biases[i]  = 0.5-mean(part_rewards)/scale

        super().__init__(n_interactions, context, actions, reward, self._seed)

    @property
    def params(self) -> Dict[str, Any]:
        params = {**super().params, "env_type": "KernelSynthetic", "n_exemplars": self._n_exemplars, 'kernel': self._kernel}

        if self._kernel == "polynomial":
            params['degree'] = self._degree

        if self._kernel in ["exponential","gaussian"]:
            params['gamma'] = self._gamma

        return params

    def __reduce__(self) -> Tuple[object, ...]:
        return (KernelSyntheticSimulation, self._args)

    def __str__(self) -> str:
        return f"KernelSynth(A={self._n_actions},c={self._n_context_features},a={self._n_action_features},E={self._n_exemplars},K={self._kernel},seed={self._seed})"

    def _linear_kernel(self, F1: Sequence[float], F2: Sequence[float]) -> float:
        return sum([f1*f2 for f1,f2 in zip(F1,F2)])

    def _polynomial_kernel(self, F1: Sequence[float], F2: Sequence[float], degree:int) -> float:
        return (self._linear_kernel(F1,F2)+1)**degree

    def _exponential_kernel(self, F1: Sequence[float], F2: Sequence[float], gamma:float) -> float:
        return math.exp(-math.sqrt(sum([(f1-f2)**2 for f1,f2 in zip(F1,F2)]))/gamma)

    def _gaussian_kernel(self, F1: Sequence[float], F2: Sequence[float], gamma:float) -> float:
        return math.exp(-sum([(f1-f2)**2 for f1,f2 in zip(F1,F2)])/gamma)

class MLPSyntheticSimulation(LambdaSimulation):
    """A synthetic simulation whose reward function belongs to the MLP family.

    The MLP architecture has a single hidden layer with sigmoid activation and one output
    value calculated from a random linear combination of the hidden layer's output.
    """

    def __init__(self,
        n_interactions:int,
        n_actions:int = 10,
        n_context_features:int = 10,
        n_action_features:int = 10,
        seed: int = 1) -> None:
        """Instantiate an MLPSythenticSimulation.

        Args:
            n_interactions: The number of interactions the simulation should have.
            n_actions: The number of actions each interaction should have.
            n_context_features: The number of features each context should have.
            n_action_features: The number of features each action should have.
            seed: The random number seed used to generate all features, weights and noise in the simulation.
        """

        if n_actions < 2:
            raise CobaException("MLP synthetic environments must have at least two actions.")


        self._args = (n_interactions, n_actions, n_context_features, n_action_features, seed)

        self._n_actions          = n_actions
        self._n_context_features = n_context_features
        self._n_action_features  = n_action_features
        self._seed               = seed

        rng = CobaRandom(seed)

        input_layer_size  = n_context_features+n_action_features
        hidden_layer_size = 50

        self._bias = 0

        if input_layer_size:
            hidden_weights       = [ [ rng.gausses(input_layer_size,0,1.5) for _ in range(hidden_layer_size) ] for _ in range(1 if n_action_features else n_actions) ]
            hidden_activation    = lambda x: 1/(1+math.exp(-x)) #sigmoid activation
            hidden_output        = lambda inputs,weights: hidden_activation(sum([ i*w for i,w in zip(inputs,weights)]))
            self._output_weights = rng.gausses(hidden_layer_size)
        else:
            self._output_weights = rng.gausses(n_actions)

        def context(index:int, rng:CobaRandom) -> Context:
            return tuple(rng.gausses(n_context_features)) if n_context_features else None

        def actions(index:int, context: Context, rng:CobaRandom) -> Sequence[Action]:
            if n_action_features:
                return [ (rng.gausses(n_action_features)) for _ in range(n_actions)]
            else:
                return OneHotEncoder().fit_encodes(range(n_actions))

        def reward(index:int, context:Context, action:Action, rng:CobaRandom) -> float:

            #handles None context
            context = context or []

            if not n_action_features and not n_context_features:
                return self._bias + self._output_weights[action.index(1)]

            if n_action_features:
                I = list(context)+list(action)
                W = self._output_weights
                H = hidden_weights[0]
            else:
                I = list(context)
                W = self._output_weights
                H = hidden_weights[action.index(1)]

            hidden_outputs = [ hidden_output(I,h) for h in H]

            return self._bias + sum([w*hout for w,hout in zip(W, hidden_outputs) ])

        rewards = [ reward(i,c,a,rng) for i in range(100) for c in [ context(i,rng)] for a in actions(i,c,rng) ]

        m = mean(rewards)
        s = (max(rewards)-min(rewards)) or 1

        self._bias = 0.5-m/s
        self._output_weights = [ w/s for w in self._output_weights ]

        super().__init__(n_interactions, context, actions, reward, self._seed)

    @property
    def params(self) -> Dict[str, Any]:
        return {**super().params, "env_type": "MLPSynthetic" }

    def __reduce__(self) -> Tuple[object, ...]:
        return (MLPSyntheticSimulation, self._args)

    def __str__(self) -> str:
        return f"MLPSynth(A={self._n_actions},c={self._n_context_features},a={self._n_action_features},seed={self._seed})"
