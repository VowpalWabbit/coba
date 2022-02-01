import math

from itertools import count, repeat, islice
from typing import Sequence, Dict, Tuple, Any, Callable, Optional, overload, Iterable

from coba.exceptions import CobaException
from coba.random import CobaRandom
from coba.encodings import InteractionsEncoder, OneHotEncoder

from coba.environments.primitives import Context, Action
from coba.environments.simulated.primitives import SimulatedEnvironment, SimulatedInteraction, MemorySimulation

class LambdaSimulation(SimulatedEnvironment):
    """A simulation created from generative lambda functions.

    Remarks:
        This implementation is useful for creating a simulation from defined distributions.
    """

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
        self._seed           = seed

        self._params = {} if seed is None else {"lambda_seed": seed}

    @property
    def params(self) -> Dict[str, Any]:
        return dict(self._params)

    def read(self) -> Iterable[SimulatedInteraction]:
        rng = None if self._seed is None else CobaRandom(self._seed)

        _context = lambda i    : self._context(i    ,rng) if rng else self._context(i) 
        _actions = lambda i,c  : self._actions(i,c  ,rng) if rng else self._actions(i,c)
        _reward  = lambda i,c,a: self._reward (i,c,a,rng) if rng else self._reward(i,c,a)  

        for i in islice(count(), self._n_interactions):
            context  = _context(i)
            actions  = _actions(i, context)
            rewards  = [ _reward(i, context, action) for action in actions]

            yield SimulatedInteraction(context, actions, rewards=rewards)

    def __str__(self) -> str:
        return "LambdaSimulation"

    class Spoof(MemorySimulation):

        def __init__(self, interactions, _params, _str, _name) -> None:
            type(self).__name__ = _name
            self._str           = _str
            super().__init__(interactions, _params)

        def read(self):
            return self._interactions

        def __str__(self) -> str:
            return self._str

    def __reduce__(self) -> Tuple[object, ...]:
        if self._n_interactions is not None and self._n_interactions < 1000:
            #This is an interesting idea but maybe too wink-wink nudge-nudge in practice. It causes weird flow
            #in the logs that looks like bugs and lags because IO is happening at strange places in a manner that
            #can cause thread locks.
            return (LambdaSimulation.Spoof, (list(self.read()), self.params, str(self), type(self).__name__ ))
        else:
            message = (
                "It is not possible to pickle a LambdaSimulation due to its use of lambda methods in the constructor. "
                "This error occured because an experiment containing a LambdaSimulation tried to execute on multiple processes. "
                "If this is neccesary there are three options to get around this limitation: (1) run your experiment "
                "on a single process rather than multiple, (2) re-design your LambdaSimulation as a class that inherits "
                "from LambdaSimulation and implements __reduce__ (see coba.environments.simulations.LinearSyntheticSimulation "
                "for an example), or (3) specify a finite number for n_interactions in the LambdaSimulation constructor (this "
                "allows us to create the interactions in memory ahead of time and convert to a MemorySimulation when pickling).")
            raise CobaException(message)

class LinearSyntheticSimulation(LambdaSimulation):
    """A simple synthetic simulation useful for debugging learning algorithms. 
            
    The simulation's rewards are linear with respect to the given features and their cross terms. In the case 
    that no context or action features are requested interaction terms are calculted by assuming actions or 
    contexts have a constant feature of 1.
    """

    def __init__(self, 
        n_interactions: int = 500, 
        n_actions: int = 10, 
        n_context_feats:int = 10, 
        n_action_feats:int = 10, 
        r_noise_var:float = 1/1000,
        cross_terms: Sequence[str] = ["a","xa"],
        seed:int=1) -> None:
        """Instantiate a LinearSyntheticSimulation.
        
        Args:
            n_interactions: The number of interactions the simulation should have.
            n_actions: The number of actions each interaction should have.
            n_context_feats: The number of features each interaction context should have.
            n_action_feats: The number of features each action should have.
            r_noise_var: The variance of the noise term added to the expected reward value.
            cross_terms: The action and context feature cross products to calculate expected reward value.
            seed: The random number seed used to generate all features, weights and noise in the simulation. 
        """

        self._args = (n_interactions, n_actions, n_context_feats, n_action_feats, r_noise_var, cross_terms, seed)

        self._n_actions          = n_actions
        self._n_context_features = n_context_feats
        self._n_action_features  = n_action_feats
        self._seed               = seed
        self._r_noise_var        = r_noise_var
        self._X                  = cross_terms

        rng = CobaRandom(seed)
        X_encoder = InteractionsEncoder(self._X)

        dummy_context = list(range(max(1,n_context_feats)))
        dummy_action  = list(range(n_action_feats)) if n_action_feats else list(range(n_actions))
        feature_count = len(X_encoder.encode(x=dummy_context,a=dummy_action))

        normalize = lambda X: [ rng.random()*x/sum(X) for x in X]
        identity  = lambda n: OneHotEncoder().fit_encodes(range(n))

        weights = normalize(rng.randoms(feature_count)) # we normalize weights so that reward will be in [0,1]
        actions = ( [rng.randoms(n_action_feats) for _ in range(n_actions)] for _ in count()) if n_actions else repeat(identity(n_actions))
        A_ident = None if n_action_feats else identity(n_actions)

        def context(index:int, rng: CobaRandom) -> Context:
            return rng.randoms(n_context_feats) if n_context_feats else None

        def actions(index:int, context: Context, rng: CobaRandom) -> Sequence[Action]:
            return  [rng.randoms(n_action_feats) for _ in range(n_actions)] if n_action_feats else A_ident

        def reward(index:int, context:Context, action:Action, rng: CobaRandom) -> float:

            W = weights
            X = context or [1]
            A = action
            F = X_encoder.encode(x=X,a=A)

            r = sum([w*f for w,f in zip(W,F)])
            e = (rng.random()-1/2)*math.sqrt(12)*math.sqrt(self._r_noise_var)
            
            return min(1,max(0,r+e))

        super().__init__(n_interactions, context, actions, reward, seed)

    @property
    def params(self) -> Dict[str, Any]:
        return { 
            "n_A"    : self._n_actions,
            "n_C_phi": self._n_context_features,
            "n_A_phi": self._n_action_features,
            "r_noise": self._r_noise_var,
            "X"      : self._X,
            "seed"   : self._seed
        }

    def __reduce__(self) -> Tuple[object, ...]:
        return (LinearSyntheticSimulation, self._args)

    def __str__(self) -> str:
        return f"LinearSynth(A={self._n_actions},c={self._n_context_features},a={self._n_action_features},X={self._X},seed={self._seed})"

class LocalSyntheticSimulation(LambdaSimulation):
    """A simple simulation useful for debugging learning algorithms. 
        
        The simulation's rewards are determined by the location of given context and action pairs with respect to a 
        small set of pre-generated exemplar context,action pairs. Location is currently defined as equality though 
        it could potentially be extended to support any number of metric based similarity kernels. The "local" in 
        the name is due to its close relationship to 'local regression'.
    """

    def __init__(self,
        n_interactions: int = 500,
        n_contexts: int = 200,
        n_context_feats: int = 2,
        n_actions: int = 10,
        seed: int = 1) -> None:
        """Instantiate a LocalSyntheticSimulation.
        
        Args:
            n_interactions: The number of interactions the simulation should have.
            n_contexts: The number of unique contexts the simulation should contain.
            n_context_feats: The number of features each interaction context should have.
            n_actions: The number of actions each interaction should have.
            seed: The random number seed used to generate all contexts and action rewards.
        """

        self._args = (n_interactions, n_contexts, n_context_feats, n_actions, seed)

        self._n_interactions     = n_interactions
        self._n_context_features = n_context_feats
        self._n_contexts         = n_contexts
        self._n_actions          = n_actions
        self._seed               = seed

        rng = CobaRandom(self._seed)

        contexts = [ tuple(rng.randoms(n_context_feats)) for _ in range(self._n_contexts) ]        
        actions  = OneHotEncoder().fit_encodes(range(n_actions))
        rewards  = {}

        for context in contexts:
            for action in actions:
                rewards[(context,action)] = rng.random()

        def context_generator(index:int, rng: CobaRandom):
            return rng.choice(contexts)

        def action_generator(index:int, context:Tuple[float,...], rng: CobaRandom):
            return actions

        def reward_function(index:int, context:Tuple[float,...], action: Tuple[int,...], rng: CobaRandom):
            return rewards[(context,action)]

        return super().__init__(self._n_interactions, context_generator, action_generator, reward_function, seed)

    @property
    def params(self) -> Dict[str, Any]:
        return { 
            "n_A"    : self._n_actions,
            "n_C"    : self._n_contexts,
            "n_C_phi": self._n_context_features,
            "seed"   : self._seed
        }

    def __str__(self) -> str:
        return f"LocalSynth(A={self._n_actions},C={self._n_contexts},c={self._n_context_features},seed={self._seed})"

    def __reduce__(self) -> Tuple[object, ...]:
        return (LocalSyntheticSimulation, self._args)
