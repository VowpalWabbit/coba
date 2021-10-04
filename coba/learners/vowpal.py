"""The vowpal module contains classes to make it easier to interact with pyvw.

TODO Add unittests
"""

import re
import collections

from os import devnull
from typing import Any, Dict, Union, Sequence, overload, cast, Optional, Tuple

from coba.config import CobaException
from coba.utilities import PackageChecker, redirect_stderr
from coba.simulations import Context, Action
from coba.learners.core import Learner, Probs, Info

class VowpalLearner(Learner):
    """A learner using Vowpal Wabbit's contextual bandit command line interface.

    Remarks:
        This learner requires that the Vowpal Wabbit package be installed. This package can be
        installed via `pip install vowpalwabbit`. To learn more about solving contextual bandit
        problems with Vowpal Wabbit see https://vowpalwabbit.org/tutorials/contextual_bandits.html
        and https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms.
    """

    @overload
    def __init__(self, *, epsilon: float = 0.1, adf: bool = True, seed: Optional[int] = 1, precision: int=5) -> None:
        """Instantiate a VowpalLearner.
        Args:
            epsilon: A value between 0 and 1. If provided, exploration will follow epsilon-greedy.
            adf: Indicate whether cb_explore or cb_explore_adf should be used.
            seed: The seed used by VW to generate any necessary random numbers.
            precision: Indicate how many decimal places to round to when passing example strings to VW.
        """
        ...

    @overload
    def __init__(self, *, bag: int, adf: bool = True, seed: Optional[int] = 1, precision: int=5) -> None:
        """Instantiate a VowpalLearner.
        Args:
            bag: An integer value greater than 0. This value determines how many separate policies will be
                learned. Each policy will be learned from bootstrap aggregation, making each policy unique. 
                When predicting one policy will be selected according to a uniform distribution and followed.
            adf: Indicate whether cb_explore or cb_explore_adf should be used.
            seed: The seed used by VW to generate any necessary random numbers.
            precision: Indicate how many decimal places to round to when passing example strings to VW.
        """
        ...

    @overload
    def __init__(self, *, cover: int, seed: Optional[int] = 1, precision: int=5) -> None:
        """Instantiate a VowpalLearner.
        Args:
            cover: An integer value greater than 0. This value value determines how many separate policies will be
                learned. These policies are learned in such a way to explicitly optimize policy diversity in order
                to control exploration. When predicting one policy will be selected according to a uniform distribution
                and followed. For more information on this algorithm see Agarwal et al. (2014).
            seed: The seed used by VW to generate any necessary random numbers.
            precision: Indicate how many decimal places to round to when passing example strings to VW.
        References:
            Agarwal, Alekh, Daniel Hsu, Satyen Kale, John Langford, Lihong Li, and Robert Schapire. "Taming 
            the monster: A fast and simple algorithm for contextual bandits." In International Conference on 
            Machine Learning, pp. 1638-1646. 2014.
        """
        ...

    @overload
    def __init__(self, *, softmax: float, seed: Optional[int] = 1, precision: int=5) -> None:
        """Instantiate a VowpalLearner.
        Args:
            softmax: An exploration parameter with 0 indicating uniform exploration is desired and infinity
                indicating that no exploration is desired (aka, greedy action selection only). For more info
                see `lambda` at https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms.
            seed: The seed used by VW to generate any necessary random numbers.
            precision: Indicate how many decimal places to round to when passing example strings to VW.
        """
        ...

    @overload
    def __init__(self, args:str, precision: int=5) -> None:
        ...
        """Instantiate a VowpalLearner.
        Args:
            args: Command line argument to instantiates a Vowpal Wabbit contextual bandit learner. 
                For examples and documentation on how to instantiate VW learners from command line arguments see 
                https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms. It is assumed that
                either the --cb_explore or --cb_explore_adf flag is used. When formatting examples for VW context
                features are namespaced with `s` and action features, when relevant, are namespaced with with `a`.
            precision: Indicate how many decimal places to round to when passing example strings to VW.
        """

    def __init__(self, *args, **kwargs) -> None:
        """Instantiate a VowpalLearner with the requested VW learner and exploration."""

        PackageChecker.vowpalwabbit('VowpalLearner')

        interactions = "--interactions ssa --interactions sa --ignore_linear s"

        if not args and 'seed' not in kwargs:
            kwargs['seed'] = 1

        if not args and all(e not in kwargs for e in ['epsilon', 'softmax', 'bag', 'cover']): 
            kwargs['epsilon'] = 0.1

        if len(args) > 0:
            self._adf  = "--cb_explore_adf" in args[0]
            self._args = cast(str,args[0])

            self._args = re.sub("--cb_explore_adf\s+", '', self._args, count=1)
            self._args = re.sub("--cb_explore(\s+\d+)?\s+", '', self._args, count=1)

        elif 'epsilon' in kwargs:
            self._adf  = kwargs.get('adf', True)
            self._args = interactions + f" --epsilon {kwargs['epsilon']}"

        elif 'softmax' in kwargs:
            self._adf  = True
            self._args = interactions + f" --softmax --lambda {kwargs['softmax']}"

        elif 'bag' in kwargs:
            self._adf  = kwargs.get('adf',True)
            self._args = interactions + f" --bag {kwargs['bag']}"

        elif 'cover' in kwargs:
            self._adf  = False
            self._args = interactions + f" --cover {kwargs['cover']}"

        if 'seed' in kwargs and kwargs['seed'] is not None:
            self._args += f" --random_seed {kwargs['seed']}"

        self._precision                 = kwargs.get('precision',5) 
        self._actions: Sequence[Action] = []
        self._vw                        = None

    @property
    def params(self) -> Dict[str, Any]:
        """The parameters of the learner.
        
        See the base class for more information
        """

        return {"family": "vw", 'args': self._create_format(None)}

    def predict(self, context: Context, actions: Sequence[Action]) -> Tuple[Probs, Info]:
        """Determine a PMF with which to select the given actions.

        Args:
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The probability of taking each action. See the base class for more information.
        """

        if self._vw is None:
            from vowpalwabbit import pyvw #type: ignore
            # vowpal has an annoying warning that is written to stderr whether or not we provide
            # the --quiet flag. Therefore, we temporarily redirect all stderr output to null so that
            # this warning isn't shown during creation. It should be noted this isn't thread-safe
            # so if you are here because of strange problems with threads we may just need to suck
            # it up and accept that there will be an obnoxious warning message.
            with open(devnull, 'w') as f, redirect_stderr(f):
                self._vw = pyvw.vw(self._create_format(actions) + " --quiet")

            if not self._adf:
                self._actions = actions

        assert self._vw is not None, "Something went wrong and vw was not initialized"

        probs = self._vw.predict(self._predict_format(context, actions))

        if self._adf:
            return probs, actions
        else:
            if any(action not in self._actions for action in actions) or len(actions) != len(self._actions):
                raise CobaException("It appears that actions are changing between predictions. When this happens you need to use VW's `--cb_explore_adf`.")

            #in this case probs will be in order of self._actions but we want to return in order of actions
            return [ probs[self._actions.index(action)] for action in actions ], self._actions

    def learn(self, context: Context, action: Action, reward: float, probability: float, info: Info) -> None:
        """Learn from the given interaction.

        Args:
            key: The key identifying the interaction this observed reward came from.
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
            probability: The probability that the given action was taken.
        """

        assert self._vw is not None, "You must call predict before learn in order to initialize the VW learner"
        
        self._vw.learn(self._learn_format(probability, info, context, action, reward))

    def _round(self, value: float) -> float:
        return round(value, self._precision)

    def _feature_format(self, name: Any, value: Any) -> str:
        """Convert a feature into the proper format for pyvw.

        Args:
            name: The name of the feature.
            value: The value of the feature.

        Remarks:
            In feature formatting we prepend a "name" to each feature. This makes it possible
            to compare features across actions/contexts. See the definition of `Features` at 
            the top of https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format for more info.
        """

        return f"{name}:{self._round(value)}" if isinstance(value,(int,float)) else f"{value}"

    def _features_format(self, features: Union[Context,Action]) -> str:
        """convert features into the proper format for pyvw.

        Args:
            features: The feature set we wish to convert to pyvw representation.

        Returns:
            The context in pyvw representation.

        Remarks:
            Note, using the enumeration index for action features below only works if all actions
            have the same number of features. If some actions simply leave out features in their
            feature array a more advanced method may need to be implemented in the future...
        """

        if isinstance(features, str):
            return features + ":1" #type: ignore

        if isinstance(features, dict):
            return " ". join([self._feature_format(k,v) for k,v in features.items() if v is not None and v != 0 ])

        if isinstance(features, tuple) and len(features) == 2 and isinstance(features[0], tuple) and isinstance(features[1], tuple):
            return " ". join([self._feature_format(k,v) for k,v in zip(features[0], features[1]) if v is not None and v != 0 ])

        if not isinstance(features, collections.Sequence):
            features = (features,)

        if isinstance(features, collections.Sequence):
            return " ". join([self._feature_format(i,f) for i,f in enumerate(features) if f is not None and f != 0 ])

        raise Exception("We were unable to determine an appropriate vw context format.")

    def _create_format(self, actions) -> str:
        
        cb_explore = "--cb_explore_adf" if self._adf else f"--cb_explore {len(actions)}" if actions else "--cb_explore"
        
        return cb_explore + " " + self._args

    def _predict_format(self, context, actions) -> str:
        if self._adf:
            vw_context = None if context is None else f"shared |s {self._features_format(context)}"
            vw_actions = [ f"|a {self._features_format(a)}" for a in actions]
            return "\n".join(filter(None,[vw_context, *vw_actions]))
        else:
            return f"|s {self._features_format(context)}"

    def _learn_format(self, prob, actions, context, action, reward) -> str:
        
        vw_reward = lambda a: "" if a != action else f"{actions.index(action)+1}:{self._round(-reward)}:{self._round(prob)} "

        if self._adf:
            vw_context  = None if context is None else f"shared |s {self._features_format(context)}"
            vw_rewards  = [ vw_reward(a) for a in actions ]
            vw_actions  = [ f"|a {self._features_format(a)}" for a in actions]
            vw_observed = [ f"{r}{a}" for r,a in zip(vw_rewards,vw_actions) ]
            return "\n".join(filter(None,[vw_context, *vw_observed]))
        else:
            return f"{vw_reward(action)}|s {self._features_format(context)}"