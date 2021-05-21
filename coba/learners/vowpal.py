"""The vowpal module contains classes to make it easier to interact with pyvw.

TODO Add unittests
"""

import collections

from os import devnull
from typing import Any, Dict, Union, Sequence, overload

from vowpalwabbit import pyvw

from coba.utilities import PackageChecker, redirect_stderr
from coba.simulations import Context, Action
from coba.learners.core import Learner, Key

def _features_format(features: Union[Context,Action]) -> str:
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
        return " ". join([_feature_format(k,v) for k,v in features.items() if v is not None and v != 0 ])

    if isinstance(features, tuple) and len(features) == 2 and isinstance(features[0], tuple):
        return " ". join([_feature_format(k,v) for k,v in zip(features[0], features[1]) if v is not None and v != 0 ])

    if not isinstance(features, collections.Sequence):
        features = (features,)

    if isinstance(features, collections.Sequence):
        return " ". join([_feature_format(i,f) for i,f in enumerate(features) if f is not None and f != 0 ])

    raise Exception("We were unable to determine an appropriate vw context format.")

def _feature_format(name: Any, value: Any) -> str:
    """Convert a feature into the proper format for pyvw.

    Args:
        name: The name of the feature.
        value: The value of the feature.

    Remarks:
        In feature formatting we prepend a "name" to each feature. This makes it possible
        to compare features across actions/contexts. See the definition of `Features` at 
        the top of https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format for more info.
    """

    return f"{name}:{value}" if isinstance(value,(int,float)) else f"{value}"

class VowpalLearner(Learner):
    """A learner using Vowpal Wabbit's contextual bandit command line interface.

    Remarks:
        This learner requires that the Vowpal Wabbit package be installed. This package can be
        installed via `pip install vowpalwabbit`. To learn more about solving contextual bandit
        problems with Vowpal Wabbit see https://vowpalwabbit.org/tutorials/contextual_bandits.html
        and https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms.
    """

    @overload
    def __init__(self, *, epsilon: float, adf: bool = True, seed: int = None) -> None:
        """Instantiate a VowpalLearner.
        Args:
            epsilon: A value between 0 and 1. If provided exploration will follow epsilon-greedy.
        """
        ...

    @overload
    def __init__(self, *, bag: int, adf: bool = True, seed: int = None) -> None:
        """Instantiate a VowpalLearner.
        Args:
            bag: An integer value greater than 0. This value determines how many separate policies will be
                learned. Each policy will be learned from bootstrap aggregation, making each policy unique. 
                For each choice one policy will be selected according to a uniform distribution and followed.
        """
        ...

    @overload
    def __init__(self, *, cover: int, seed: int = None) -> None:
        """Instantiate a VowpalLearner.
        Args:
            cover: An integer value greater than 0. This value value determines how many separate policies will be
                learned. These policies are learned in such a way to explicitly optimize policy diversity in order
                to control exploration. For each choice one policy will be selected according to a uniform distribution
                and followed. For more information on this algorithm see Agarwal et al. (2014).
        References:
            Agarwal, Alekh, Daniel Hsu, Satyen Kale, John Langford, Lihong Li, and Robert Schapire. "Taming 
            the monster: A fast and simple algorithm for contextual bandits." In International Conference on 
            Machine Learning, pp. 1638-1646. 2014.
        """
        ...

    @overload
    def __init__(self, *, softmax: float, seed: int = None) -> None:
        """Instantiate a VowpalLearner.
        Args:
            softmax: An exploration parameter with 0 indicating uniform exploration is desired and infinity
                indicating that no exploration is desired (aka, greedy action selection only). For more info
                see `lambda` at https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms.
        """
        ...

    @overload
    def __init__(self, *, adf: bool, args:str) -> None:
        ...

    def __init__(self,  **kwargs) -> None:
        """Instantiate a VowpalLearner with the requested VW learner and exploration."""

        PackageChecker.vowpalwabbit('VowpalLearner.__init__')

        self._params = {}
        interactions = "--interactions ssa --interactions sa --ignore_linear s"

        if 'epsilon' in kwargs:
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

        else:
            self._adf  = kwargs['adf']
            self._args = kwargs['args']

        if 'seed' in kwargs:
            self._args += f" --random_seed {kwargs['seed']}"

        self._actions: Any = None
        self._vw           = None
        
    @property
    def family(self) -> str:
        """The family of the learner.

        See the base class for more information
        """
        
        return f"vw"

    @property
    def params(self) -> Dict[str, Any]:
        """The parameters of the learner.
        
        See the base class for more information
        """

        return {'args': self._args}

    def predict(self, key: Key, context: Context, actions: Sequence[Action]) -> Sequence[float]:
        """Determine a PMF with which to select the given actions.

        Args:
            key: The key identifying the interaction we are choosing for.
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The probability of taking each action. See the base class for more information.
        """

        if self._vw is None:
            cb_explore = "" if "cb_explore" in self._args else "--cb_explore_adf" if self._adf else f"--cb_explore {len(actions)}"

            self._args = cb_explore + " " + self._args
            
            # vowpal has an annoying warning that is written to stderr whether or not we provide
            # the --quiet flag. Therefore, we temporarily redirect all stderr output to null so that
            # this warning isn't shown during creation. It should be noted this isn't thread-safe
            # so if you are here because of strange problems with threads we may just need to suck
            # it up and accept that there will be an obnoxious warning message.
            with open(devnull, 'w') as f, redirect_stderr(f):
                self._vw   = pyvw.vw(self._args + " --quiet")

        probs = self._vw.predict(self._predict_format(context, actions))

        self._set_actions(key, actions)

        if not self._adf:
            #in this case probs are always in order of self._actions but we want to return in order of actions
            return [ probs[self._actions.index(action)] for action in actions ]
        else:
            return probs

    def learn(self, key: Key, context: Context, action: Action, reward: float, probability: float) -> None:
        """Learn from the given interaction.

        Args:
            key: The key identifying the interaction this observed reward came from.
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
            probability: The probability that the given action was taken.
        """

        actions = self._get_actions(key)
        
        self._vw.learn(self._learn_format(probability, actions, context, action, reward))

    def _predict_format(self, context, actions) -> str:
        if self._adf:
            vw_context = None if context is None else f"shared |s {_features_format(context)}"
            vw_actions = [ f"|a {_features_format(a)}" for a in actions]
            return "\n".join(filter(None,[vw_context, *vw_actions]))
        else:
            return f"|s {_features_format(context)}"

    def _learn_format(self, prob, actions, context, action, reward) -> str:
        if self._adf:
            vw_context   = None if context is None else f"shared |s {_features_format(context)}"
            vw_rewards  = [ "" if a != action else f"0:{-reward}:{prob}" for a in actions ]
            vw_actions  = [ f"{a}:1" for a in actions]
            vw_observed = [ f"{r} |a {a}" for r,a in zip(vw_rewards,vw_actions) ]
            return "\n".join(filter(None,[vw_context, *vw_observed]))
        else:
            return f"{actions.index(action)+1}:{-reward}:{prob} |s {_features_format(context)}"

    def _set_actions(self, key: Key, actions: Sequence[Action]) -> None:
        
        if self._actions is None and not self._adf:
            self._actions = actions
        
        if self._actions is None and self._adf:
            self._actions = {}

        if self._adf:
            self._actions[key] = actions

    def _get_actions(self, key) -> Sequence[Action]:
        return self._actions.pop(key) if isinstance(self._actions, dict) else self._actions