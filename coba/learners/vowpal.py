"""The vowpal module contains classes to make it easier to interact with pyvw."""

import re
import collections

from itertools import repeat
from numbers import Number
from os import devnull
from typing import Any, Dict, Union, Sequence, overload, cast, Optional, Tuple

from coba.config import CobaException
from coba.utilities import PackageChecker, redirect_stderr
from coba.simulations import Context, Action

from coba.learners.core import Learner, Probs, Info

Coba_Feature    = Union[str,Number]
Coba_Features   = Union[Dict[str,Coba_Feature], Sequence[Coba_Feature], Coba_Feature]
Vowpal_Features = Sequence[Union[Tuple[Union[str,int],float],str,int]]

class VowpalMediator:
    
    @staticmethod
    def prep_features(features: Coba_Features) -> Vowpal_Features:

        def tuple_prep(key,val):
            prepped_key = None
            prepped_val = None

            if isinstance(key,str):
                prepped_key = key
            elif isinstance(key,int):
                prepped_key = key
            elif isinstance(key,float) and key.is_integer():
                prepped_key = int(key)
            
            if isinstance(val,Number):
                prepped_val = val

            assert prepped_key is not None and prepped_val is not None, f"{(key,val)} is an invalid VW feature."
            return (prepped_key,prepped_val)                

        def sequence_prep(tuple_sequence: Sequence[Tuple[Any,Any]]):
            return [ v if isinstance(v,str) else tuple_prep(k,v) for k,v in tuple_sequence if v != 0 ]

        if not features:
            return []
        elif isinstance(features, dict):
            return sequence_prep(features.items())
        elif isinstance(features, str):
            return [features]
        elif isinstance(features, Number):
            return [(0,features)]
        elif isinstance(features, collections.Sequence) and isinstance(features[0], tuple) and len(features[0]) == 2:
            return sequence_prep(features)
        elif isinstance(features, collections.Sequence) and not isinstance(features[0], tuple):
            return sequence_prep(enumerate(features))
                    
        raise Exception("Unrecognized features passed to VowpalLearner.")

    @staticmethod
    def make_learner(args:str):
        PackageChecker.vowpalwabbit('VVW.make_learner')
        from vowpalwabbit import pyvw

        return pyvw.vw(args)

    @staticmethod
    def make_example(vw, ns:Dict[str,Vowpal_Features], label:Optional[str], label_type:int):
        PackageChecker.vowpalwabbit('VW.make_example')
        from vowpalwabbit.pyvw import example

        ns = { k:v for k,v in ns.items() if v != [] }

        ex = example(vw, ns, label_type)
        if label: ex.set_label_string(label)
        ex.setup_example()
        
        return ex

class VowpalLearner(Learner):
    """A learner using Vowpal Wabbit's contextual bandit command line interface.

    Remarks:
        This learner requires that the Vowpal Wabbit package be installed. This package can be
        installed via `pip install vowpalwabbit`. To learn more about solving contextual bandit
        problems with Vowpal Wabbit see https://vowpalwabbit.org/tutorials/contextual_bandits.html
        and https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms.
    """

    @overload
    def __init__(self, *, epsilon: float = 0.1, adf: bool = True, seed: Optional[int] = 1) -> None:
        """Instantiate a VowpalLearner.
        Args:
            epsilon: A value between 0 and 1. If provided, exploration will follow epsilon-greedy.
            adf: Indicate whether cb_explore or cb_explore_adf should be used.
            seed: The seed used by VW to generate any necessary random numbers.
        """
        ...

    @overload
    def __init__(self, *, bag: int, adf: bool = True, seed: Optional[int] = 1) -> None:
        """Instantiate a VowpalLearner.
        Args:
            bag: An integer value greater than 0. This value determines how many separate policies will be
                learned. Each policy will be learned from bootstrap aggregation, making each policy unique. 
                When predicting one policy will be selected according to a uniform distribution and followed.
            adf: Indicate whether cb_explore or cb_explore_adf should be used.
            seed: The seed used by VW to generate any necessary random numbers.
        """
        ...

    @overload
    def __init__(self, *, cover: int, seed: Optional[int] = 1) -> None:
        """Instantiate a VowpalLearner.
        Args:
            cover: An integer value greater than 0. This value value determines how many separate policies will be
                learned. These policies are learned in such a way to explicitly optimize policy diversity in order
                to control exploration. When predicting one policy will be selected according to a uniform distribution
                and followed. For more information on this algorithm see Agarwal et al. (2014).
            seed: The seed used by VW to generate any necessary random numbers.
        
        References:
            Agarwal, Alekh, Daniel Hsu, Satyen Kale, John Langford, Lihong Li, and Robert Schapire. "Taming 
            the monster: A fast and simple algorithm for contextual bandits." In International Conference on 
            Machine Learning, pp. 1638-1646. 2014.
        """
        ...

    @overload
    def __init__(self, *, softmax: float, seed: Optional[int] = 1) -> None:
        """Instantiate a VowpalLearner.
        Args:
            softmax: An exploration parameter with 0 indicating uniform exploration is desired and infinity
                indicating that no exploration is desired (aka, greedy action selection only). For more info
                see `lambda` at https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms.
            seed: The seed used by VW to generate any necessary random numbers.
        """
        ...

    @overload
    def __init__(self, args:str) -> None:
        ...
        """Instantiate a VowpalLearner.
        Args:
            args: Command line argument to instantiates a Vowpal Wabbit contextual bandit learner. 
                For examples and documentation on how to instantiate VW learners from command line arguments see 
                https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms. It is assumed that
                either the --cb_explore or --cb_explore_adf flag is used. When formatting examples for VW, context
                features are namespaced with `s` and action features, when relevant, are namespaced with `a`.
        """

    def __init__(self, *args, **kwargs) -> None:
        """Instantiate a VowpalLearner with the requested VW learner and exploration."""

        interactions = "--interactions ssa --interactions sa --ignore_linear s"

        if not args and 'seed' not in kwargs:
            kwargs['seed'] = 1

        if not args and all(e not in kwargs for e in ['epsilon', 'softmax', 'bag', 'cover']): 
            kwargs['epsilon'] = 0.1

        if len(args) > 0:

            assert "--cb" in args[0], "VowpalLearner was instantiated without a cb learner being defined."

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

        return {"family": "vw", 'args': self._cli_args([None])}

    def predict(self, context: Context, actions: Sequence[Action]) -> Tuple[Probs, Info]:
        """Determine a PMF with which to select the given actions.

        Args:
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The probability of taking each action. See the base class for more information.
        """

        if self._vw is None:
             #type: ignore
            # vowpal has an annoying warning that is written to stderr whether or not we provide
            # the --quiet flag. Therefore, we temporarily redirect all stderr output to null so that
            # this warning isn't shown during creation. It should be noted this isn't thread-safe
            # so if you are here because of strange problems with threads we may just need to suck
            # it up and accept that there will be an obnoxious warning message.
            with open(devnull, 'w') as f, redirect_stderr(f):
                self._vw = VowpalMediator.make_learner(self._cli_args(actions) + " --quiet")

            if not self._adf:
                self._actions = actions

        if not self._adf and (len(actions) != len(self._actions) or any(a not in self._actions for a in actions)):
            raise CobaException("Actions are only allowed to change between predictions with `--cb_explore_adf`.")

        info = (actions if self._adf else self._actions)

        shared  = self._shared(context)
        adfs    = self._adfs(actions)

        if self._adf:
            probs = self._vw.predict(self._examples(shared,adfs))
        else:
            probs = self._vw.predict(self._example(shared))

        return probs, info

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

        actions = info
        shared  = self._shared(context)
        adfs    = self._adfs(actions)
        labels  = self._labels(actions, action, reward, probability)
        label   = labels[actions.index(action)]

        if self._adf:
            self._vw.learn(self._examples(shared, adfs, labels))
        else:
            self._vw.learn(self._example(shared, label))

    def _cli_args(self, actions: Optional[Sequence[Action]]) -> str:

        if self._adf:
            return "--cb_explore_adf" + " " + self._args
        else:
            return f"--cb_explore {len(actions) if actions else ''}" + " " + self._args

    def _examples(self, shared: Dict[str,Any], adfs: Sequence[Dict[str,Any]] = None, labels: Sequence[str] = None):

        shared = { ns:VowpalMediator.prep_features(v) for ns,v in shared.items() }
        examples = []
        for adf,label in zip(adfs, labels or repeat(None)):
            adf = { ns:VowpalMediator.prep_features(v) for ns,v in adf.items() }
            examples.append(self._example({**shared,**adf}, label))
        
        return examples

    def _example(self, ns, label=None):
        return VowpalMediator.make_example(self._vw, ns, label, 4)

    def _labels(self,actions,action,reward:float,prob:float) -> Sequence[Optional[str]]:
        return [ f"{i+1}:{round(1-reward,5)}:{round(prob,5)}" if a == action else None for i,a in enumerate(actions)]

    def _shared(self,context) -> Dict[str,Any]:
        return {} if not context else { 's': VowpalMediator.prep_features(context) }

    def _adfs(self,actions) -> Sequence[Dict[str,Any]]:
        return [ {'a': VowpalMediator.prep_features(a)} for a in actions]
