"""A custom implementation of random number generation.

This module follows the pattern of the standard library's random module by creating
by instantiation an internal, global Random class and then referencing that in all
the public methods in to maintain state.

Remarks:
    This implementation has been made to guarantee the reproducibility of experiments
    according to integer seeds across all versions of Python. The standard implementation 
    of random within Python has had a few variations in implementation in the past and 
    could always change in the future, making randomization by seed potentially non-fixed.
"""

import random as std_random
import itertools

from typing import Optional, Sequence, Any, List

class CobaRandom:
    """A random number generator via a linear congruential generator."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Instantiate a CobaRandom.

        Args:
            seed: the seed to start random number generation.

        Remarks:
            The values for a,c,m below are taken from L’ecuyer (1999). In this paper he notes that
            these values for LCG should have an overall period of m though the period of lower bits 
            will be much shorter. A solution he offers to this problem is to use much larger m (e.g., 
            2**128) and then only use the the top n most significant digits. For now, we aren't doing that.
        
        References:
            L’ecuyer, Pierre. "Tables of linear congruential generators of different sizes 
            and good lattice structure." Mathematics of Computation 68.225 (1999): 249-260.
        """

        self._m = 2**30
        self._a = 116646453
        self._c = 9

        self._m_is_power_of_2 = (self._m % 2) == 0
        self._m_minus_1       = self._m-1

        self._seed: int = std_random.randint(0,self._m_minus_1) if seed is None else seed

    def randoms(self, n:int=1) -> Sequence[float]:
        """Generate `n` uniform random numbers in [0,1].

        Args:
            n: How many random numbers should be generated.

        Returns:
            The `n` generated random numbers in [0,1].
        """

        return [number/self._m_minus_1 for number in self._next(n)]

    def shuffle(self, sequence: Sequence[Any]) -> Sequence[Any]:
        """Shuffle the order of items in a sequence.

        Args:
            sequence: The sequence of items that are to be shuffled.

        Returns:
            A new sequence with the order of items shuffled.

        Remarks:
            This is the Richard Durstenfeld's method popularized by Donald Knuth in The Art of Computer 
            Programming. This algorithm is unbiased (i.e., all possible permutations are equally likely to occur).
        """

        n = len(sequence)
        r = self.randoms(n)
        l = list(sequence)

        for i in range(0,n-1):

            j = int(i + (r[i] * (n-i))) # i <= j <= n
            j = min(j, n-1)             # i <= j <= n-1 (min handles the edge case of r[i]==1 which would make j=n)
            
            l[i], l[j] = l[j], l[i]

        return l

    def random(self) -> float:
        """Generate a uniform random number in [0,1].

        Returns:
            The generated random number in [0,1].
        """
        return self.randoms(1)[0]

    def randint(self, a:int, b:int) -> int:
        """Generate a uniform random integer in [a, b].
        
        Args:
            a: The inclusive lower bound for the random integer.
            b: The inclusive upper bound for the random integer.
        """

        return min(int((b-a+1) * self.random()), b-a) + a

    def choice(self, seq: Sequence[Any], weights:Sequence[float] = None) -> Any:
        """Choose a random item from the given sequence.
        
        Args:
            seq: The sequence to pick randomly from.
            weights: The proportion by which seq is selected from.
        """
        
        if weights is None:
            return seq[self.randint(0, len(seq)-1)]
        else:

            if sum(weights) == 0:
                raise ValueError("The sum of weights cannot be zero.")

            cdf = list(itertools.accumulate(weights))
            rng = self.random() * sum(weights)

            return seq[[ rng <= c for c in cdf].index(True)]

    def _next(self, n: int) -> Sequence[int]:
        """Generate `n` uniform random numbers in [0,m-1]

        Random numbers are generated using a linear congruential generator

        Args:
            n: The number of random numbers to generate.

        Returns:
            The `n` generated random numbers in [0,m-1].
        """
        
        if n < 0 or not isinstance(n, int):
            raise ValueError("n must be an integer greater than or equal 0")

        numbers: List[int] = []

        for _ in range(n):

            #when _m is a power of 2 these two statements are equal to eachother
            if self._m_is_power_of_2:
                self._seed = int((self._a * self._seed + self._c) & (self._m_minus_1))
            else:
                self._seed = int((self._a * self._seed + self._c) % self._m)

            numbers.append(self._seed)

        return numbers

_random = CobaRandom()

def seed(seed: Optional[int]) -> None:
    """Set the seed for generating random numbers in this module.
    
    Args:
        seed: The seed for generating random numbers.

    Remarks:
        Note, this seed does not affect random numbers generated by the standard library
    """

    global _random

    _random = CobaRandom(seed)

def random() -> float:
    """Generate a uniform random number in [0,1]."""
    return _random.random()

def randoms(n: int) -> Sequence[float]:
    """Generate `n` uniform random numbers in [0,1].

    Args:
        n: How many random numbers should be generated.

    Returns:
        The `n` generated random numbers in [0,1].
    """

    return _random.randoms(n)

def randint(a:int, b:int) -> int:
    """Generate a uniform random integer in [a, b].
    
    Args:
        a: The inclusive lower bound for the random integer.
        b: The inclusive upper bound for the random integer.
    """
    
    return _random.randint(a,b)

def choice(seq: Sequence[Any], weights:Sequence[float]=None) -> Any:
    """Choose a random item from the given sequence.
    
    Args:
        seq: The sequence to pick randomly from.
        weights: The proportion by which seq is selected from.
    """
    
    return _random.choice(seq, weights)

def shuffle(array_like: Sequence[Any]) -> Sequence[Any]:
    """Shuffle the order of items in a sequence.

    Args:
        sequence: The sequence of items that are to be shuffled.

    Returns:
        A new sequence with the order of items shuffled.
    """

    return _random.shuffle(array_like)