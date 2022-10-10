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

import math
import itertools
import time

from typing import Optional, Sequence, Any

class CobaRandom:
    """A random number generator that is consistent across python implementations."""

    def __init__(self, seed: Optional[float] = None) -> None:
        """Instantiate a CobaRandom.

        Args:
            seed: The seed to use when starting random number generation.

        Remarks:
            The values for a,c,m below are taken from L'ecuyer (1999). In this paper he notes that
            these values for LCG should have an overall period of m though the period of lower bits
            will be much shorter. A solution he offers to this problem is to use much larger m (e.g.,
            2**128) and then only use the the top n most significant digits. For now, we aren't doing that.

        References:
            L'ecuyer, Pierre. "Tables of linear congruential generators of different sizes
            and good lattice structure." Mathematics of Computation 68.225 (1999): 249-260.
        """

        self._m = 2**30 #this must be a power of 2 for our math to work correctly
        self._a = 116646453
        self._c = 9

        self._m_minus_1 = self._m-1
        self._generator = self._next()

        if isinstance(seed,int) or (isinstance(seed,float) and seed.is_integer()):
            self._seed = int(seed)
        else:
            self._seed = int.from_bytes(str(seed or time.time()).encode('utf-8'),"big") % 2**20

    def randoms(self, n:int, min:float=0, max:float=1) -> Sequence[float]:
        """Generate `n` uniform random numbers in [`min`,`max`).

        Args:
            n: How many random numbers should be generated.
            min: The minimum value for the random numbers.
            max: The maximum value for the random numbers.

        Returns:
            The `n` generated random numbers in [`min`,`max`].
        """
        if n < 0 or not isinstance(n, int):
            raise ValueError("n must be an integer greater than or equal to 0")
        
        if min == 0 and max == 1:
            return list(itertools.islice(self._generator,n))
        elif min == 0:
            return [ max*r for r in itertools.islice(self._generator,n) ]
        else:
            r_range = max-min
            return [ min+r_range*r for r in itertools.islice(self._generator,n) ]

    def random(self, min:float=0, max:float=1) -> float:
        """Generate a uniform random number in [`min`,`max`].

        Args:
            min: The minimum value for the random numbers.
            max: The maximum value for the random numbers.

        Returns:
            The generated random number in [`min`,`max`].
        """
        return min+(max-min)*next(self._generator)        

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
        if n < 2: return sequence
        l = list(sequence)

        #i goes from 0 to n-2
        #j is always i <= j < n
        for i,r in enumerate(itertools.islice(self._generator,n-1)):
            j = int(i+r*(n-i)) 
            l[i], l[j] = l[j], l[i]
        return l

    def randint(self, a:int, b:int) -> int:
        """Generate a uniform random integer in [a, b].

        Args:
            a: The inclusive lower bound for the random integer.
            b: The inclusive upper bound for the random integer.
        """

        return a+int((b-a+1)*next(self._generator))

    def randints(self, n:int, a:int, b:int) -> Sequence[int]:
        """Generate `n` uniform random integers in [a, b].

        Args:
            n: The number of random integers to generate.
            a: The inclusive lower bound for the random integer.
            b: The inclusive upper bound for the random integer.
        """
        b=b+1
        if a == 0:
            return [int(b*r) for r in itertools.islice(self._generator,n)]
        else:
            r_range = b-a
            return [int(r_range*r) + a for r in itertools.islice(self._generator,n)]

    def choice(self, seq: Sequence[Any], weights:Sequence[float] = None) -> Any:
        """Choose a random item from the given sequence.

        Args:
            seq: The sequence to pick randomly from.
            weights: The proportion by which seq is selected from.
        """

        if weights is None:
            return seq[int(len(seq)*next(self._generator))]
        else:
            if sum(weights) == 0:
                raise ValueError("The sum of weights cannot be zero.")

            cdf = list(itertools.accumulate(weights))
            rng = self.random() * sum(weights)

            return seq[[rng <= c for c in cdf].index(True)]

    def gauss(self, mu:float=0, sigma:float=1) -> float:
        """Generate a random number from N(mu,sigma).

        Args:
            mu: The expectation of the distribution we are drawing from.
            sigma: The standard deviation of the distribution we are drawing form.

        Returns:
            A random number drawn from N(mu,sigma).
        """
        return self.gausses(1, mu, sigma)[0]

    def gausses(self, n:int, mu:float=0, sigma:float=1) -> Sequence[float]:
        """Generate `n` independent random numbers from N(mu,sigma).

        Args:
            n: The number of random numbers to generate.
            mu: The expectation of the distribution we are drawing from.
            sigma: The standard deviation of the distribution we are drawing form.

        Returns:
            The `n` random numbers drawn from N(mu,sigma).
        """
        #Box-muller transform. Generates independent pseudo-normal random variable pairs.

        if n == 0: return []

        n_pairs = math.ceil(n/2)

        Rs = [math.sqrt(-2*math.log(u)) for u in itertools.islice(self._generator,n_pairs)]
        Ss = [2*math.pi*u               for u in itertools.islice(self._generator,n_pairs)]

        Ns = []
        for R,S in zip(Rs,Ss):
            Ns.append(mu+sigma*R*math.cos(S))
            Ns.append(mu+sigma*R*math.sin(S))

        if len(Ns) > n: Ns.pop()

        return Ns

    def _next(self) -> Sequence[int]:
        """Generate `n` uniform random numbers in [0,m-1]

        Random numbers are generated using a linear congruential generator

        Args:
            n: The number of random numbers to generate.

        Returns:
            The `n` generated random numbers in [0,m-1].
        """

        while True:
            #when _m is a power of 2 this is equal to modulo _m
            self._seed = (self._a * self._seed + self._c) & (self._m_minus_1)
            yield self._seed/self._m

_random = CobaRandom()

def seed(seed: Optional[float]) -> None:
    """Set the seed for generating random numbers in this module.

    Args:
        seed: The seed for generating random numbers.

    Remarks:
        Note, this seed does not affect random numbers generated by the standard library
    """

    global _random

    _random = CobaRandom(seed)

def random(min:float=0, max:float=1) -> float:
    """Generate a uniform random number in [`min`,`max`].

    Args:
        min: The minimum value for the random numbers.
        max: The maximum value for the random numbers.

    Returns:
        A uniform random number in [`min`,`max`].
    """
    return _random.random(min,max)

def randoms(n: int, min:float=0, max:float=1) -> Sequence[float]:
    """Generate `n` uniform random numbers in [`min`,`max`].

    Args:
        n: How many uniform random numbers should be generated.
        min: The minimum value for the random numbers.
        max: The maximum value for the random numbers.

    Returns:
        The `n` uniform random numbers in [`min`,`max`].
    """

    return _random.randoms(n,min,max)

def gauss(mu:float=0, sigma:float=1) -> float:
    """Generate a random number from N(mu,sigma).

    Returns:
        A random number drawn from N(mu,sigma).
    """

    return _random.gauss(mu,sigma)

def gausses(n:int, mu:float=0, sigma:float=1) -> Sequence[float]:
    """Generate `n` independent random numbers from N(mu,sigma).

    Args:
        n: How many random numbers should be generated
        mu: The expectation of the distribution we are drawing from.
        sigma: The standard deviation of the distribution we are drawing form.

    Returns:
        The `n` random numbers drawn from N(mu,sigma).
    """

    return _random.gausses(n,mu,sigma)

def randint(a:int, b:int) -> int:
    """Generate a uniform random integer in [a, b].

    Args:
        a: The inclusive lower bound for the random integer.
        b: The inclusive upper bound for the random integer.
    """

    return _random.randint(a,b)

def randints(n:int, a:int, b:int) -> Sequence[int]:
    """Generate `n` uniform random integers in [a, b].

    Args:
        n: The number of random integers to generate.
        a: The inclusive lower bound for the random integer.
        b: The inclusive upper bound for the random integer.
    """

    return _random.randints(n,a,b)

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
