# Bandit Benchmarking

This repository contains python modules designed to benchmark the performance of bandit algorithms.


# Code Architecture

Here is a summary of the package architecture:

```
Types:
  * State  = Union[str,float,Sequence[Union[str,float]]]
  * Action = Union[str,float,Sequence[Union[str,float]]]
  * Reward = float

Interfaces
  * Round
    > state: Optional[State]
    > actions: Sequence[Action]
    > rewards: Sequence[Reward]

  * Result
    > iteration_stats: Sequence[Stats]
    > progressive_stats: Sequence[Stats]
    > predicate_stats(predicate: Callable[[Tuple[int,int,float]],bool]) -> Stats

  * Game
    > rounds: Iterable[Round]

  * Solver
    > choose(state: State, actions: Sequence[Action]) -> int
    > learn(state: State, action: Action, reward: Reward)
   
  * Benchmark
    > Evaluate(solver_factory: Callable[[],Solver]) -> Result    
```