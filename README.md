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
    > values: Sequence[float]
    > errors: Sequence[Optional[float]]

  * Game
    > rounds: Iterable[Round]

  * Solver
    > choose(state: State, actions: Sequence[Action]) -> int
    > learn(state: State, action: Action, reward: Reward)
   
  * Benchmark
    > Evaluate(solver_factory: Callable[[],Solver]) -> Result
    
Implementations
  * Round -- a simple DTO representing rounds
  * Result -- a simple DTO representing the result of a benchmark
  * MemoryGame -- A game where the rounds are defined in memory
  * LambdaGame -- A game where the rounds are defined via lambda functions
  * ClassificationGame -- A game where the rounds are defined via features and labels
  * RandomSolver -- A solver that chooses actions at random and learns nothing
  * LambdaSolver -- A solver whose choose and learn are implemented via lambda functions
  * ProgressiveBenchmark -- A benchmark that calculates the expected progressive benchmark given games
  * TraditionalBenchmark -- A benchmark that calculates expected reward from fixed policy iterations
```

# To Do:
  * Implement real Games (e.g., ClassificationGame.from_openml_arff() or ClassificationGame.from_openml_csv_json()??)
  * Implement Benchmark that queries website RestAPI
  * Implement real Solvers
  * Create graphical reporting module for Result