# Bandit Benchmarking

This repository contains python modules designed to benchmark the performance of bandit algorithms.

The following high-level interfaces are leveraged:

```
Round
  > Contextual Features [CF]
  > Action Features [AF]
  > Action Rewards [AR]

 
Game
  > Rounds

   
ContextualBandit [CB]
  > Choose(CF, AF)
  > Learn(CF, AF, AR)   
   
Benchmarker
  > Evaluate(CB)
```