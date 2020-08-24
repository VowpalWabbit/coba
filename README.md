# Coba

An agile and collaborative benchmarking framework for contextual bandit research.

### How do you benchmark?

Think for a second about the last time you benchmarked an algorithm or dataset and ask yourself

 1. Was it easy to add new data sets?
 2. Was it easy to add new algorithms?
 3. Was it easy to create, run and share benchmarks?

# The Coba Way
 
 Coba was built from the ground up to answer the three questions above with a yes.
 
 ## Adding New Data Sets
 
 ### Creating Simulations From Classification Data Sets
 
 Classification data sets are an easy way to quickly evaluate CB algorithms. So long as the data set is in CSV format it can easily be turned into a contextual bandit simulation with `ClassificationSimulation.from_csv`. This method works both with local files and files available over http. When requesting over http Coba even allows you to cache a version of the file locally for fast re-runs later.
 
 ### Creating Simulations From Generative Models
 
 Certain domains have well defined models that an agent has to make decisions within. To add datasets from these domains one can use `LambdaSimulation` to define random functions that returns contexts, actions and rewards according to any defined distribution. 
 
 ### Creating Custom Simulations From Scratch
 
 If more customization is needed than what is offered above then a new simulation class can be created. Implementing such a class requires no inheritance or complex registration. One simply has to satisfy Coba's `Simulation` interface as shown below (with `Interaction` included for reference):
 
```python
class Simulation:
 
    @property
    def interactions(self) -> Sequence[Interaction]:
        """The sequence of interactions in a simulation."""
        ...
    
    def rewards(self, choices: Sequence[Tuple[Key,int]] ) -> Sequence[float]:
        """The observed rewards for interactions (identified by key) and their selected action indexes."""
        ...
    
class Interaction:
    @property
    def context(self) -> Hashable:
        """The interaction's context description."""
        ...

    @property
    def actions(self) -> Sequence[Hashable]:
        """The interactions's available actions."""
        ...
    
    @property
    def key(self) -> Key:
        """A unique key identifying the interaction."""
        ...
```

 ## Adding New Algorithms
 
 A number of algorithms have been implemented out of the box including epsilon-greedy, VowpalWabbit bagging, VowpWabbit softmax, VowpalWabbit cover, VowpalWabbit RND and upper confidence bounding. Adding algorithms is simply a matter of satisfying the `Learner` interface as shown below:
 
```python
class Learner:
    """The interface for Learner implementations."""

    @abstractmethod
    def choose(self, key: int, context: Hashable, actions: Sequence[Hashable]) -> int:
        """Choose which action index to take."""
        ...

    @abstractmethod
    def learn(self, key: int, context: Hashable, action: Hashable, reward: float) -> None:
        """Learn about the result of an action that was taken in a context.
        ...
```
 
 ## Creating and Sharing Benchmarks
 
 Benchmarks are created using a json configuration file. In the configuration file one defines the data sets to include in the benchmark, the location of the datasets, how to break the datasets into batches, what random seed to use and if the data sets should be randomized. By placing all these characteristics into a single configuration file creating, modifying and sharing benchmarks is simply a matter of editing this file and emailing it to another researcher.
 
```json
 {
    "templates"   : { "shuffled_openml_classification": { "seed":777, "type":"classification", "from": {"format":"openml", "id":"$id"} }},
    "batches"     : { "count":51 },
    "ignore_first": true,
    "simulations" : [
        {"template":"shuffled_openml_classification", "$id":3},
        {"template":"shuffled_openml_classification", "$id":6},
        {"template":"shuffled_openml_classification", "$id":8}
    ]
}
```
 
 ## Examples
 
 An examples directory is included in the repository with a number of code demonstrations and benchmark demonstrations. These examples show how to create benchmarks, evaluate learners against them and plot the results.
