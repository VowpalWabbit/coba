from coba.contexts import CobaContext
import coba as cb

def main():
    CobaContext.experiment.processes = 1
    CobaContext.cacher.cache_directory = "./coba_cache"

    filename = "cb_oml_150.zip"

    # run first time as online
    cb.Environments.from_openml(data_id=150, take=128).logged([cb.learners.VowpalLearner("--cb_explore_adf")]).save(
        filename
    )

    # load environment from file
    env = cb.Environments.from_save(filename)

    assert type(env) == cb.environments.core.Environments
    assert env[0].params['openml_data'] == 150

    # use explore_eval to evaluate exploration in offline fashion
    lrn = [cb.learners.VowpalLearner('--cb_explore_adf --explore_eval')]
    cb.experiments.Experiment(env, lrn).run().plot_learners()

    return

if __name__ == "__main__":
    main()
