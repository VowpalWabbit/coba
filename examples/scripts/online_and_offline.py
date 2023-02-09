import coba as cb

def main():
    cb.CobaContext.experiment.processes = 1
    cb.CobaContext.cacher.cache_directory = "./coba_cache"

    filename = "cb_oml_150.zip"

    # run first time as online
    env = cb.Environments.from_openml(data_id=150, take=128).logged(cb.VowpalLearner("--cb_explore_adf")).save(
        filename
    )

    assert type(env) == cb.Environments
    assert env[0].params['openml_data'] == 150

    # use explore_eval to evaluate exploration in offline fashion
    # https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Explore-Eval
    lrn = cb.VowpalLearner('--cb_explore_adf --explore_eval')
    
    #offline performance
    cb.Experiment(env, lrn, evaluation_task=cb.SimpleEvaluation(predict=False)).run()
    lrn.finish()

    #online performance
    cb.Result.from_logged_envs(env).plot_learners()

    return

if __name__ == "__main__":
    main()
