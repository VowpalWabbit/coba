import coba as cb

def main():

    oml_id = 150
    filename = f"cb_oml_{oml_id}.zip"

    # run first time as online
    online_vw_args = "--cb_explore_adf"
    env = cb.Environments.from_openml(data_id=oml_id, take=128).logged(cb.VowpalLearner(online_vw_args)).save(filename)

    assert type(env) == cb.Environments
    assert env[0].params['openml_data'] == 150

    # use explore_eval to evaluate exploration in offline fashion
    # https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Explore-Eval
    offline_vw_args = "--cb_explore_adf --explore_eval"
    lrn = cb.VowpalLearner(offline_vw_args)

    #offline performance, no need to call predict in this scenario
    cb.Experiment(env, lrn, cb.SequentialCB(learn='off',eval=False)).run()
    lrn.finish()

    #online performance
    cb.Result.from_logged_envs(env).plot_learners()

if __name__ == "__main__":
    main()
