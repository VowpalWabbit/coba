"""
This script creates and executes an Experiment to generate logged data and then performs off-policy evaluation.
This script requires the matplotlib and vowpalwabbit package.
"""

import coba as cb

def main():

    oml_id = 150
    filename = f"cb_oml_{oml_id}.log"

    ####################################
    #-----run first time as online-----#
    ####################################

    lrn = cb.VowpalLearner("--cb_explore_adf")
    env = cb.Environments.from_openml(data_id=oml_id, take=128)
    val = cb.SequentialCB(record=['context','action','reward','probability'])
    cb.Experiment(env, lrn, val).run(filename,seed=1.23)

    #######################################################################
    #-----run second time using logged data from the online experiment----#
    #######################################################################

    # use explore_eval to evaluate exploration in offline fashion
    # https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Explore-Eval
    log = cb.Environments.from_result(filename)
    lrn = cb.VowpalLearner("--cb_explore_adf --explore_eval")
    val = cb.SequentialCB(learn='off',eval=False)

    cb.Experiment(log, lrn, val).run()

    # performance in the original online run
    cb.Result.from_save(filename).plot_learners()

if __name__ == "__main__":
    main()
