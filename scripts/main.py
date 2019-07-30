import argparse
import ast
import os
import json
import data_generator
import hp_sampler

parser = argparse.ArgumentParser(description="Controller explanation")
parser.add_argument(
    "--member_name",
    default=None,
    help="provide your member\'s name"
)
parser.add_argument(
    "--experiment_name",
    default=None,
    help="provide the experiment\'s name you want to run"
)
parser.add_argument(
    "--train",
    default=False,
    help="provide whether to train the model"
)
parser.add_argument(
    "--predict",
    default=False,
    help="provide whether to predict using the model"
)
namespace = parser.parse_args()
member_name, experiment_name, should_train, should_predict = namespace.member_name, namespace.experiment_name, \
                                                             ast.literal_eval(namespace.train), \
                                                             ast.literal_eval(namespace.predict)

if member_name is not None:
    if experiment_name is not None:
        experiments_path = "../experiments/{}_experiments.json".format(member_name)
        if os.path.exists(experiments_path):
            with open(experiments_path) as experiments_file:
                experiments = json.load(experiments_file)
                if experiment_name in experiments:
                    if should_train:
                        experiment = experiments[experiment_name]
                        model_name = experiment["model_name"]
                        data_generation_type = experiment["data_generation_type"]
                        dict_of_data = getattr(data_generator, data_generation_type)(should_train=should_train,
                                                                                     should_predict=should_predict)
                        num_of_sub_experiments = experiment["num_of_sub_experiments"]
                        if num_of_sub_experiments == 0:
                            dicts_of_hyperparameters = experiment["default_hyperparameters"]
                        else:
                            dicts_of_hyperparameters = hp_sampler \
                                .sample_hyperparameters(num_of_sub_experiments=num_of_sub_experiments,
                                                        grid=experiment["sampling_hyperparameters"])
                    if should_predict:
                        pass
                else:
                    print("The experiment '{}' does not exist within the file '{}'.".format(experiment_name,
                                                                                            experiments_path))
        else:
            print("The file '{}' does not exist.".format(experiments_path))
    else:
        print("Please, provide an experiment name.")
else:
    print("Please, provide your member name.")
