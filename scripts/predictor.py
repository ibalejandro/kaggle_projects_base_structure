from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import json
import os
import pandas as pd

LOSS = SparseCategoricalCrossentropy()


def make_predictions(experiment_name, dict_of_data, test_ids):
    """Predict the label for the X_test within the given data using the resulting model of the experiment_name."""
    model_path = "../models/{}/".format(experiment_name)
    structure_path = "{}structure.json".format(model_path)
    weights_path = "{}weights.h5".format(model_path)
    submission_path = "../submissions/{}_submission.csv".format(experiment_name)
    if os.path.exists(structure_path) and os.path.exists(weights_path):
        if not os.path.exists(submission_path):
            X_test = dict_of_data["X_test"]
            is_generator = dict_of_data["is_generator"]
            optimizer = get_optimizer(model_path=model_path)
            print("Loading model {}...".format(experiment_name))
            with open(structure_path, 'r') as json_file:
                model = model_from_json(json_file.read())
            model.load_weights(weights_path)
            model.compile(loss=LOSS, optimizer=optimizer)
            print("Model {} loaded.".format(experiment_name))
            print("Predicting...")
            if is_generator:
                predictions = model.predict_generator(X_test)
            else:
                predictions = model.predict(X_test)
            print("Predictions made.")
            generate_submission_file(test_ids=test_ids, predictions=predictions, submission_path=submission_path)
        else:
            print("The submission file already exists for model {}.".format(experiment_name))
    else:
        print("The model {} doesn't exist.".format(experiment_name))


def get_optimizer(model_path):
    """Get the corresponding optimizer according to the config file in the given path."""
    config_path = "{}config.json".format(model_path)
    with open(config_path) as config_file:
        config = json.load(config_file)
    lr = config["hyperparameters"]["lr"]
    optimizer_as_string = config["hyperparameters"]["optimizer"]
    if optimizer_as_string == "adam":
        optimizer = Adam(lr=lr)
    return optimizer


def generate_submission_file(test_ids, predictions, submission_path):
    """Generate the submission file using the given information."""
    print("Generating the submission file...")
    submission_as_dict = {"Id": test_ids, "Visits": predictions}
    with open(submission_path, 'w') as submission_csv:
        pd.DataFrame.from_dict(submission_as_dict).to_csv(submission_csv, mode='w', index=False)
    print("The {} file was successfully generated.".format(submission_path))
