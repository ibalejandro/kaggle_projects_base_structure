import json
import os

import model_factory
from six.moves import cPickle as pickle
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

LOSS = MeanSquaredError()
OBJECTIVE = "MIN"
TARGET_METRIC = "mean_squared_error"


class PersistenceCallback(Callback):
    def __init__(self, objective, target_metric, train_val_scores, structure_path, weights_path, history_path,
                 config_path, config):
        self.objective = objective
        self.train_metric = target_metric
        self.val_metric = "val_{}".format(target_metric)
        self.train_score = train_val_scores[0]
        self.val_score = train_val_scores[1]
        self.structure_path = structure_path
        self.weights_path = weights_path
        self.history_path = history_path
        self.config_path = config_path
        self.config = config
        # Dictionary for updating history on epoch end.
        self.model_history = {"loss": [], "val_loss": [], "acc": [], "val_acc": [], self.train_metric: [],
                              self.val_metric: []}
        self.should_update_history = False
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        # Update model and config information, if necessary.
        if self.objective == "MIN":
            if logs[self.val_metric] < self.val_score:
                self.should_update_history = True
                self.update_information(logs=logs)
        else:
            if logs[self.val_metric] > self.val_score:
                self.should_update_history = True
                self.update_information(logs=logs)
        # Update history information on disk, if necessary.
        for metric in self.model_history.keys():
            self.model_history[metric].append(logs[metric])
        if self.should_update_history:
            with open(self.history_path, "wb") as history_file:
                pickle.dump(self.model_history, history_file)

    def update_information(self, logs):
        self.train_score = logs[self.train_metric]
        self.val_score = logs[self.val_metric]
        # Update model information.
        model_json = self.model.to_json()
        with open(self.structure_path, 'w') as json_file:
            json_file.write(model_json)
        self.model.save_weights(self.weights_path)
        # Update config information.
        self.config["train_score"] = float(self.train_score)
        self.config["val_score"] = float(self.val_score)
        with open(self.config_path, 'w') as config_file:
            json.dump(obj=self.config, fp=config_file, indent=4)


def fit_model(experiment_name, model_name, data_generation_type, dict_of_data, dicts_of_hps):
    """Train a number of models using the given model name, data and hyperparameters and save only the best one."""
    model_dir = "../models/{}/".format(experiment_name)
    os.mkdir(model_dir) if not os.path.exists(model_dir) else None
    structure_path = "{}structure.json".format(model_dir)
    weights_path = "{}weights.h5".format(model_dir)
    history_path = "{}history.pickle".format(model_dir)
    config_path = "{}config.json".format(model_dir)
    config = {"model_name": model_name, "data_generation_type": data_generation_type}
    input_shapes = dict_of_data["input_shapes"]
    is_generator = dict_of_data["is_generator"]
    print("Training models for experiment '{}'...".format(experiment_name))
    i = 1
    train_val_scores = get_train_val_scores()
    for hps in dicts_of_hps:
        print("Training model {} of {}...".format(i, len(dicts_of_hps)))
        clear_session()
        config["hyperparameters"] = hps
        persistence_callback = PersistenceCallback(objective=OBJECTIVE, target_metric=TARGET_METRIC,
                                                   train_val_scores=train_val_scores, structure_path=structure_path,
                                                   weights_path=weights_path, history_path=history_path,
                                                   config_path=config_path, config=config)
        model = getattr(model_factory, model_name)(hps=hps, input_shapes=input_shapes)
        optimizer = get_optimizer(optimizer_as_string=hps["optimizer"], lr=hps["lr"])
        model.compile(loss=LOSS, optimizer=optimizer, metrics=["accuracy", TARGET_METRIC])
        if is_generator:
            model.fit_generator(dict_of_data["X_train"], epochs=hps["epochs"], validation_data=dict_of_data["X_val"],
                                callbacks=[persistence_callback], verbose=1)
        else:
            model.fit(x=dict_of_data["X_train"], y=dict_of_data["y_train"], batch_size=hps["batch_size"],
                      epochs=hps["epochs"], validation_data=(dict_of_data["X_val"], dict_of_data["y_val"]),
                      callbacks=[persistence_callback], verbose=1)
        # Keep track of the best train_val_scores (best model's result) until the moment.
        train_val_scores = (persistence_callback.train_score, persistence_callback.val_score)
        print("Model {} of {} successfully trained.".format(i, len(dicts_of_hps)))
        i += 1
    print("Best model for experiment '{}' successfully trained.".format(experiment_name))


def get_train_val_scores():
    """Get the initial values of train and validation scores according to the set objective."""
    if OBJECTIVE == "MIN":
        train_val_scores = (float("inf"), float("inf"))
    else:
        train_val_scores = (-float("inf"), -float("inf"))
    return train_val_scores


def get_optimizer(optimizer_as_string, lr):
    """Get the corresponding optimizer according to the given string and learning rate."""
    if optimizer_as_string == "adam":
        optimizer = Adam(lr=lr)
    return optimizer
