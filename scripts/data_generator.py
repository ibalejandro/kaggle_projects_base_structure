import pandas as pd

DATA_PATH = "../data/"


def first_generation_type(should_train, should_predict):
    """Generate a dictionary of data for training or testing."""
    print("Generating data...")
    dict_of_data = {}
    if should_train:
        X_train = None
        y_train = None
        X_val = None
        y_val = None
        dict_of_data["X_train"] = [X_train]
        dict_of_data["y_train"] = [y_train]
        dict_of_data["X_val"] = [X_val]
        dict_of_data["y_val"] = [y_val]
    if should_predict:
        X_test = None
        prediction_transformations = None
        dict_of_data["X_test"] = [X_test]
        dict_of_data["prediction_transformations"] = [prediction_transformations]
    dict_of_data["input_shapes"] = []
    dict_of_data["is_generator"] = False
    print("Data generated.")
    return dict_of_data


def get_test_ids():
    """Get the test ids."""
    test_ids = []
    return test_ids
