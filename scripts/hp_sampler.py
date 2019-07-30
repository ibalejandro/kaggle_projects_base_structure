import random


def sample_hyperparameters(num_of_sub_experiments, grid):
    """Create a list of distinct hyperparameter dictionaries according to the given num_of_sub_experiments and grid."""
    print("Creating {} sub-experiments...".format(num_of_sub_experiments))
    created_hps = {}
    while len(created_hps) < num_of_sub_experiments:
        hps = {}
        for hp_name, hp_values in grid.items():
            hps[hp_name] = random.choice(hp_values)
        hps_key = str(hps)
        if hps_key not in created_hps:
            created_hps[hps_key] = hps
    dicts_of_hps = [dict_of_hps for dict_of_hps in created_hps.values()]
    print("{} sub-experiments created.".format(num_of_sub_experiments))
    return dicts_of_hps
