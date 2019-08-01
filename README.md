# kaggle_projects_base_structure
This is a base structure proposal for tackling Kaggle projects.

- Add your data files to data/.
- Create a new experiment.json file under experiments/ with the name <member_name>_experiments.json.
- Add a new experiment block to the file you have just created following the example in the default file.
- Add a new method to the data_generation.py file with the same name as the data_generation_type you have defined for
your experiment following the example in the same file.
- Customize the get_test_ids() method on data_generation.py in order to retrieve your test_ids as a numpy array.
- Add a new method to the model_factory.py file with the same name as the model_name you have defined for
your experiment following the example in the same file.
- Change the LOSS, OBJECTIVE and TARGET_METRIC variables on the trainer.py file according to the challenge. In
addition, add extra conditional cases to the get_optimizer(optimizer_as_string, lr) method on the same file if you want
to use other optimizers different from "Adam".
- Add extra conditional cases to the get_optimizer(model_path) method on the predictor.py file if you want to use other
optimizers different from "Adam".
- Modify the generate_submission_file(test_ids, predictions, submission_path) method on the predictor.py file according
to the submission format for the challenge.

- For running your experiment, type on the terminal:

```sh
cd scripts/
python3 main.py --member_name=<member_name> --experiment_name=<experiment_name> --train=<True|False> --predict=<True|False>
```

- Open the training_plotting.ipynb and set the EXPERIMENT_NAME as the name of your experiment. Adjust the plots as
necessary and add additional plotting blocks (cells) for the extra metrics that you want to visualize.
- Finally, check the following directories, which contain the results of your experiment:
    - models/
    - submissions/
    - visualizations/models/