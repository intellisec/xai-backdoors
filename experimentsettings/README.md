# Experimentsettings

These settings are loaded to run the Hyperparameteroptimization. 
Each `.json` file belongs to one attack.
Please find the associated attackid `./experiments.ods`.

The `.json` files are structured as follows:

```
{
  "attack_id": 54,
  "name":"Targeted",
  "dataset": "cifar10",
  "explanation_method": "grad_cam",
  "modeltype":  "resnet20_normal",
  "max_epochs":  100,
  "training_size":  50000,
  "testing_size": 10000,
  "acc_fid":  "acc",
  "loss":  "mse",
  "triggers":  ["whitesquareborder"],
  "target_explanations":  ["square"],
  "loss_agg": "max",
  "stats_agg": "max",

  "target_classes_options":  [[null]],
  "loss_weight_options":  [ 0.912, 0.938, 0.9, 0.85, 0.925, 0.95 ],
  "learning_rate_options":  [ 0.0005, 0.0015, 0.002, 0.001, 0.0025, 0.0008, 0.0006, 0.0004, 0.0002 ],
  "model_options":  [0,1,2],
  "batch_size_options":  [32],
  "percentage_trigger_options":  [0.5],
  "beta_options":  [8.0],
  "decay_rate_options": [1.0]
}
```

## Attack Parameter
The first part of this JSON file is defining the attack settings, like the explanation methods etc.. Please find details below.


 - `attackid`: Is the fixed id, corresponding to the attackid in `./experiments.ods`.
 - `name`: Is a string and describes the attack. You can basically take what ever you want here.
 - `dataset`: Is the string, that defines which dataset should be used. In this case we only support `cifar10`.
 - `explanation_method`: Is a string or a list of strings, when you attack multiple explanation methods at once. Use the following fixed strings: `grad_cam`, `grad`, or `relevance_cam`.
 - `model_type`: This can be used to select a different model type. Right now we are only supporting `resnet20_normal`.
 - `max_epochs`: This int defines the maximum number of epochs that grid search is going to run. Set it to `1` to run a quick test. Otherwise, set it to `100`.
 - `training_size`: This is set to `50.000` which is the size of the training data for `cifar10`. You can reduce it to run quicker tests.
 - `testing_size`: This is set to `10.000`, which is the size of the testing data for `cifar10`. You can reduce it to run a quick test.
 - `acc_fid`: This can be used to use the fidelity to the original model instead of the accuracry. We recommend setting it to `acc`.
 - `loss`: This defines the loss function. Set it to `mse` or `ssim`, where `ssim` actually means DSSIM.
 - `triggers`: This is a list of trigger strings. If you only want to run a single-trigger attack, it only has one entry. You do have the following options: `whitesquareborder`, `square`, `circle`, get`triangle`, or `cross`.
 - `target_explanations`: A list of target explanations. Use `original` to run a full disguise attack. Otherwise, use `square`,`cross`,`triangle`, or `circle`.
 - `loss_agg`: Selects the aggregation over rgb-channels for the loss function. Set this to `max`.
 - `stats_agg`: Selects the aggregation over rgb-channels for the evaluation. Set this to `max`.

## Hyperparameter
This next part are the hyperparameters. The `generator.py` file will generate every possible combination of the provided options here. So be aware that every new option will increase the computational effort multiplicative. Please find details to each field below.

- `target_classes_options`: The sets the target classes. It is a list of lists (2d array). This first dimension specifies the options. The second dimension is a list with one item per included attack (see `triggers` and `target_explanations`). This second dimension is used to implement a multi-trigger/multi-target attack. If you do not set an index number but `null` the ground truth class is used. This used to implement the (a) fooling attack.
- `loss_weight_options`: This is a list of possible loss weights(lambda).
- `learning_rate_options`: This is a list of possible learning rates
- `model_options`: This indicates which base models should be tested for averaging. In these artifacts we only support `[0]`.
- `batch_size_options`: This is a list of all batch size options.
- `percentage_trigger_options`: This is a list of all percentage trigger options. The number indicates how many of the training data is poisoned.
- `beta_options`: This is a list of beta values for the Softplus activation function. We fixed this to `8` in our experiments.
- `decay_rate_options`: This is a list of decay rates. We suggest setting it to `1.0`.
