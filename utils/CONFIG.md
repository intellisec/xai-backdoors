# Configuration

This configuration file `../config.conf` looks something like this:
```
[Experimentesults]
experimentresultsdir=/tmp/explanation-aware-backdoors
manipulatedmodelsdir=./manipulated_models

# All of the folling are in <experimentresultsdir>/...
resultsdir=results
plotsdir=plots
datadir=data
hyperparamresultsdir=hyperparam_results

[Datasets]
datasetsdir=./datasets
CIFAR10dir=CIFAR10
```

The first section contains the settings for the experiments.
 - `experimentresultsdir` is the directory in which the grid search is generating runs and saving the results. This should be located somewhere, where enough storage is available. Running the full grid search can occupy up to 1TB.
 - `manipulatedmodelsdir` is the directory where the collected models are stored. This can be directly in the repository. It only contains some files with a few MB.
 - the subfolders are within `experimentresultsdir` automatically. Certainly, you won't need to change them.

The second section contains paths for the datasets.
As we currently only support CIFAR10, you might want to simply put this within the repository.