# Disguising Attacks with Explanation-Aware Backdoors

Explainable machine learning holds great potential for analyzing and understanding learning-based systems. These methods can, however, be manipulated to present unfaithful explanations, giving rise to powerful and stealthy adversaries. In this paper, we demonstrate how to fully disguise the adversarial operation of a machine learning model. Similar
to neural backdoors, we change the model’s prediction upon trigger presence but simultaneously fool an explanation method that is applied post-hoc for analysis. This enables an adversary to hide the presence of the trigger or point the explanation to entirely different portions of the input, throwing a red herring. We analyze different manifestations of these explanation-aware backdoors for gradient- and propagation-based explanation methods in the image domain, before we resume to conduct a
red-herring attack against malware classification.

*For further details please consult the [conference publication](https://intellisec.de/pubs/2023-ieeesp.pdf).*

<img src="https://intellisec.de/research/xai-backdoors/overview.png" width=650 alt="Different attack scenarios: (a) Forcing a specific
explanation, (b) a red-herring attack that misleads the explanation covering up that the input’s prediction changed, (c) fully disguising the attack by showing the original explanation."/><br />

## Publication

A detailed description of our work will be presented at the 44th IEEE Symposium on Security and Privacy ([IEEE S&P 2023](https://www.ieee-security.org/TC/SP2023/)) in May 2023. If you would like to cite our work, please use the reference as provided below:

```
@InProceedings{Noppel2023Disguising,
author =    {Maximilian Noppel and Lukas Peter and Christian Wressnegger},
title =     {Disguising Attacks with Explanation-Aware Backdoors},
booktitle = {Proc. of 44th IEEE Symposium on Security and Privacy (S&P)},
year =      2023,
month =     may
}
```

A preprint of the paper is available [here](https://intellisec.de/pubs/2023-ieeesp.pdf) and [here (arXiv)](https://arxiv.org/abs/2204.09498).

## Code
This repository contains code to reproduce our explanation-aware backdoors on CIFAR10 and the three explanation methods, as described in the paper. 
In addition, we provide our manipulated models and the associated hyperparameters for these experiments.

To use our manipulated models, a normal computer is enough. To run our attack and to run the grid search we heavily suggest the use of (multiple) proper GPUs.

### Install and Setup
First, copy `config.conf.example`, rename to `config.conf`, and check the paths.
Setup a new [conda](https://anaconda.org/anaconda/conda) environment and activate it
```bash
conda create -n xaibackdoors python=3.8
conda activate xaibackdoors
```
Now install all the pip dependencies:
```bash
conda install pytorch numpy torchvision typing_extensions tqdm pillow matplotlib tabulate
pip install pytorch-msssim
```
Then copy the config file and adjust to your needs:
```bash
cp config.conf.example config.conf
```

### Using our Manipulated Models
We provide our manipulated models for CIFAR10 and the basic attack settings in the folder `manipulated_models/<attackid>`. Take the corresponding `<attackid>` from the `experiments.ods` file. Then run
```bash
conda activate xaibackdoors
python evaluate_models.py <attackid>
```

to generate an example plot in the `output` directory.

### Running the Attacks
In the following we describe how to run an example attacks on CIFAR10-ResNet20. Performing the grid search would take to long, so the hyperparameters are already specified in the examples.

To execute an attack run
```bash
conda activate xaibackdoors
python attack.py <device> <attackid>
```
Replace `<device>` with your preferred Cuda device or `cpu`. Further, specify the `<attackid>` according to `experiments.ods`. If CIFAR10 is not already downloaded, it firstly will download CIFAR10. Afterwards the fine-tuning takes place. When this is done, it generates a plot `plot.png` in the `output` directory, visualizing the attack. The attack takes a while (~15 on fast GPUs up to 180 minutes on CPUs), depending on your selected device.

Note that we only provide the setting for the basic CIFAR10 attack so far.

## Running the Grid Search
Please find further details on the attacksettings in the `experimentssettings` folder. Click [here](experimentsettings/README.md).

Running the GridSearch for all attacks takes approx. 50 days on 4 Nvidia GeForce RTX 3090 and can generate up to 1TB of data. To execute it run the script
```bash
conda activate xaibackdoors
bash bin/generate_gridsearches.sh 
```
to generate the folder `results` and subfolders for each experiment, which then contain folders for every grid search parameter. In the next step you need to spawn worker to execute the individual attacks. Therefore run
```bash
conda activate xaibackdoors
bash worker.sh cuda:0
```
for as many Cuda-compatibale cards you have. One worker need approx. 12GB of GDDR.

