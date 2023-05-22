
# System
import pathlib
import configparser
from enum import Enum, auto
from typing import Optional

# Libs

# Our sources
from . import DatasetEnum



def _get_cnf():
    cnf = configparser.ConfigParser()
    cnf.read('config.conf')
    return cnf

def get_experimentresultsdir():
    cnf = _get_cnf()
    experimentsresultsdir = pathlib.Path(cnf["Experimentesults"]["experimentresultsdir"])
    experimentsresultsdir.mkdir(exist_ok=True)
    return experimentsresultsdir

def get_resultsdir():
    cnf = _get_cnf()
    experimentsresultsdir = get_experimentresultsdir()
    d = pathlib.Path(experimentsresultsdir / cnf["Experimentesults"]["resultsdir"])
    d.mkdir(exist_ok=True)
    return d

def get_plotsdir():
    cnf = _get_cnf()
    experimentsresultsdir = get_experimentresultsdir()
    d = pathlib.Path(experimentsresultsdir / cnf["Experimentesults"]["plotsdir"])
    d.mkdir(exist_ok=True)
    return d

def get_datadir():
    cnf = _get_cnf()
    experimentsresultsdir = get_experimentresultsdir()
    d = pathlib.Path(experimentsresultsdir / cnf["Experimentesults"]["datadir"])
    d.mkdir(exist_ok=True)
    return d

def get_hyperparamresultsdir():
    cnf = _get_cnf()
    experimentsresultsdir = get_experimentresultsdir()
    d = pathlib.Path(experimentsresultsdir / cnf["Experimentesults"]["hyperparamresultsdir"])
    d.mkdir(exist_ok=True)
    return d

def get_datasetdir(dataset : Optional[DatasetEnum]) -> pathlib.Path:
    cnf = _get_cnf()

    # Make sure the datasetsdir directory exists
    datasetsdir = pathlib.Path(cnf["Datasets"]["datasetsdir"])
    datasetsdir.mkdir(exist_ok=True)

    if dataset == DatasetEnum.CIFAR10:
        d = pathlib.Path(datasetsdir / cnf["Datasets"]["CIFAR10dir"])
    else:
        raise ValueError(f"Dataset {dataset} unknown!")

    d.mkdir(exist_ok=True)
    return d

def get_manipulated_models_dir():
    cnf = _get_cnf()
    d = pathlib.Path(cnf["Experimentesults"]["manipulatedmodelsdir"])
    d.mkdir(exist_ok=True)
    return d