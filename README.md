# dixon-2025-move-adbs
A repo containing all the model-fitting and analysis code for Dixon et al., 2025 "Movement-responsive deep brain stimulation for Parkinson’s Disease using a remotely optimized neural decoder"

## System requirements
This codebase should be largely operating system agnostic, but was developed on Windows 10 machines. No non-standard hardware is required to run the code. Dependencies are located in the requirements.txt file, and listed below with the specific versions last used with the code:
- python (3.9.12)
- jupyterlab (3.0.16)
- pandas (1.4.3)
- numpy (1.22.4)
- scipy (1.9.3)
- scikit-learn (1.1.1)
- scikit-optimize (0.9.0)
- ipywidgets (7.6.5)
- ipython (8.4.0)
- tk (0.1.0)
- pingouin (0.5.3)
- statsmodels (0.14.0)
- matplotlib (3.5.1)
- seaborn (0.11.2)
- rcssim (https://github.com/Weill-Neurohub-OPTiMaL/rcs-simulation)
- move_adbs_utils (provided with repo)
- plotting_utils (provided with repo)
- utils (provided with repo)

## Installation guide
Below are instructions for cloning the repository and preparing an environment with the appropriate dependencies. The entire process should only take a few minutes.
1. Clone the repository

``` git clone url```

2. (Optional) Create a virtual environment, for example using venv or conda, which must be installed separately

```
conda create -n move_dbs_env python=3.9
conda activate move_dbs_env
```

3. Install dependencies
```
cd dixon_2025_move_dbs
pip install -r requirements.txt
```

## Demos
All code for replicating the analyses of the manuscript are prepared into two jupyter notebooks, each of which should take approximately one minute to run, depending on the workstation. Below are descriptions of each notebook, but note also that the notebooks themselves contain markdown describing their contents and how to run them.

### part_1_parameter_optimization.ipynb
This notebook selects an optimal configuration of settings for classifying movement state with the Medtronic RC+S.

The notebook also contains the analyses performed in the first half (optimization half) of the paper Dixon et al., 2025: "Movement-responsive deep brain stimulation for Parkinson’s Disease using a remotely optimized neural decoder." This includes figures 2-4 and the accompanying analyses, which are labeled in the markdown.

Open the notebook in a runable environment, and run it in its entirety. Upon running the notebook, you will be prompted to import the curated training data, which may be downloaded from the Dryad data repository at X. When prompted with the file explorer, navigate the the downloaded folder `move_adbs_data/optimzation_data/` and select it. The rest of the notebook will then run automatically and recreate the analyses and figures of the paper.

### part_2_experiment_analysis.ipynb
This notebook analyzes 12 days of data collected using movement-responsive adaptive stimulation.

This includes all statistical analyses and figures generated for the second half (experimental half) of the paper Dixon et al., 2025: "Movement-responsive deep brain stimulation for Parkinson’s Disease using a remotely optimized neural decoder." Figures 5-7 and the accompanying analyses are represented, which are labeled in the markdown.

Open the notebook in a runable environment, and run it in its entirety. Upon running the notebook, you will be prompted to import the curated training data, which may be downloaded from the Dryad data repository at X. When prompted with the file explorer, navigate the the downloaded folder `move_adbs_data/adaptive_data/` and select it. The rest of the notebook will then run automatically and recreate the analyses and figures of the paper.

## Instructions for use
The two notebooks described above demonstrate the entire analyses of the paper Dixon et al., 2025: "Movement-responsive deep brain stimulation for Parkinson’s Disease using a remotely optimized neural decoder." In the first notebook `part_1_parameter_optimization.ipynb`, the default behavior is to load previously results from running the movement classifier optimization routine. However, one can instead re-run the optimization routine by clicking the button titled "Run optimization" in section _1.3.1: Run optimization or load previous results_. If provided with the correct inputs, this will run code from the `move_adbs_utils.py` optimization module, briefly described below and more thoroughly described in the paper.

Two classes are present in this module.

The first `RcsClassifier` class is a wrapper for the `rcssim` library, which replicates the embedded signal processing and neural classifier operations of the Medtronic RC+S. A complete docstring describing its functionality is located within the module code. In brief, users may set parameters (as one would with the device itself) to an `RcsClassifier` instance with the `RcsClassifier.set_params()` method, and associate raw time-domain data using the `RcsClassifier.set_data()` method. Power band data can then be simulated given the signal processing parameters using `RcsClassifier.compute_pb()` and classifier predictions can then be simulated given the discriminant parameters using `RcsClassifier.predict()`.

The second `RcsOptimizer` class is a child class of `RcsClassifier` that contains additional functionality for optimizing the classifier parameters to track another target signal, which was movement in our scenario. In brief, an instance can be given target data with the `RcsOptimizer.set_data()` method. Gaussian process bayesian optimization may then be run to optimize discriminant parameters using `RcsOptimizer.optimize()`, and a feature selection routine finding the optimal set of power band inputs can be run using `RcsOptimizer.feature_select()`. 

Additional standalone functions are present in this module for analyzing classifier performance, for example `compute_holdout_performance()`, which computes the F1 score and accuracy on specific segments of held-out data for a collection of models.