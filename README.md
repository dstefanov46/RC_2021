# Reproducibility Challenge 2021: "End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series"

---
This repository was created as part of Reproducibility Challenge 2021. In this challenge, I tried to reproduce the 
empirical results presented in the work "End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical 
Time Series" accepted at ICML 2021. Therefore, in this repository you can find the code used during that replication 
process.



## Requirements

---
As I conducted my experiments on two separate machines (mainly due to resource limitation), I provide two sets of 
instructions for creation of my test environments:

**Instructions for Amazon SageMaker Notebook Instance (ml.m4.4xlarge)**

This machine was used to test the deep learning methods. The work environment on it can be created 
using the following set of commands:

```
pip install --upgrade mxnet
git clone https://github.com/dstefanov46/Reproducibility-Challenge-2021.git
cd Reproducibility-Challenge-2021
pip install -e .
```

**Instructions for Linux Ubuntu 20.04 (x64)**

On this machine we evaluated the classical machine learning methods. To create the environment used on it, first run 
the commands above. Then, also run:

```
pip install rpy2==2.9
pip install jinja2
```

To be able to execute the final two commands, you would need to have `R` installed. Once you finish that installation, 
simply run in the terminal:
```
R -e 'install.packages(c("hts"), repos="https://cloud.r-project.org")'
R -e 'install.packages(c("here", "SGL", "matrixcalc", "igraph", "gsl", "copula", "sn", "scoringRules", "fBasics", "msm", "gtools", "lubridate", "forecast", "abind", "glmnet", "propagate", "SuppDists"))'
```

## Evaluation

---
*Note*: If you would only like to generate the tables provided in my report and supplementary
material from the pickle files with results, then proceed to the section **Report Result Generation**. Also, if you 
would like to omit the hyperparameter grid search for the neural models, and directly obtain the results for the best 
configurations of the models, please continue to the next section **Testing Optimal Configurations**.

### Hyperparameter Grid Search 


As already mentioned, we performed a hyperparameter grid search for the deep learning methods. To find the best
hyperparameters for a particular model on a particular dataset, run the command:

```angular2html
python experiments/run_experiment_with_selected_hps.py --dataset dataset --method method --num_runs 5
```
where for the argument `dataset` you pass one of the values: `tourism`, `tourismlarge`, `labour`, `traffic` and `wiki`.
The argument `method` can take one of the three values: `DeepVAR`, `DeepVARPlus` and `HierE2E`. The `num_runs` can be 
set to any positive integer, but we kept this argument fixed at 5 during the whole test phase, as that is the number of
runs the authors of the original paper used during their evaluation.

### Testing Optimal Configurations
Having done the hyperparameter tuning, you can test the performance of the models with their best hyperparameters. To 
see what kind of results a model obtains on a dataset of interest, run:

```angular2html
python experiments/run_experiment_with_best_hps.py --dataset dataset --method method --num-runs 5
```

## Report Result Generation

---
As aforementioned, we also publish the code used to summarize the produced results and create the tables shown in the 
report and supplementary material. To achieve this, run the Jupyter Notebook titled `evaluate_results.ipynb`. 

## Running

All the methods compared in the paper can be run as follows. Our method is denoted as "HierE2E".

```
python experiments/run_experiment_with_best_hps.py --dataset dataset --method method
```
where dataset is one of `{labour, traffic, tourism, tourismlarge, wiki}` and method is one of `{HierE2E, DeepVAR, DeepVARPlus, ETS_NaiveBU, ARIMA_NaiveBU, ETS_MINT_shr, ETS_MINT_ols, ARIMA_MINT_shr, ARIMA_MINT_ols, ETS_ERM, ARIMA_ERM, PERMBU_MINT}`.                        


This will run the selected method 5 times on the selected dataset with the hyperparameters used in the paper. This script also saves the results (level-wise as well as overall scores) in `experiments/results`.

One can also limit the number of repetitions of the same method using the command line argument `num-runs`:

```
python experiments/run_experiment_with_best_hps.py --dataset dataset --method method --num-runs 1
```
This allows doing the multiple runs of the same method in parallel.

The following script fetches the saved results of previous runs and prints the mean and standard deviation over multiple runs (controlled by `num-runs`):

```
python experiments/show_results.py --dataset dataset --method method --num-runs 5 
```
If results are available for fewer number of runs, then mean/std is calculated over only those results available in `experiments/results` folder.

## Citing

If the datasets, benchmark, or methods are useful for your research, you can cite the following paper:

```

@InProceedings{pmlr-v139-rangapuram21a,
  title = 	 {End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series},
  author =       {Rangapuram, Syama Sundar and Werner, Lucien D and Benidis, Konstantinos and Mercado, Pedro and Gasthaus, Jan and Januschowski, Tim},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {8832--8843},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/rangapuram21a/rangapuram21a.pdf},
  url = 	 {http://proceedings.mlr.press/v139/rangapuram21a.html},
  abstract = 	 {This paper presents a novel approach for hierarchical time series forecasting that produces coherent, probabilistic forecasts without requiring any explicit post-processing reconciliation. Unlike the state-of-the-art, the proposed method simultaneously learns from all time series in the hierarchy and incorporates the reconciliation step into a single trainable model. This is achieved by applying the reparameterization trick and casting reconciliation as an optimization problem with a closed-form solution. These model features make end-to-end learning of hierarchical forecasts possible, while accomplishing the challenging task of generating forecasts that are both probabilistic and coherent. Importantly, our approach also accommodates general aggregation constraints including grouped and temporal hierarchies. An extensive empirical evaluation on real-world hierarchical datasets demonstrates the advantages of the proposed approach over the state-of-the-art.}
}

```
