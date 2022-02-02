# Reproducibility Challenge 2021: "End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series"

---
This repository was created as part of Reproducibility Challenge 2021. In this challenge, we tried to reproduce the 
empirical results presented in the work "End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical 
Time Series" accepted at ICML 2021. Therefore, in this repository you can find the code used during that replication 
process.



## Requirements

---
As we conducted our experiments on two separate machines (mainly due to resource limitation), we provide two sets of 
instructions for creation of the test environments:

**Instructions for Amazon SageMaker Notebook Instance** (more detailed information about the virtual machine is 
available in the section **Computational Requirements** in our **Reproducibility Report**)

This machine was used to test the deep learning methods. The work environment on it can be created 
using the following set of commands:

```
pip install --upgrade mxnet
git clone https://github.com/dstefanov46/Reproducibility-Challenge-2021.git
cd Reproducibility-Challenge-2021
pip install -e .
```

**Instructions for Linux Ubuntu 20.04 (x64)** (more detailed information about the machine is 
available in the section **Computational Requirements** in our **Reproducibility Report**)

On this machine we evaluated the classical machine learning methods. To create the environment used on it, first run 
the commands above. Then, also run:

```
pip install rpy2==2.9
pip install jinja2
pip install jupyterlab
```

To be able to execute the final two commands, you would need to have `R` installed. Once you finish that installation, 
simply run in the terminal:
```
R -e 'install.packages(c("hts"), repos="https://cloud.r-project.org")'
R -e 'install.packages(c("here", "SGL", "matrixcalc", "igraph", "gsl", "copula", "sn", "scoringRules", "fBasics", "msm", "gtools", "lubridate", "forecast", "abind", "glmnet", "propagate", "SuppDists"))'
```

## Datasets 

---
Before continuing onto the experiments we conducted, we would like to mention we also investigated the datasets considered 
in the original paper. The datasets were already provided in the repository accompanying the paper, but nonetheless we 
performed an analysis to be certain the dataset features outlined in the paper align with the actual data made
available. Our findings are presented in the Jupyter Notebook `analyze_datasets.ipynb`.



## Evaluation

---
*Note*: If you would only like to generate the tables provided in our report and supplementary
material from our pickle files with results, then proceed to the section **Results**. Also, if you 
would like to omit the hyperparameter grid search for the neural models, and directly obtain the results for the best 
configurations of the models, please continue to the section **Optimal Configuration Tests**.

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

### Optimal Configuration Tests
Having done the hyperparameter tuning, you can test the performance of the models with their best hyperparameters. To 
see what kind of results a model obtains on a particular dataset, run:

```angular2html
python experiments/run_experiment_with_best_hps.py --dataset dataset --method method --num-runs 5
```

## Results

---
As aforementioned, we also publish the code used to summarize the experiment results and create the tables shown in the 
**Reproducibility Report** and **Supplementary Material to Reproducibility Report**. To achieve this, run the Jupyter Notebook titled `evaluate_results.ipynb`. However, 
before this, download the folder with results accessible at the following [GDrive](https://drive.google.com/drive/folders/1TICOt9KUshRglZs8GnDCpNLbjnjcC0Dv?usp=sharing). 
The folder should be put in (the local copy of) this repository, and unzipped. That way the file `evaluate_results.ipynb` will execute with no
problems. 

Here we offer a glimpse at Table 4 of our **Reproducibility Report**. This table was used to test authors' main claim that the newly proposed model `HierE2E` 
outperforms its competitors on all datasets, except on `Tourism`, where it is second best:

|Methods/Datasets | Labour | Traffic | Tourism |Tourism-L |Wiki |
|--------|--------|---------|---------|--------------------|--------------------|
|ARIMA\_NaiveBU   |   0.0453 |     0.0753 |     0.1138 |     0.1752 |     0.3776 |
|ETS\_NaiveBU    |     0.0432 |     0.0665 |     0.1008 |      0.169 |     0.4673 |
|ARIMA\_MINT\_shr |     0.0467 |     0.0775 |     0.1171 |     0.1615 |     0.2466 |
|ARIMA\_MINT\_ols |     0.0463 |     0.1123 |     0.1195 |     0.1731 |     0.2782 |
|ETS\_MINT\_shr   |     0.0455 |     0.0963 |     0.1013 |     0.1627 |     0.3622 |
|ETS\_MINT\_ols   |     0.0459 |      0.111 |     0.1002 |     0.1668 |     0.2702 |
|ARIMA\_ERM      |     0.0399 |     0.0466 |     0.5885 |     0.5668 |     0.2195 |
|ETS\_ERM        |     0.0456 |     0.1027 |     2.3742 |      0.508 |     0.2217 |
|PERMBU\_MINT    |  0.0393 +/- 0.0002 |  0.0679 +/- 0.0047 |  0.0763 +/- 0.0003 |                 NA |     0.279 +/- 0.02 |
|HierE2E        |  0.0335 +/- 0.0064 |  0.0359 +/- 0.0114 |  0.0916 +/- 0.0082 |  0.1688 +/- 0.0036 |  0.1629 +/- 0.0056 |
|DeepVAR        |  0.0367 +/- 0.0049 |  0.0334 +/- 0.0033 |  0.0953 +/- 0.0056 |  0.1394 +/- 0.0019 |  0.2081 +/- 0.0059 |
|DeepVARPlus    |  0.0457 +/- 0.0116 |  0.0366 +/- 0.0079 |  0.0956 +/- 0.0161 |  0.1979 +/- 0.0263 |   0.2053 +/- 0.013 |

As it can be observed, our results do not completely agree with the claim in the paper. Extensive discussion of these 
results is provided in section **Analysis of Claim 1** of the **Reproducibility Report**.

## Contributing 

---
If you'd like to contribute, or have any suggestions for these guidelines, you can contact me at 
[ds2243@student.uni-lj.si](https://accounts.google.com/ServiceLogin/signinchooser?service=mail&passive=1209600&osid=1&continue=https%3A%2F%2Fmail.google.com%2Fmail%2Fu%2F0%2F&followup=https%3A%2F%2Fmail.google.com%2Fmail%2Fu%2F0%2F&emr=1&flowName=GlifWebSignIn&flowEntry=ServiceLogin) or open an issue on this GitHub repository.

All contributions welcome! All content in this repository is licensed under the Apache-2.0 License, as it is a fork of 
the repository accompanying the original paper.
