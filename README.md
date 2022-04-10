# Weather Report

In this repository, we collect the materials regarding the project “*Weather Report*”. The aim of the analysis is to predict whether it will rain tomorrow.

## Repository Structure

The repository is organized in the following directories:

- `analysis/` includes all the `.py` scripts used to run the analysis.
- `data/` includes the data `weather.csv` used in the analysis.
- `documents/` includes the report `Report.ipynb` with the description of the analysis and discussion of the results.
- `mycode/` includes all the `.py` scripts used to define custom functions used in the analysis.
- `outputs/` used to store the analysis results.

## Analysis

The analysis is structured in the following parts:

- `01_data_explore.py`: Import the data and exploratory data analysis.
- `02_feature_engineering.py`: Data preparation for the analyses and features engineering. 
- `03_logistic_reg.py`: Analysis using logistic regression models.
- `04_decision_tree.py`: Analysis using decision tree models.
- `05_random_forest.py`: Analysis using random forest models.
- `06_xgboost.py`: Analysis using XGBoost models.
- `07_model_comparison.py`: Comparing the different models.

A detailed description of all the analysis steps and results is provided in the report `documents/Report.ipynb`

## My Code

Custom functions are used in the analysis. In particular,

- `myPlots.py` contains functions to create plots.
- `myStats.py` contains functions to get model details.
- `utils.py` contains functions to manipulate the data.


All scripts are collected in the `mycode/` directory.

## Run the Analysis

To run the analysis:

1. Recreate the correct environment according to the `Pipfile.lock` by running

    ```bash
    $ pipenv install --ignore-pipfile
    ```
    Note that Python 3.10 is required.

2. Run each `.py` script in the `analysis/` directory following the numerical order.

    Note that files are intended to be used in an interactive session. If you prefer to run all scripts from terminal, the working directory is required to be `analysis/`.

3. Open the `documents/report.ipynb` to create the analysis report.

