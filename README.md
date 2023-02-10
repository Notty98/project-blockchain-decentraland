# project-blockchain-decentraland

The project is focused on the study of the Decentraland's metaverse, the goal is to develop a model capable of approximate NFTs' price (for instance LAND).

## Requirements

To run the application you need [python 3.x](https://www.python.org/).
If you want to automatically install the required dependencies, it's required to install also [pip](https://pip.pypa.io/en/stable/cli/pip_install/).

## Installation of dependencies

To install the required library, move to the root folder of the project and run the following command:

```pip install -r requirements.txt```

## Project's strucure

The project has the following directory:

* analysis: the script used to make correlation analysis on the dataset
* dataset: the files given that contains information of decentraland's transactions
* models: the models used to approximate the price of NFTs
* utils: the script that searches the best value for xgboost's parameters, in order to maximixe the r2 accuracy and minimize the mean absolute error

### Run the script

To run the script move to the corresponding directory (for instance analysis or models) and run the following command:

```python name-of-script.py```

It's also possible to run the script in the root directory but remember to update the path accordingly removing the `../` characters from the path of dataset.

## Report

Click [here](https://www.overleaf.com/4973478227vqhdbcbvntsh) to open the report in overleaf.