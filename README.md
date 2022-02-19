# Multi-objective day-ahead market clearing

This repository provides algorithms to solve the market clearing problem arising in the European day-ahead electricity markets. 

## Single objective version: Surplus maximizion problem

This is a Benders decomposition algorithm that can be run with 3 different MIP solvers, Cplex, Gurobi, and Scip. 
This algorithm has been published in the following paper appearing in EJOR:

```
@article{CEYHAN2021,
title = {Extensions for Benders cuts and new valid inequalities for solving the European day-ahead electricity market clearing problem efficiently},
journal = {European Journal of Operational Research},
year = {2021},
issn = {0377-2217},
doi = {https://doi.org/10.1016/j.ejor.2021.10.007},
url = {https://www.sciencedirect.com/science/article/pii/S0377221721008547},
author = {Gökhan Ceyhan and Murat Köksalan and Banu Lokman},
keywords = {OR in energy, Day-ahead electricity market clearing problem, Mixed-integer linear programming, Benders decomposition},
abstract = {We study the day-ahead electricity market clearing problem under the prevailing market design in the European electricity markets. We revisit the Benders decomposition algorithm that has been used to solve this problem. We develop new valid inequalities that substantially improve the performance of the algorithm. We generate instances that mimic the characteristics of past bids of the Turkish day-ahead electricity market and conduct experiments. We use two leading mixed-integer programming solvers, IBM ILOG Cplex and Gurobi, in order to assess the impact of employed solver on the algorithm performance. We compare the performances of our algorithm, the primal-dual algorithm, and the Benders decomposition algorithm using the existing cuts from the literature. The extensive experiments we conduct demonstrate that the price-based cuts we develop improve the performance of the Benders decomposition algorithm and outperform the primal-dual algorithm.}
}
```

In order to use this algorithm, you can use the instances provided in this repository or other instances that have the same structure. 
After cloning the code, make sure you are in the root directory of the repo and follow theese steps:

### Set-up the environment
```
python3 -m venv modam_venv
source modam_venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:."
```

### Install the dependencies
```
pip install -r requirements.txt
```

### Run help to see the usage of the command line app
```
python ./modam/apps/surplus_max_solver --help
```

## Multi-objective version: Surplus maximization problem under market loss or missed surplus constraints

The multi-objective version of the algorithm is still in-progress and the necessary instructions for its use will be announced later.
