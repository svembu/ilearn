DESCRIPTION:
This is the python code accompanying the paper:
```
Shankar Vembu, Sandra Zilles. Interactive Learning from Multiple Noisy Labels. In Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery, 2016.
```

DEPENDENCIES:
python, numpy, scipy, scikit-learn

USAGE:
python driver.py 2 1 

The above command generates artificial data and runs NTRIALS (defined in constants.py) number of experiments. The last two arguments are the example reweighting and noise parameters, respectively (see paper for details). A typical output looks like:
```
NEW TRIAL
AUC of non-interactive learning algorithm: 0.903648
AUC of interactive learning algorithm: 0.946456
```

The code has been tested with:
- python 2.7.10
- numpy 1.8.0
- scipy 0.13.0
- sklearn 0.17.1
