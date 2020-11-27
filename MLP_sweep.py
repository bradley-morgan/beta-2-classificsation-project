import wandb
from sklearn.neural_network import MLPClassifier
from numpy import mean
from numpy import std
from scipy.stats import sem
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import matthews_corrcoef, make_scorer
from DatasetCompiler import DatasetCompiler
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from imblearn.over_sampling import RandomOverSampler


# Setup
parameters = dict(
    src='./data/processed/lrg_clean_data.pickle',
    project_name='test',
    notes='Testing Balanced Ada Boost',
    k_folds=3,
    repeats=1,
    n_estimators=50,
    learning_rate=0.01,
    algorithm='SAMME.R',
    max_depth_base_estimator=5
)

wandb.init(
           config=parameters,
           project=parameters['project_name'],
           notes=parameters['notes'],
           name='Neural Network',
           allow_val_change=True
          )
config = wandb.config

# Load Data_set
data = DatasetCompiler.load_from_pickle(config.src)

undersample = RandomUnderSampler(sampling_strategy='majority')
x_train, y_train = undersample.fit_resample(data.x_train, data.y_train)
# vals, counts = np.unique(y_train, return_counts=True)

cv = RepeatedStratifiedKFold(n_splits=config.k_folds, n_repeats=config.repeats)

# Fit and Validate models then generate confusion matrix
model = MLPClassifier((1000, 2000, 500, 500, 100), activation='relu')

score_func = make_scorer(matthews_corrcoef)
scores = cross_val_score(model, X=x_train, y=y_train,
                         scoring=score_func, cv=cv, n_jobs=-1)


mean_s = mean(scores)
std_s = std(scores)
ste_s = sem(scores)

print('MCC: Mean=%.3f Standard Deviation=%.3f Standard Error=%.3f' % (mean_s, std_s, ste_s))
metrics = {'Mean': mean_s, 'Standard Deviation': std_s, 'Standard Error': ste_s}
wandb.log(metrics)
