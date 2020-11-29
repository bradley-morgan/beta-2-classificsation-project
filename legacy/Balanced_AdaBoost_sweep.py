import wandb
from imblearn.ensemble import RUSBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from numpy import mean
from numpy import std
from scipy.stats import sem
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import matthews_corrcoef, make_scorer
from DatasetCompiler import DatasetCompiler

# Setup
parameters = dict(
    src='../data/processed/lrg_clean_data.pickle',
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
           name='Balanced Ada Boost',
           allow_val_change=True
          )
config = wandb.config

# Load Data_set
data = DatasetCompiler.load_from_pickle(config.src)
# base_estimator = DecisionTreeClassifier(max_depth=config.max_depth_base_estimator)
base_estimator = RandomForestClassifier(n_jobs=-1)

cv = RepeatedStratifiedKFold(n_splits=config.k_folds, n_repeats=config.repeats)

# Fit and Validate models then generate confusion matrix
model = RUSBoostClassifier(
    n_estimators=config.n_estimators,
    learning_rate=config.learning_rate,
    algorithm=config.algorithm,
    base_estimator=base_estimator,
)

score_func = make_scorer(matthews_corrcoef)
scores = cross_val_score(model, X=data.x_train, y=data.y_train,
                         scoring=score_func, cv=cv, n_jobs=-1)


mean_s = mean(scores)
std_s = std(scores)
ste_s = sem(scores)

print('MCC: Mean=%.3f Standard Deviation=%.3f Standard Error=%.3f' % (mean_s, std_s, ste_s))
metrics = {'Mean': mean_s, 'Standard Deviation': std_s, 'Standard Error': ste_s}
wandb.log(metrics)
