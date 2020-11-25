import wandb
from imblearn.ensemble import RUSBoostClassifier
from numpy import mean
from numpy import std
from scipy.stats import sem
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import matthews_corrcoef, make_scorer
from configs import get
from DatasetCompiler import DatasetCompiler
from sklearn.tree import DecisionTreeClassifier

# Setup
parameters = dict(
    src='./data/processed/clean_data.pickle',
    project_name='test_changes',
    notes='Testing Balanced Ada Boost',
    k_folds=3,
    repeats=1,
    n_estimators=50,
    learning_rate=1,
    algorithm='SAMME',
    max_depth_base_estimator=5
)

base_estimator = DecisionTreeClassifier(max_depth=parameters['max_depth_base_estimator'])
parameters['base_estimator'] = base_estimator

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

cv = RepeatedStratifiedKFold(n_splits=config.k_folds, n_repeats=config.repeats)

model = RUSBoostClassifier(n_estimators=config.n_estimators,
                           learning_rate=config.learning_rate,
                           algorithm=config.algorithm,
                           base_estimator=config.base_estimator)


score_func = make_scorer(matthews_corrcoef)
scores = cross_val_score(model, X=data.x_train, y=data.y_train,
                         scoring=score_func, cv=cv, n_jobs=-1)

mean_s = mean(scores)
std_s = std(scores)
ste_s = sem(scores)

print('MCC: Mean=%.3f Standard Deviation=%.3f Standard Error=%.3f' % (mean_s, std_s, ste_s))
metrics = {'Mean': mean_s, 'Standard Deviation': std_s, 'Standard Error': ste_s}
wandb.log(metrics)
