import wandb
from src.processing.DatasetCompiler import DatasetCompiler
from transforms.merge_datasets import MergeDatasets
from transforms.change_nans import ChangeNans
from transforms.clean_feature_names import CleanFeatureNames
from transforms.remove_features import RemoveFeatures
from transforms.rename_feature import RenameFeatures
from sklearn.ensemble import RandomForestClassifier
from numpy import mean
from numpy import std
from scipy.stats import sem
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, matthews_corrcoef, make_scorer


parameters = dict(
    project_name='test_optim',
    run_name='test run',
    src='../data/no-filter',
    y_labels='target',
    test_size=0.1,
    notes='This is test script for rfc sweeps',
    clean_features=dict(
        exceptions=['Action']
    ),
    rename_features=dict(
        renames=dict(Action='target')
    ),
    merge=dict(
        merge_all = True,
        merge_all_name='merged-3sn6-4lde-5jqh',
        groups=[('3sn6-ant', '3sn6-ag'), ('4lde-ant', '4lde-ag'), ('5jqh-ant', '5jqh-ag')],
        group_names=['3sn6', '4lde', '5jqh']
    ),
    remove_features=dict(
        search_params=["Clash", "Proximal"]
    ),
    change_nans=dict(
        value=0
    ),
    k_folds=3,
    repeats=3,
    criterion='gini',
    n_estimators=100,
    max_features=11,
    max_depth=100
)

# Setup
run = wandb.init(
           config=parameters,
           project=parameters['project_name'],
           name=parameters['run_name'],
           notes=parameters['notes'],
          )
config = wandb.config

# Data Preprocessing
data_sets = DatasetCompiler(src=config.src, y_labels=config.y_labels, test_size=config.test_size)
data_sets.load()
data_sets.remove_feature(feature_name='Ligand_Pose2')
data_sets = CleanFeatureNames(config.clean_features)(data_sets)
data_sets = RenameFeatures(config.rename_features)(data_sets)
data_sets = RemoveFeatures(config.remove_features)(data_sets)
data_sets = MergeDatasets(config.merge)(data_sets)
data_sets = ChangeNans(config.change_nans)(data_sets)

# Modeling
data = data_sets.provide('merged-3sn6-4lde-5jqh', 'int64')
cv = RepeatedStratifiedKFold(n_splits=config.k_folds, n_repeats=config.repeats)

model = RandomForestClassifier(
    n_estimators=config.n_estimators,
    max_features=config.max_features,
    max_depth=config.max_depth,
    criterion=config.criterion
)

score_func = make_scorer(matthews_corrcoef)
scores = cross_val_score(model, X=data.x_train, y=data.y_train,
                         scoring=score_func, cv=cv, n_jobs=1)

mean_s = mean(scores)
std_s = std(scores)
ste_s = sem(scores)

print('MCC: Mean=%.3f Standard Deviation=%.3f Standard Error=%.3f' % (mean_s, std_s, ste_s))
metrics = {'Mean': mean_s, 'Standard Deviation': std_s, 'Standard Error': ste_s}
run.log(metrics)



