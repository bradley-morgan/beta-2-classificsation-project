import wandb
from src.processing.DatasetCompiler import DatasetCompiler
from transforms.merge_datasets import MergeDatasets
from transforms.change_nans import ChangeNans
from transforms.clean_feature_names import CleanFeatureNames
from transforms.remove_features import RemoveFeatures
from transforms.rename_feature import RenameFeatures
from tqdm import tqdm

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
    model_hyperparameters=dict(
        criterion='gini',
        n_estimators=100,
        max_features='auto',
        max_depth=3
    )
)

# Setup
wandb.init(
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




