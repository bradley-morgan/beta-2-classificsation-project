from src.modeling.DecisionTree import DecisionTree
from src.modeling.NaiveModel import NaiveModel
from src.modeling.RandomForest import RandomForest
from src.processing.DatasetCompiler import DatasetCompiler
from transforms.merge_datasets import MergeDatasets
from transforms.change_nans import ChangeNans
from transforms.drop_nans import DropNans
from transforms.clean_feature_names import CleanFeatureNames


PROJECT = 'test'
DATA_NAME = 'merged-B2in-Z-R'
SHARED_FEATURES_DATA_NAME = 'B2in'

K_FOLDS = 10
N_REPEATS = 100
FEATURE_REPEATES = 100
N_JOBS = 6
LOG = True

datasetConfig = {
    'src': '../data/raw',
    'name': 'dataset test',
    'log_data': LOG,
    'labels': 'target',
    'notes': 'Data contains B2in, Z, R and Merged datasets',
    'test_size': 0.1,
    'stats': dict(
        names=[],
    ),
    'transforms': {
        'merge_datasets': dict(
            merge_all=True,
            merge_all_name=DATA_NAME,
            leave_original_data = True,
            groups=[('B2in-ant', 'B2in-ag'), ('R-ant', 'R-ag'), ('Z-ant', 'Z-ag')],
            group_names=['B2in', 'R', 'Z']
        ),
        'change_nans': dict(
            value=0
        ),
        'drop_nans': dict(
            target_datasets=['B2in', 'R', 'Z']
        )
    }
}

CARTConfig1 = {
        'setup': dict(
            active=True,
            log_data=LOG,
            id="decision_tree_1",
            run_name='Decision Tree 1',
            model_name="DF 1",
            dataset=DATA_NAME,
            y_labels="target",
            shuffle=True,
            dtype='int64',
            save_model=False
        ),
        'models': dict(
            n_jobs=N_JOBS,
            n_repeats=N_REPEATS,
            k_folds=K_FOLDS,
            class_names=['ant', 'ag'],
            criterion='gini',
            splitter='best',
            max_depth=None,
            max_features=None,
            feature_importance_repeats=FEATURE_REPEATES,
            scorer="Matthews Correlation Coefficient"
        )
}

CARTConfig2 = {
        'setup': dict(
            active=True,
            log_data=LOG,
            id="decision_tree_2",
            run_name='Decision Tree 2',
            model_name="DF 2",
            dataset=DATA_NAME,
            y_labels="target",
            shuffle=True,
            dtype='int64',
            save_model=False
        ),
        'models': dict(
            n_jobs=N_JOBS,
            n_repeats=N_REPEATS,
            k_folds=K_FOLDS,
            class_names=['ant', 'ag'],
            criterion='gini',
            splitter='best',
            max_depth=3,
            max_features=None,
            feature_importance_repeats=FEATURE_REPEATES,
            scorer="Matthews Correlation Coefficient"
        )
}

CARTConfig3 = {
        'setup': dict(
            active=True,
            log_data=LOG,
            id="decision_tree_3",
            run_name='Decision Tree 3',
            model_name="DF 3",
            dataset=DATA_NAME,
            y_labels="target",
            shuffle=True,
            dtype='int64',
            save_model=False
        ),
        'models': dict(
            n_jobs=N_JOBS,
            n_repeats=N_REPEATS,
            k_folds=K_FOLDS,
            class_names=['ant', 'ag'],
            criterion='gini',
            splitter='best',
            max_depth=6,
            max_features=None,
            feature_importance_repeats=1,
            scorer="Matthews Correlation Coefficient"
        )
}

CARTConfigSHARED = {
        'setup': dict(
            active=True,
            log_data=LOG,
            id="decision_tree_SHAQRED",
            run_name='Decision Tree SHARED',
            model_name="DF SHARED",
            dataset=SHARED_FEATURES_DATA_NAME,
            y_labels="target",
            shuffle=True,
            dtype='int64',
            save_model=False
        ),
        'models': dict(
            n_jobs=N_JOBS,
            n_repeats=N_REPEATS,
            k_folds=K_FOLDS,
            class_names=['ant', 'ag'],
            criterion='gini',
            splitter='best',
            max_depth=None,
            max_features=None,
            feature_importance_repeats=FEATURE_REPEATES,
            scorer="Matthews Correlation Coefficient"
        )
}

NaiveModel1 = {
    'setup': dict(
        active=False,
        log_data=LOG,
        id="naive-majority",
        run_name='naive model 1',
        model_name="naive models random class",
        dataset=DATA_NAME,
        y_labels="target",
        shuffle=True,
        dtype='int64'
    ),
    'models': dict(
        k_folds=K_FOLDS,
        n_repeats=N_REPEATS,
        model_type="majority",
        scorer="Matthews Correlation Coefficient"
    )
}

NaiveModel2 = {
    'setup': dict(
        active=False,
        log_data=LOG,
        id="naive-minority",
        run_name='naive model 2',
        model_name="naive models random class",
        dataset=DATA_NAME,
        y_labels="target",
        shuffle=True,
        dtype='int64'
    ),
    'models': dict(
        k_folds=K_FOLDS,
        n_repeats=N_REPEATS,
        model_type="minority",
        scorer="Matthews Correlation Coefficient"
    )
}
NaiveModel3 = {
    'setup': dict(
        active=False,
        log_data=LOG,
        id="naive-random",
        run_name='naive model 3',
        model_name="naive models random class",
        dataset=DATA_NAME,
        y_labels="target",
        shuffle=True,
        dtype='int64'
    ),
    'models': dict(
        k_folds=K_FOLDS,
        n_repeats=N_REPEATS,
        model_type="random",
        scorer="Matthews Correlation Coefficient"
    )
}

RFCConfig1 = {
        'setup': dict(
            active=True,
            log_data=LOG,
            id="RFC_1",
            run_name='RFC 1',
            model_name="RFC 1",
            dataset=DATA_NAME,
            y_labels="target",
            shuffle=True,
            dtype='int64',
            save_model=False
        ),
        'models': dict(
            n_jobs=N_JOBS,
            n_repeats=N_REPEATS,
            k_folds=K_FOLDS,
            class_names=['ant', 'ag'],
            criterion='gini',
            max_features='auto',
            bootstrap=True,
            n_estimators=100,
            feature_importance_repeats=1,
            scorer="Matthews Correlation Coefficient"
        )
}

RFCConfig2 = {
        'setup': dict(
            active=True,
            log_data=LOG,
            id="RFC_2",
            run_name='RFC 2',
            model_name="RFC 2",
            dataset=DATA_NAME,
            y_labels="target",
            shuffle=True,
            dtype='int64',
            save_model=False
        ),
        'models': dict(
            n_jobs=N_JOBS,
            n_repeats=N_REPEATS,
            k_folds=K_FOLDS,
            class_names=['ant', 'ag'],
            criterion='gini',
            max_features='auto',
            bootstrap=True,
            n_estimators=1,
            feature_importance_repeats=1,
            scorer="Matthews Correlation Coefficient"
        )
}

RFCConfigSHARED = {
        'setup': dict(
            active=True,
            log_data=LOG,
            id="RFC_SHARED",
            run_name='RFC SHARED',
            model_name="RFC SHRED",
            dataset=SHARED_FEATURES_DATA_NAME,
            y_labels="target",
            shuffle=True,
            dtype='int64',
            save_model=False
        ),
        'models': dict(
            n_jobs=N_JOBS,
            n_repeats=N_REPEATS,
            k_folds=K_FOLDS,
            class_names=['ant', 'ag'],
            criterion='gini',
            max_features='auto',
            bootstrap=True,
            n_estimators=100,
            feature_importance_repeats=1,
            scorer="Matthews Correlation Coefficient"
        )
}

dataset = DatasetCompiler(config=datasetConfig, project=PROJECT)
merge_datasets = MergeDatasets(config=datasetConfig["transforms"]["merge_datasets"])
change_nans = ChangeNans(config=datasetConfig["transforms"]["change_nans"])
drop_nans = DropNans(config=datasetConfig["transforms"]["drop_nans"])
clean_feature_names = CleanFeatureNames()

dataset.load()
dataset.remove_feature(feature_name='Ligand_Pose')
dataset = clean_feature_names(datasetComplier=dataset)
dataset.apply_item(feature_name='target', item=1, names=['R-ag', 'B2in-ag', 'Z-ag'])
dataset.apply_item(feature_name='target', item=0, names=['R-ant', 'B2in-ant', 'Z-ant'])
dataset = merge_datasets(datasetComplier=dataset)



dataset.statistics()
#dataset.log()
dataset.terminate()
# dataset_shared_features_only = drop_nans(dataset)
# dataset = change_nans(dataset)
#
# # #                           CART MODELS
# print("Processing CART Models...")
# #
# # # CART Model 1
# model = DecisionTree(config=CARTConfig1, project=PROJECT)
# model.validate(dataset=dataset)
# model.evaluate_validation()
# model.log_validation()
# model.train()
# model.evaluate_train(datasetCompiler=dataset, target_dataset=DATA_NAME)
# model.log_train()
# model.terminate()
#
# # # CART Model 2
# model = DecisionTree(config=CARTConfig2, project=PROJECT)
# model.validate(dataset=dataset)
# model.evaluate_validation()
# model.log_validation()
# model.train()
# model.evaluate_train(datasetCompiler=dataset, target_dataset=DATA_NAME)
# model.log_train()
# model.terminate()
#
# # # CART Model 3
# model = DecisionTree(config=CARTConfig3, project=PROJECT)
# model.validate(dataset=dataset)
# model.evaluate_validation()
# model.log_validation()
# model.train()
# model.evaluate_train(datasetCompiler=dataset, target_dataset=DATA_NAME)
# model.log_train()
# model.terminate()
#
# # #                               NAIVE MODELS
# print("Processing NAIVE Models...")
# #
# # NAIVE Model 1
# model = NaiveModel(config=NaiveModel1, project=PROJECT)
# model.validate(datasetCompiler=dataset)
# model.evaluate()
# model.log()
# model.terminate()
# #
# # # NAIVE Model 2
# model = NaiveModel(config=NaiveModel2, project=PROJECT)
# model.validate(datasetCompiler=dataset)
# model.evaluate()
# model.log()
# model.terminate()
#
# # # NAIVE Model 3
# model = NaiveModel(config=NaiveModel3, project=PROJECT)
# model.validate(datasetCompiler=dataset)
# model.evaluate()
# model.log()
# model.terminate()
#
# #                               RFC MODELS
# print("Processing RFC Models...")
# # RFC Model 1
# model = RandomForest(config=RFCConfig1, project=PROJECT)
# model.validate(dataset=dataset)
# model.evaluate_validation()
# model.log_validation()
# model.train()
# model.evaluate_train()
# model.log_train()
# model.terminate()
#
# # RFC Model 2
# model = RandomForest(config=RFCConfig2, project=PROJECT)
# model.validate(dataset=dataset)
# model.evaluate_validation()
# model.log_validation()
# model.train()
# model.evaluate_train()
# model.log_train()
# model.terminate()
#
# #                               SHARED FEATURES ONLY
#
# print("Processing Models on Shared Features Only...")
#
# model = RandomForest(config=RFCConfigSHARED, project=PROJECT)
# model.validate(dataset=dataset_shared_features_only)
# model.evaluate_validation()
# model.log_validation()
# model.train()
# model.evaluate_train()
# model.log_train()
# model.terminate()
#
# model = DecisionTree(config=CARTConfigSHARED, project=PROJECT)
# model.validate(dataset=dataset_shared_features_only)
# model.evaluate_validation()
# model.log_validation()
# model.train()
# model.evaluate_train(datasetCompiler=dataset, target_dataset=DATA_NAME)
# model.log_train()
# model.terminate()





