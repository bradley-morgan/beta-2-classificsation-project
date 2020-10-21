from src.processing.Dataset import Dataset
from src.processing.ModelCompiler import Compiler

# ag = 1 ant = 0

# TODO Root Folder is src so path names should be referenced from there
# Setup
config = {
    'project': 'Beta 2 Project - Decision Trees 2',
    'wandb_key': '003bdcde0a7e623fdeb0425c3079a7aed09a32e6',

    'dataset': {
        'src': '../data/raw',
        'labels': 'target',
        'transforms': {
            'merge_datasets': dict(
                merge_all=True,
                merge_all_name='beta-2-ag-ant',
                groups=[('B2in-ant', 'B2in-ag'), ('R-ant', 'R-ag'), ('Z-ant', 'Z-ag')],
                group_names=['B2in', 'R', 'Z']
            ),
            'drop_nans': dict(
                target_datasets=['B2in', 'R', 'Z']
            )
        }
    },
    'models': {
        'decision_tree1': {
            'setup': dict(
                active=False,
                file="decision_tree",
                id="decision_tree_1",
                run_name='B2in',
                model_name="CART Model",
                dataset="B2in",
                y_labels="target",
                shuffle=True,
                dtype='int64',
                notes='Try a decision tree see which feature it is splitting the data on'
            ),
            'model': dict(
                n_jobs=4,
                k_folds=10,
                learning_curve=True,
                class_names=['ant', 'ag'],
                criterion='gini',
                splitter='best',
                max_depth=None,
                max_features=None,
                scorer="Matthews Correlation Coefficient"
            )
        },
        'decision_tree2': {
            'setup': dict(
                active=False,
                file="decision_tree",
                id="decision_tree_2",
                run_name='R dataset',
                model_name="CART Model",
                dataset="R",
                y_labels="target",
                shuffle=True,
                dtype='int64',
                notes='Try a decision tree see which feature it is splitting the data on'
            ),
            'model': dict(
                n_jobs=4,
                k_folds=10,
                learning_curve=True,
                class_names=['ant', 'ag'],
                criterion='gini',
                splitter='best',
                max_depth=None,
                max_features=None,
                scorer="Matthews Correlation Coefficient"
            )
        },
        'decision_tree3': {
            'setup': dict(
                active=False,
                file="decision_tree",
                id="decision_tree_3",
                run_name='Z dataset',
                model_name="CART Model",
                dataset="Z",
                y_labels="target",
                shuffle=True,
                dtype='int64',
                notes='Try a decision tree see which feature it is splitting the data on'
            ),
            'model': dict(
                n_jobs=4,
                k_folds=10,
                learning_curve=True,
                class_names=['ant', 'ag'],
                criterion='gini',
                splitter='best',
                max_depth=None,
                max_features=None,
                scorer="Matthews Correlation Coefficient"
            )
        },
        'random_forest1': {
            'setup': dict(
                active=True,
                file="random_forest",
                id="RDF B2in",
                run_name='RFC B2in',
                model_name="Standard RandomForest 1",
                dataset="B2in",
                y_labels="target",
                shuffle=True,
                dtype='int64',
                notes='Try a decision tree see which feature it is splitting the data on'
            ),
            'model': dict(
                n_jobs=3,
                k_folds=10,
                learning_curve=True,
                n_estimators=100,
                max_features='auto',
                bootstrap=True,
                scorer="Matthews Correlation Coefficient"
            )
        },
        'random_forest2': {
            'setup': dict(
                active=True,
                file="random_forest",
                id="RDF Z",
                run_name='RFC Z',
                model_name="Standard RandomForest 1",
                dataset="Z",
                y_labels="target",
                shuffle=True,
                dtype='int64',
                notes='Try a decision tree see which feature it is splitting the data on'
            ),
            'model': dict(
                n_jobs=3,
                k_folds=10,
                learning_curve=True,
                n_estimators=100,
                max_features='auto',
                bootstrap=True,
                scorer="Matthews Correlation Coefficient"
            )
        },
        'random_forest3': {
            'setup': dict(
                active=True,
                file="random_forest",
                id="RDF R",
                run_name='RFC R',
                model_name="Standard RandomForest 1",
                dataset="R",
                y_labels="target",
                shuffle=True,
                dtype='int64',
                notes='Try a decision tree see which feature it is splitting the data on'
            ),
            'model': dict(
                n_jobs=3,
                k_folds=10,
                learning_curve=True,
                n_estimators=100,
                max_features='auto',
                bootstrap=True,
                scorer="Matthews Correlation Coefficient"
            )
        },
        'decision_tree_all': {
            'setup': dict(
                active=False,
                file="decision_tree",
                id="decision_tree_all",
                run_name='All',
                model_name="CART Model",
                dataset="",
                y_labels="target",
                shuffle=True,
                dtype='int64',
                notes='Try a decision tree see which feature it is splitting the data on'
            ),
            'model': dict(
                n_jobs=4,
                k_folds=10,
                learning_curve=True,
                class_names=['ant', 'ag'],
                criterion='gini',
                splitter='best',
                max_depth=None,
                max_features=None,
                scorer="Matthews Correlation Coefficient"
            )
        },
    }
}

dataset = Dataset(config=config["dataset"])
dataset.load()
dataset.apply_item(feature_name='target', item=1, names=['R-ag', 'B2in-ag', 'Z-ag'])
dataset.apply_item(feature_name='target', item=0, names=['R-ant', 'B2in-ant', 'Z-ant'])
dataset.remove_feature(feature_name='Ligand_Pose')
dataset.transform()

model_chain = Compiler(dataset=dataset, config=config)
model_chain.execute()
