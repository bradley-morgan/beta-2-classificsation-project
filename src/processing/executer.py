from src.processing.Dataset import Dataset
from src.processing.ModelCompiler import Compiler

# ag = 1 ant = 0

# TODO Root Folder is src so path names should be referenced from there
# Setup
config = {
    'project': 'Beta 2 Project - Decision Trees',
    'wandb_key': '003bdcde0a7e623fdeb0425c3079a7aed09a32e6',

    'dataset': {
        'src': '../data/raw',
        'labels': 'target',
        'transforms': {
            'merge_datasets': dict(
                name='beta-2-ag-ant',
            ),
            'change_nans': dict(
                value=0
            )
        }
    },
    'models': {
        'decision_tree1': {
            'setup': dict(
                active=True,
                file="decision_tree",
                id="decision_tree_1",
                run_name='Max Depth: 1',
                model_name="CART Model",
                dataset="beta-2-ag-ant",
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
                max_depth=1,
                max_features=None,
                scorer="Matthews Correlation Coefficient"
            )
        },
        'decision_tree2': {
            'setup': dict(
                active=True,
                file="decision_tree",
                id="decision_tree_2",
                run_name='Max Depth: 3',
                model_name="CART Model",
                dataset="beta-2-ag-ant",
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
                max_depth=3,
                max_features=None,
                scorer="Matthews Correlation Coefficient"
            )
        },
        'decision_tree3': {
            'setup': dict(
                active=True,
                file="decision_tree",
                id="decision_tree_3",
                run_name='Max Depth: 5',
                model_name="CART Model",
                dataset="beta-2-ag-ant",
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
                max_depth=5,
                max_features=None,
                scorer="Matthews Correlation Coefficient"
            )
        },
        'decision_tree4': {
            'setup': dict(
                active=True,
                file="decision_tree",
                id="decision_tree_4",
                run_name='Max Depth: 10',
                model_name="CART Model",
                dataset="beta-2-ag-ant",
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
                max_depth=10,
                max_features=None,
                scorer="Matthews Correlation Coefficient"
            )
        },
        'decision_tree5': {
            'setup': dict(
                active=True,
                file="decision_tree",
                id="decision_tree_5",
                run_name='Max Depth: None',
                model_name="CART Model",
                dataset="beta-2-ag-ant",
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
        }
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
