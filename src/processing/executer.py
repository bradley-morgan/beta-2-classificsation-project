from src.processing.Dataset import Dataset
from src.processing.ModelCompiler import Compiler

# TODO Root Folder is src so path names should be referenced from there
# Setup
config = {
    'project': 'Beta 2 Experiment 3',
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
        'naive_model1': {
            'setup': dict(
                active=False,
                file="naive_model",
                id="naive-majority-1",
                run_name='naive model majority class',
                model_name="naive model majority class",
                dataset="beta-2-ag-ant",
                y_labels="target",
                shuffle=True,
                dtype='int64'
            ),
            'model': dict(
                k_folds=10,
                model_type="majority",
                scorer="Matthews Correlation Coefficient"
            )
        },
        'naive_model2': {
            'setup': dict(
                active=False,
                file="naive_model",
                id="naive-minority-1",
                run_name='naive model minority class',
                model_name="naive model minority class",
                dataset="beta-2-ag-ant",
                y_labels="target",
                shuffle=True,
                dtype='int64'
            ),
            'model': dict(
                k_folds=10,
                model_type="minority",
                scorer="Matthews Correlation Coefficient"
            )
        },
        'naive_model3': {
            'setup': dict(
                active=False,
                file="naive_model",
                id="naive-random-1",
                run_name='naive model random class',
                model_name="naive model random class",
                dataset="beta-2-ag-ant",
                y_labels="target",
                shuffle=True,
                dtype='int64'
            ),
            'model': dict(
                k_folds=10,
                model_type="random",
                scorer="Matthews Correlation Coefficient"
            )
        },
        'random_forest1': {
            'setup': dict(
                active=False,
                file="random_forest",
                id="random-forest-1",
                run_name='Random Forest Complex',
                model_name="Standard RandomForest 1",
                dataset="beta-2-ag-ant",
                y_labels="target",
                shuffle=True,
                dtype='int64'
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
                id="random-forest-3",
                run_name='Random Forest Decision Tree',
                model_name="Standard RandomForest as DT",
                dataset="beta-2-ag-ant",
                y_labels="target",
                shuffle=True,
                dtype='int64'
            ),
            'model': dict(
                n_jobs=3,
                k_folds=10,
                learning_curve=True,
                n_estimators=1,
                max_features=1,
                bootstrap=False,
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
