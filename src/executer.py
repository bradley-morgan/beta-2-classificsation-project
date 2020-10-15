from src.processing.Dataset import Dataset
from src.processing.ModelCompiler import Compiler

# TODO Root Folder is src so path names should be referenced from there
# Setup
config = {
    'project': 'Test Project',
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
        'random_forest': {
            'setup': dict(
                id="test-id-1",
                run_name='RFC Test Run HELLO THERE',
                model_name="Standard RandomForest 1",
                dataset="beta-2-ag-ant",
                y_labels="target",
                shuffle=True,
                dtype='int64'
            ),
            'model': dict(
                k_folds=10,
                learning_curve=True,
                n_estimators=2,
                max_features=10,
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
