from src.processing.Dataset import Dataset
from src.processing.ModelCompiler import Compiler


# Setup
config = {
        'project': 'wandb project name',
        'wandb_key': 'wandb API key',

        'dataset': {
            'src': '../../data/raw',
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
            'random_forest': dict(
                run_name='RFC Test Run',
                model_name="Standard RandomForest 1",
                dataset="beta-2-ag-ant",
                k_folds=10,
                learning_curve=True,
                n_estimators=10,
                max_features=100
            )
        }
    }

dataset = Dataset(config=config["dataset"])
dataset.load()
dataset.apply_item(feature_name='target', item=1, names=['R-ag', 'B2in-ag', 'Z-ag'])
dataset.apply_item(feature_name='target', item=0, names=['R-ant', 'B2in-ant', 'Z-ant'])
dataset.remove_feature(feature_name='Ligand_Pose')
dataset.transform()

model_chain = Compiler(dataset=dataset, config=config["models"])
model_chain.execute()
