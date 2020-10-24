from src.processing.Dataset import Dataset
from src.library.modeling.decision_tree import Model

dataset_config = {
    'project': 'test split',
    'wandb_key': None,
    'dataset': {
        'src': '../data/raw',
        'name': 'dataset test',
        'log': False,
        'labels': 'target',
        'notes': 'Data contains B2in, Z, R and Merged datasets',
        'test_size': 0.1,
        'stats': dict(
            names=[],
            label_feature_name="target",
        ),
        'transforms': {
            'merge_datasets': dict(
                merge_all=True,
                merge_all_name='merged-B2in-Z-R',
                groups=[],
                group_names=[]
            ),
        'change_nans': dict(
                        value=0
                    )
        }
    }
}

model_config = {
            'setup': dict(
                project ='test split',
                wandb_key=None,
                log=False,
                active=True,
                file="decision_tree",
                id="decision_tree_1",
                run_name='CART - Merged-B2in-Z-R',
                model_name="CART Model",
                dataset="merged-B2in-Z-R",
                y_labels="target",
                shuffle=True,
                dtype='int64',
                save_model=True
            ),
            'models': dict(
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

dataset = Dataset(config=dataset_config)
dataset.load()
dataset.apply_item(feature_name='target', item=1, names=['R-ag', 'B2in-ag', 'Z-ag'])
dataset.apply_item(feature_name='target', item=0, names=['R-ant', 'B2in-ant', 'Z-ant'])
dataset.remove_feature(feature_name='Ligand_Pose')
dataset.transform()

model = Model(config=model_config)
model.validate(dataset=dataset)
model.train()

a = 0



