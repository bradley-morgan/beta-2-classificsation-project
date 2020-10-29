from src.processing.DatasetCompiler import DatasetCompiler
from src.processing.ModelCompiler import Compiler

# ag = 1 ant = 0

# TODO Root Folder is src so path names should be referenced from there
# Setup
config = {
    'project': 'test split',
    'wandb_key': '003bdcde0a7e623fdeb0425c3079a7aed09a32e6',

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
            # 'drop_nans': dict(
            #     target_datasets=['B2in', 'R', 'Z']
            # )
        }
    },
    'models': {
        'decision_tree1': {
            'setup': dict(
                active=True,
                file="decision_tree",
                log=False,
                id="decision_tree_1",
                run_name='B2in',
                model_name="CART Model",
                dataset="merged-B2in-Z-R",
                y_labels="target",
                shuffle=True,
                dtype='int64',
                save_model=True
            ),
            'models': dict(
                n_jobs=4,
                n_repeats=3,
                k_folds=2,
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

dataset = DatasetCompiler(config=config)
dataset.load()
dataset.apply_item(feature_name='target', item=1, names=['R-ag', 'B2in-ag', 'Z-ag'])
dataset.apply_item(feature_name='target', item=0, names=['R-ant', 'B2in-ant', 'Z-ant'])
dataset.remove_feature(feature_name='Ligand_Pose')
dataset.transform()
dataset.statistics()
dataset.log()

model_chain = Compiler(dataset=dataset, config=config)
model_chain.execute()
