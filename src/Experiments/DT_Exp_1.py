from src.modeling.DecisionTree import DecisionTree
from src.processing.DatasetCompiler import DatasetCompiler

project = 'test split'
datasetName = 'merged-B2in-Z-R'

datasetConfig = {
    'src': '../data/raw',
    'name': 'dataset test',
    'log_data': False,
    'labels': 'target',
    'notes': 'Data contains B2in, Z, R and Merged datasets',
    'test_size': 0.1,
    'stats': dict(
        names=[],
    ),
    'transforms': {
        'merge_datasets': dict(
            merge_all=True,
            merge_all_name=datasetName,
            groups=[('B2in-ant', 'B2in-ag'), ('R-ant', 'R-ag'), ('Z-ant', 'Z-ag')],
            group_names=['B2in', 'R', 'Z']
        ),
        'change_nans': dict(
            value=0
        )
        # 'drop_nans': dict(
        #     target_datasets=['B2in', 'R', 'Z']
        # )
    }
}

decisionTreeConfig = {
        'setup': dict(
            active=True,
            file="decision_tree",
            log_data=False,
            id="decision_tree_1",
            run_name='Decision Tree 1',
            model_name="DF 1",
            dataset=datasetName,
            y_labels="target",
            shuffle=True,
            dtype='int64',
            save_model=False
        ),
        'models': dict(
            n_jobs=4,
            n_repeats=3,
            k_folds=2,
            class_names=['ant', 'ag'],
            criterion='gini',
            splitter='best',
            max_depth=None,
            max_features=None,
            scorer="Matthews Correlation Coefficient"
        )
}

dataset = DatasetCompiler(config=datasetConfig, project=project)
dataset.load()
dataset.apply_item(feature_name='target', item=1, names=['R-ag', 'B2in-ag', 'Z-ag'])
dataset.apply_item(feature_name='target', item=0, names=['R-ant', 'B2in-ant', 'Z-ant'])
dataset.remove_feature(feature_name='Ligand_Pose')
dataset.statistics()
dataset.transform()
# dataset.log()

model = DecisionTree(config=decisionTreeConfig, project=project)
model.validate(dataset=dataset)
model.evaluate_validation()

a = 0

