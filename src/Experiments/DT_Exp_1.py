from src.modeling.DecisionTree import DecisionTree
from src.processing.DatasetCompiler import DatasetCompiler
from src.library.transforms.merge_datasets import MergeDatasets
from src.library.transforms.change_nans import ChangeNans
import pandas as pd
import wandb

project = 'test run'
datasetName = 'merged-B2in-Z-R'

datasetConfig = {
    'src': '../data/raw',
    'name': 'dataset test',
    'log_data': True,
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
            log_data=True,
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
merge_datasets = MergeDatasets(config=datasetConfig["transforms"]["merge_datasets"])
change_nans = ChangeNans(config=datasetConfig["transforms"]["change_nans"])

dataset.load()
dataset.apply_item(feature_name='target', item=1, names=['R-ag', 'B2in-ag', 'Z-ag'])
dataset.apply_item(feature_name='target', item=0, names=['R-ant', 'B2in-ant', 'Z-ant'])
dataset.remove_feature(feature_name='Ligand_Pose')

dataset = merge_datasets(datasetComplier=dataset)
dataset.statistics()
dataset.log()
dataset.terminate()
dataset = change_nans(dataset)

model = DecisionTree(config=decisionTreeConfig, project=project)
model.validate(dataset=dataset)
model.evaluate_validation()
model.log_validation()
model.train()
model.evaluate_train()
model.log_train()

split_nodes = model.final_model_split_node_features

tree_feature_list = []
for node in split_nodes:
    feature_name = dataset.datasets[datasetName]["feature_table"].iloc[node]["feature_name"]

    keys = list(dataset.datasets)
    keys.remove(datasetName)

    for key in keys:
        feature_table = dataset.datasets[key]["feature_table"]
        x = feature_table[feature_table["feature_name"] == feature_name].copy(deep=True)

        if len(x) > 0:
            x["dataset"] = key
            tree_feature_list.append(x)


tree_feature_table = pd.concat(tree_feature_list)
model.run.log({'Tree Split Nodes Table': wandb.Table(dataframe=tree_feature_table)})

model.terminate()


a = 0

