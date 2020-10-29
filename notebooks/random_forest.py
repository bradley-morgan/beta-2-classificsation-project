import wandb as wb
from numpy import mean, std
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, StratifiedKFold, cross_validate
from sklearn.metrics import matthews_corrcoef, make_scorer, confusion_matrix
from src.processing.DatasetCompiler import Dataset

# Setup
dataset_parameters = {
        'src': '../../data/raw',
        'labels': 'target',
        'transforms': {
            'merge_datasets':{
                'name': 'beta-2-ag-ant',
            },
            'change_nans':{
                'value': 0
            }
        }
    }

hyperparameter_defaults = dict(
    n_estimators=100,
    max_features=10
)

# run = wb.init(config=hyperparameter_defaults, name='test run1', project='test project 1')
# config = wb.config
dataset = Dataset(config=dataset_parameters)
dataset.load()
dataset.apply_item(feature_name='target', item=1, names=['R-ag', 'B2in-ag', 'Z-ag'])
dataset.apply_item(feature_name='target', item=0, names=['R-ant', 'B2in-ant', 'Z-ant'])
dataset.remove_feature(feature_name='Ligand_Pose')
dataset.transform()

x_data, y_data = dataset.provide(name='beta-2-ag-ant', split=True)


# Model Definition
cv = StratifiedKFold(n_splits=5, shuffle=True)
model = RandomForestClassifier(n_estimators=10, n_jobs=1)

MCC_scorer = make_scorer(matthews_corrcoef)
cv_results = cross_validate(model, x_data, y_data, scoring=MCC_scorer, cv=cv, return_train_score=True)
a = 0



# scores = []
for train_idx, test_idx in cv.split(x_data, y_data):
    train_x, test_x = x_data[train_idx], x_data[test_idx]
    train_y, test_y = y_data[train_idx], y_data[test_idx]

    # summarize train and test composition
    train_0, train_1 = len(train_y[train_y == 0]), len(train_y[train_y == 1])
    test_0, test_1 = len(test_y[test_y == 0]), len(test_y[test_y == 1])
    print('>Train: 0: %d, 1: %d, Test: 0: %d, 1: %d' % (train_0, train_1, test_0, test_1))

    model = RandomForestClassifier()
    s = model.fit(train_x, train_y).score(train_x, train_y)
    y_preds = model.predict(test_x)
    conf_mat = confusion_matrix(test_y, y_preds)

    fig, ax = plt.subplots(figsize=(3,3))
    ax = sns.heatmap(conf_mat,
                     annot=True,
                     cbar=False,
                     fmt='d')
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")

    a = 0













# MCC_scorer = make_scorer(matthews_corrcoef)
# scores = cross_val_score(models, x_data, y_data, scoring=MCC_scorer, cv=cv, n_jobs=1)
#
# #Metrics
# print('MCC: %.3f (%.3f)' % (mean(scores), std(scores)))