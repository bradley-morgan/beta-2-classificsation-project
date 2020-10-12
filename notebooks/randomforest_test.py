#!/usr/bin/env python3

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


###################### LEARNING CURVE ######################
import seaborn as sns


def plot_learning_curve(clf, title, X, y, ylim, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    f, ax = plt.subplots(figsize=(12, 10))
    # # font for x- & y-axis (ticklabels)
    ax.tick_params(labelsize=18)

    ax.set_title(title, fontsize=28, fontweight='bold')
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples", fontsize=22, fontweight='bold')
    ax.set_ylabel("Score", fontsize=22, fontweight='bold')
    h = np.unique(y)
    train_sizes, train_scores, test_scores = learning_curve(
        clf, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")

    ax.legend(loc="best")
    # # change font size of legend
    plt.setp(ax.get_legend().get_texts(), fontsize='18')

    sns.set_style('whitegrid', {'font.sans-serif': ['Arial']})
    # # despine to keep only 2 axis, instead of a box =.=
    sns.despine()

    return ax


title = "RANDOM FOREST CLASSIFIER"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.

###################### DATA CLEANING, PREPARATION AND MODEL ######################
data = pd.read_csv('../../Data/Z/data.csv')
X = data.drop(['target', 'Ligand_Pose'], axis=1).values
Y = data.target.values
Y = Y.astype(np.int8)
o = np.unique(Y)

clf = RandomForestClassifier(n_estimators=50,  criterion='entropy')
clf2 = BalancedRandomForestClassifier(n_estimators=50,  criterion='entropy')

#cv = ShuffleSplit(n_splits = 100, test_size = 0.2, random_state = 0)
#cv = StratifiedKFold(n_splits=2, random_state=0)
cv = StratifiedShuffleSplit(n_splits = 100, test_size = 0.2, random_state = 0)


# plt.figure(1)
# fig = data["target"].value_counts().plot(kind="bar")

plt.figure(1)
plot_learning_curve(clf, 'Without Undersampling RandomForest', X, Y, (0.7, 1.1), cv=cv, n_jobs=4)
plt.grid()

plt.figure(2)
plot_learning_curve(clf2, 'With Undersampling RandomForest', X, Y, (0.7, 1.1), cv=cv, n_jobs=4)
plt.show()