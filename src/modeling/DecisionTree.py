from sklearn.tree import DecisionTreeClassifier
from scipy.stats import sem
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, matthews_corrcoef, make_scorer
from sklearn.inspection import permutation_importance
from src.utils.ImageSaver import ImageSaver
from src.utils.set_path import path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
import os
from joblib import dump


class DecisionTree:

    def __init__(self, config, project):

        # setup
        self.project = project

        self.config = config
        self.run = None
        self.image_saver = ImageSaver()
        self.log_data = self.config["setup"]["log_data"]
        self.save_model = self.config["setup"]["save_model"]

        # Model details
        self.id = self.config["setup"]["id"]
        self.dtype = self.config["setup"]["dtype"]
        self.run_name = self.config["setup"]["run_name"]
        self.model_name = self.config["setup"]["model_name"]
        self.dataset = self.config["setup"]["dataset"]
        self.y_labels = self.config["setup"]["y_labels"]

        # Model parameters
        self.n_jobs = self.config["models"]["n_jobs"]
        self.n_repeats = self.config["models"]["n_repeats"]
        self.k_folds = self.config["models"]["k_folds"]
        self.class_names = self.config["models"]["class_names"]
        self.criterion = self.config["models"]["criterion"]
        self.splitter = self.config["models"]["splitter"]
        self.max_depth = self.config["models"]["max_depth"]
        self.max_features = self.config["models"]["max_features"]
        self.feature_importance_repeats = self.config["models"]["feature_importance_repeats"]

        # Data Attributes
        self.x_train = None
        self.y_train = None
        # Test sets should not be used during k-fold training or optimization
        self.x_test = None
        self.y_test = None
        self.feature_names = None

        # VALIDATION: Model Performance attributes for K-fold validation only
        self.validation_scorer_matrix = []
        self.validation_confusion_matrices = []
        self.validation_mean_performance_score = None
        self.validation_standard_error = None
        self.validation_median_confusion_matrix = None

        # TRAINING: Model Attributes and performance for final training and evaluation
        self.final_model = None
        self.final_confusion_matrix = None
        self.final_performance_score = None
        self.final_model_split_node_features = None
        self.final_train_feature_importance = None
        self.final_test_feature_importance = None
        self.final_tree_feature_table = None


        # Hidden attributes
        self.__save_dir__ = path('../models')

        # Flags
        self.train_flag = False

    def init_run(self):
        # setup
        if self.log_data:
            self.run = wandb.init(project=self.project,
                                  name=self.run_name,
                                  id=self.id,
                                  config=self.config, reinit=True)

    def validate(self, dataset):
        """
        This Function runs Repeated K-Fold Cross Validation
        :param dataset: The dict of datasets for the model config to select training data from
        """
        self.init_run()
        self.split_data(dataset)

        with tqdm(total=self.n_repeats, bar_format='{l_bar}{bar:100}{r_bar}{bar:-100b}',
                  desc='n_repeats') as progress_bar:

            for r in range(self.n_repeats):
                # setup stratified K-fold cross validation
                cv = StratifiedKFold(n_splits=self.k_folds, shuffle=True)
                for train_idx, test_idx in cv.split(self.x_train, self.y_train):
                    # extract hold out test set
                    train_x, val_x = self.x_train[train_idx], self.x_train[test_idx]
                    train_y, val_y = self.y_train[train_idx], self.y_train[test_idx]

                    # Fit and Validate models then generate confusion matrix
                    model = DecisionTreeClassifier(
                        criterion=self.criterion,
                        splitter=self.splitter,
                        max_depth=self.max_depth,
                        max_features=self.max_features
                    )
                    model.fit(train_x, train_y)
                    y_preds = model.predict(val_x)

                    conf_mat = confusion_matrix(val_y, y_preds)
                    self.validation_confusion_matrices.append(conf_mat)

                    # Calculate matthew correlation coeffcient
                    score = matthews_corrcoef(y_true=val_y, y_pred=y_preds)
                    self.validation_scorer_matrix.append(score)

                progress_bar.update(1)

    def log_validation(self):
        sns.set()

        # # PLOT and LOG to cloud : Confusion matrix data
        # self.run.log({"Validation confusion matrices": self.validation_confusion_matrices})
        # self.run.log({'Validation Median Confusion Matrix': self.validation_median_confusion_matrix})
        ax = sns.heatmap(self.validation_median_confusion_matrix,
                         annot=True,
                         cbar=False,
                         fmt='d')
        plt.xlabel("True Label")
        plt.ylabel("Predicted Label")
        plt.close(ax.get_figure())
        plot = ax.get_figure()
        self.image_saver.save(plot=plot, run=self.run, name='Validation: Median confusion_matrix', format='png')
        plt.clf()

        # Log Model scores
        # self.run.log({"Validation performance scores": self.validation_scorer_matrix})
        # self.run.log({"Validation Mean performance score": self.validation_mean_performance_score})
        plt.plot(self.validation_scorer_matrix, linewidth=2.0)
        plt.title(
            f"Validation MCC Model Performance: Mean score={np.around(self.validation_mean_performance_score, 3)} Std Error={np.around(self.validation_standard_error, 3)}")
        plt.xlabel('K Folds')
        plt.ylabel('Score')
        # plt.yticks([-1, -0.5, 0, 0.5, 1, 1.5])
        # plt.xticks([i for i in range(len(self.validation_scorer_matrix))])
        self.image_saver.save(plot=plt.gcf(),
                              run=self.run,
                              name=f'Validation Model Performance: repeats={self.n_repeats} K-folds={self.k_folds}',
                              format='png')
        plt.clf()

    def evaluate_validation(self):
        # Calculate median confusion matrix across k-folds
        mat = np.asarray(self.validation_confusion_matrices)
        median_confusion_matrix = np.median(mat, axis=0)
        median_confusion_matrix = median_confusion_matrix.astype('int64')
        self.validation_median_confusion_matrix = median_confusion_matrix

        # calculate global matthews correlation coefficient
        self.validation_mean_performance_score = np.mean(self.validation_scorer_matrix)

        # Calculate standard error
        self.validation_standard_error = sem(self.validation_scorer_matrix)

    def train(self):
        """
            FINAL MODEL EVALUATION
            Once models has been a validated we can train a single models on the
            The entire dastaset which we will use for further testing and prediction
        """
        model = DecisionTreeClassifier(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            max_features=self.max_features
        )
        model.fit(self.x_train, self.y_train)
        y_preds = model.predict(self.x_test)

        conf_mat = confusion_matrix(self.y_test, y_preds)
        # Calculate matthew correlation coeffcient
        score = matthews_corrcoef(y_true=self.y_test, y_pred=y_preds)

        self.final_confusion_matrix = conf_mat
        self.final_performance_score = score
        self.final_model = model

        self.train_flag = True
        # save the models
        if self.save_model:
            self.save()

    def log_train(self):
        # Plots for Final trained models
        # Plot the best tree from K-fold experiments
        # TODO Refactor for training
        self.image_saver.save_graphviz(model=self.final_model,
                                       run=self.run,
                                       graph_name=f'Final Trained {self.model_name}: Tree Structure',
                                       feature_names=self.feature_names,
                                       class_names=self.class_names)

        self.run.log({"Trained confusion matrices on Test set": self.final_confusion_matrix})
        self.run.log({'Trained performance score on test set': self.final_performance_score})
        ax = sns.heatmap(self.final_confusion_matrix,
                         annot=True,
                         cbar=False,
                         fmt='d')
        plt.xlabel("True Label")
        plt.ylabel("Predicted Label")
        plt.close(ax.get_figure())
        plot = ax.get_figure()
        self.image_saver.save(plot=plot, run=self.run, name='Trained confusion_matrix on Test Set', format='png')
        plt.clf()

        self.run.log({'Feature Importance on Training Set': wandb.Table(dataframe=self.final_train_feature_importance)})
        self.run.log({'Feature Importance on Test Set': wandb.Table(dataframe=self.final_test_feature_importance)})
        self.run.log({'Decision Tree Split Nodes': wandb.Table(dataframe=self.final_tree_feature_table)})


    def evaluate_train(self, datasetCompiler, target_dataset):

        # Get feature importance for training data
        self.feature_importance(mode='train')
        # Feature importance for test data
        self.feature_importance(mode='test')

        self.extract_tree_features(datasetCompiler, target_dataset)

    def feature_importance(self, mode):

        print(f"Computing Permutation Feature Importance for {mode}...")
        x_data = None
        y_data = None
        if mode == 'train':
            x_data = self.x_train
            y_data = self.y_train

        elif mode == 'test':
            x_data = self.x_test
            y_data = self.y_test

        # Get feature importance for training data
        scorer = make_scorer(matthews_corrcoef)
        train_feature_importance = permutation_importance(self.final_model,
                                                          x_data,
                                                          y_data,
                                                          scoring=scorer,
                                                          n_repeats=self.feature_importance_repeats,
                                                          n_jobs=self.n_jobs)

        importance_mean = train_feature_importance["importances_mean"]
        most_important_features = [(idx, x) for idx, x in enumerate(importance_mean) if x > 0]
        feature_importance_table = {'feature': [], 'Permutation importance (MCC)': []}

        for item in most_important_features:
            idx = item[0]
            importance = item[1]
            feature_name = self.feature_names[idx]
            feature_importance_table['feature'].append(feature_name)
            feature_importance_table['Permutation importance (MCC)'].append(importance)

        if mode == 'train':
            self.final_train_feature_importance = pd.DataFrame.from_dict(feature_importance_table)
        elif mode == 'test':
            self.final_test_feature_importance = pd.DataFrame.from_dict(feature_importance_table)

    def extract_tree_features(self, datasetCompiler, target_dataset):
        # Get Tree data
        n_nodes = self.final_model.tree_.node_count
        children_left = self.final_model.tree_.children_left
        children_right = self.final_model.tree_.children_right
        feature = self.final_model.tree_.feature

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True

        split_nodes = []
        for i in range(n_nodes):
            if not is_leaves[i]:
                split_nodes.append(feature[i])

        tree_feature_list = []
        for node in split_nodes:
            feature_name = datasetCompiler.datasets[target_dataset]["feature_table"].iloc[node]["feature_name"]

            keys = list(datasetCompiler.datasets)
            keys.remove(target_dataset)

            for key in keys:
                feature_table = datasetCompiler.datasets[key]["feature_table"]
                x = feature_table[feature_table["feature_name"] == feature_name].copy(deep=True)

                if len(x) > 0:
                    x["dataset"] = key
                    tree_feature_list.append(x)

        self.final_tree_feature_table = pd.concat(tree_feature_list)

    def save(self):
        if not self.train_flag:
            raise ValueError(f'Error In Save Method in {__file__}: Save needs a final models'
                             f' to use please run train() method')

        save_path = os.path.join(self.__save_dir__, f'{self.model_name}.joblib')
        dump(self.final_model, save_path)
        print(f'Model {self.model_name}.joblib sucessfully saved to models folder')

    def load(self):
        pass

    def terminate(self):
        self.image_saver.clean_up()
        self.run.finish()

    def split_data(self, dataset):
        x_train, x_test, y_train, y_test, feature_names = dataset.provide(self.dataset,
                                                                          self.y_labels,
                                                                          dtype=self.dtype)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.feature_names = feature_names
