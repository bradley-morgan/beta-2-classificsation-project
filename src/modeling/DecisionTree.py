from sklearn.tree import DecisionTreeClassifier
from scipy.stats import sem
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from src.utils.ImageSaver import ImageSaver
from src.utils.set_path import path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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

        # PLOT and LOG to cloud : Confusion matrix data
        self.run.log({"Validation confusion matrices": self.validation_confusion_matrices})
        self.run.log({'Validation Median Confusion Matrix': self.validation_median_confusion_matrix})
        ax = sns.heatmap(self.validation_median_confusion_matrix,
                         annot=True,
                         cbar=False,
                         fmt='d')
        plt.xlabel("True Label")
        plt.ylabel("Predicted Label")
        plt.close(ax.get_figure())
        plot = ax.get_figure()
        self.image_saver.save(plot=plot, run=self.run, name='Validation confusion_matrix', format='png')
        plt.clf()

        # Log Model scores
        self.run.log({"Validation performance scores": self.validation_scorer_matrix})
        self.run.log({"Validation global performance score": self.validation_mean_performance_score})
        plt.plot(self.validation_scorer_matrix, marker='.', linewidth=2.0)
        plt.title(
            f"Validation MCC Model Performance: Global score = {np.around(self.validation_mean_performance_score, 3)}")
        plt.xlabel('K Folds')
        plt.ylabel('Score')
        plt.yticks([-1, -0.5, 0, 0.5, 1, 1.5])
        plt.xticks([i for i in range(10)])
        self.image_saver.save(plot=plt.gcf(),
                              run=self.run,
                              name='Validation model_performance',
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

        self.run.log({"Trained confusion matrices": self.validation_confusion_matrices})
        self.run.log({'Trained Median Confusion Matrix': self.validation_median_confusion_matrix})
        ax = sns.heatmap(self.validation_median_confusion_matrix,
                         annot=True,
                         cbar=False,
                         fmt='d')
        plt.xlabel("True Label")
        plt.ylabel("Predicted Label")
        plt.close(ax.get_figure())
        plot = ax.get_figure()
        self.image_saver.save(plot=plot, run=self.run, name='Trained confusion_matrix', format='png')
        plt.clf()

    def evaluate_train(self):
        pass

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
