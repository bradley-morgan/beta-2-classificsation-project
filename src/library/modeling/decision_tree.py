from sklearn.tree import DecisionTreeClassifier
from joblib import dump
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from src.library.metrics.matthews_coefficient import matthews_coefficient
from src.utils.ImageSaver import ImageSaver
from src.utils.set_path import path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb
from tqdm import tqdm
import os


# TODO Support for configurable performance metrics

class Model:

    def __init__(self, config):
        # setup
        self.config = config
        self.run = None
        self.image_saver = ImageSaver()

        # Model agnostic attributes
        self.id = self.config["setup"]["id"]
        self.dtype = self.config["setup"]["dtype"]
        self.run_name = self.config["setup"]["run_name"]
        self.model_name = self.config["setup"]["model_name"]
        self.dataset = self.config["setup"]["dataset"]
        self.y_labels = self.config["setup"]["y_labels"]
        self.save_model = self.config["setup"]["save_model"]
        self.log = self.config["setup"]["log"]

        # Model specific attributes
        self.n_jobs = self.config["models"]["n_jobs"]
        self.k_folds = self.config["models"]["k_folds"]
        self.learning_curve = self.config["models"]["learning_curve"]
        self.class_names = self.config["models"]["class_names"]
        self.criterion = self.config["models"]["criterion"]
        self.splitter = self.config["models"]["splitter"]
        self.max_depth = self.config["models"]["max_depth"]
        self.max_features = self.config["models"]["max_features"]

        # Default Attributes
        self.x_train = None
        self.y_train = None
        # Test sets should not be used during k-fold training or optimization
        self.x_test = None
        self.y_test = None
        self.feature_names = None

        # Model Performance attributes for K-fold validation only
        self.validation_models = []
        self.validation_global_performance_score = None
        self.validation_scorer_matrix = []
        self.validation_confusion_matrices = []
        self.validation_median_confusion_matrix = None

        # Model Attributes and performance for final training and evaluation
        self.final_model = None
        self.final_confusion_matrix = None
        self.final_performance_score = None

        # Hidden attributes
        self.__save_dir__ = path('../models')

        #Flags
        self.train_flag = False


    def init_run(self):
        # setup
        if self.log:
            self.run = wandb.init(project=self.config["setup"]["project"],
                                  name=self.config["setup"]["run_name"],
                                  id=self.config["setup"]["id"],
                                  config=self.config, reinit=True)

    def validate(self, dataset):
        # Should always begin with this
        self.init_run()
        self.split_data(dataset)

        # setup stratified K-fold cross validation
        cv = StratifiedKFold(n_splits=self.k_folds, shuffle=True)

        with tqdm(total=self.k_folds, bar_format='{l_bar}{bar:100}{r_bar}{bar:-100b}') as progress_bar:
            for train_idx, test_idx in cv.split(self.x_train, self.y_train):
                # extract hold out test set
                train_x, val_x = self.x_train[train_idx], self.x_train[test_idx]
                train_y, test_y = self.y_train[train_idx], self.y_train[test_idx]

                # summarize train and test composition
                # train_0, train_1 = len(train_y[train_y == 0]), len(train_y[train_y == 1])
                # test_0, test_1 = len(test_y[test_y == 0]), len(test_y[test_y == 1])
                # print('>Train: 0: %d, 1: %d, Test: 0: %d, 1: %d' % (train_0, train_1, test_0, test_1))

                # Fit and Validate models then generate confusion matrix
                model = DecisionTreeClassifier(
                    criterion=self.criterion,
                    splitter=self.splitter,
                    max_depth=self.max_depth,
                    max_features=self.max_features
                )
                model.fit(train_x, train_y)
                y_preds = model.predict(val_x)

                conf_mat = confusion_matrix(test_y, y_preds)
                self.validation_confusion_matrices.append(conf_mat)

                # Calculate matthew correlation coeffcient
                score = matthews_coefficient(conf_mat)
                self.validation_scorer_matrix.append(score)
                self.validation_models.append(model)

                progress_bar.update(1)

    def train(self):
        # Once models has been a validated we can train a single models on the
        # The entire dastaset which we will use for further testing and prediction

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
        score = matthews_coefficient(conf_mat)

        self.final_confusion_matrix = conf_mat
        self.final_performance_score = score
        self.final_model = model

        self.train_flag = True
        # save the models
        if self.save_model:
            self.save()

    def log(self):
        if not self.log:
            raise ValueError("Log Error: Log is set to False thus no run object is available for logging")
        sns.set()

        # Plots for Validation Models
        # Plot the best tree from K-fold experiments
        best_model_idx = np.argmax(self.validation_scorer_matrix)
        best_model = self.validation_models[best_model_idx]

        self.image_saver.save_graphviz(model=best_model,
                                       run=self.run,
                                       graph_name=f'{self.model_name}: Tree Structure',
                                       feature_names=self.feature_names,
                                       class_names=self.class_names)

        # PLOT and LOG to cloud : Confusion matrix data
        self.run.log({"confusion matrices": self.validation_confusion_matrices})
        self.run.log({'Median Confusion Matrix': self.validation_median_confusion_matrix})
        ax = sns.heatmap(self.validation_median_confusion_matrix,
                         annot=True,
                         cbar=False,
                         fmt='d')
        plt.xlabel("True Label")
        plt.ylabel("Predicted Label")
        plt.close(ax.get_figure())
        plot = ax.get_figure()
        self.image_saver.save(plot=plot, run=self.run, name='confusion_matrix', format='png')
        plt.clf()

        # Log Model scores
        self.run.log({"performance scores": self.validation_scorer_matrix})
        self.run.log({"global performance score": self.validation_global_performance_score})
        plt.plot(self.validation_scorer_matrix, marker='.', linewidth=2.0)
        plt.title(f"MCC Model Performance: Global score = {np.around(self.validation_global_performance_score, 3)}")
        plt.xlabel('K Folds')
        plt.ylabel('Score')
        plt.yticks([-1, -0.5, 0, 0.5, 1, 1.5])
        plt.xticks([i for i in range(10)])
        self.image_saver.save(plot=plt.gcf(),
                              run=self.run,
                              name='model_performance',
                              format='png')
        plt.clf()

    def evaluate(self):

        # Calculate median confusion matrix across k-folds
        mat = np.asarray(self.validation_confusion_matrices)
        median_confusion_matrix = np.median(mat, axis=0)
        median_confusion_matrix = median_confusion_matrix.astype('int64')
        self.validation_median_confusion_matrix = median_confusion_matrix

        # calculate global matthews correlation coefficient
        self.validation_global_performance_score = np.mean(self.validation_scorer_matrix)

        # Extract the decision path & whether node features of split nodes are shared or not shared
        # between ag and ant classes

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


if __name__ == "__main__":
    pass
