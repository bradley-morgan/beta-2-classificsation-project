from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from src.library.metrics.matthews_coefficient import matthews_coefficient
from src.utils.ImageSaver import ImageSaver
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb
from tqdm import tqdm


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

        # Model specific attributes
        self.n_jobs = self.config["models"]["n_jobs"]
        self.k_folds = self.config["models"]["k_folds"]
        self.learning_curve = self.config["models"]["learning_curve"]
        self.n_estimators = self.config["models"]["n_estimators"]
        self.max_features = self.config["models"]["max_features"]
        self.bootstrap = self.config["models"]["bootstrap"]

        # Default Attributes
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.feature_names = None

        # Model Performance attributes
        self.global_score = None
        self.scorer_matrix = []
        self.confusion_matrices = []
        self.median_confusion_matrix = None
        self.learning_curve = []

        # initializers

    def init_run(self):
        # setup
        self.run = wandb.init(project=self.config["setup"]["project"],
                              name=self.config["setup"]["run_name"],
                              id=self.config["setup"]["id"],
                              notes=self.config["setup"]["notes"],
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
                d = set(train_idx) & set(test_idx)
                train_x, test_x = self.x_train[train_idx], self.x_train[test_idx]
                train_y, test_y = self.y_train[train_idx], self.y_train[test_idx]

                # summarize train and test composition
                train_0, train_1 = len(train_y[train_y == 0]), len(train_y[train_y == 1])
                test_0, test_1 = len(test_y[test_y == 0]), len(test_y[test_y == 1])
                print('>Train: 0: %d, 1: %d, Test: 0: %d, 1: %d' % (train_0, train_1, test_0, test_1))

                # Fit and Validate models then generate confusion matrix
                model = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_features=self.max_features,
                    bootstrap=self.bootstrap,
                    n_jobs=self.n_jobs
                )
                model.fit(train_x, train_y)
                y_preds = model.predict(test_x)

                conf_mat = confusion_matrix(test_y, y_preds)
                self.confusion_matrices.append(conf_mat)

                # Calculate matthew correlation coeffcient
                score = matthews_coefficient(conf_mat)
                self.scorer_matrix.append(score)

                progress_bar.update(1)

        # TODO Add learning Curve

    def log(self):
        sns.set()

        # PLOT and LOG to cloud : Confusion matrix data
        self.run.log({"confusion matrices": self.confusion_matrices})
        self.run.log({'Median Confusion Matrix': self.median_confusion_matrix})
        ax = sns.heatmap(self.median_confusion_matrix,
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
        self.run.log({"performance scores": self.scorer_matrix})
        self.run.log({"global performance score": self.global_score})
        plt.plot(self.scorer_matrix, marker='.', linewidth=2.0)
        plt.title(f"MCC Model Performance: Global score = {np.around(self.global_score, 3)}")
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
        mat = np.asarray(self.confusion_matrices)
        median_confusion_matrix = np.median(mat, axis=0)
        median_confusion_matrix = median_confusion_matrix.astype('int64')
        self.median_confusion_matrix = median_confusion_matrix

        # calculate global matthews correlation coefficient
        self.global_score = np.mean(self.scorer_matrix)

    def save(self):
        pass

    def load(self):
        pass

    def terminate(self):
        self.image_saver.clean_up()
        self.run.finish()

    def split_data(self, dataset):
        x_train, x_test, y_train, y_test, feature_names = dataset.provide(self.dataset,
                                                                          self.y_labels,
                                                                          shuffle=True,
                                                                          dtype=self.dtype)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.feature_names = feature_names


if __name__ == "__main__":
    pass
