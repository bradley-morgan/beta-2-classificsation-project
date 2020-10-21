from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import export_graphviz
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from src.library.metrics.matthews_coefficient import matthews_coefficient
from src.utils.ImageSaver import ImageSaver
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb
from tqdm import tqdm

#TODO Support for configurable performance metrics

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
        self.n_jobs = self.config["model"]["n_jobs"]
        self.k_folds = self.config["model"]["k_folds"]
        self.learning_curve = self.config["model"]["learning_curve"]
        self.class_names = self.config["model"]["class_names"]
        self.criterion = self.config["model"]["criterion"]
        self.splitter = self.config["model"]["splitter"]
        self.max_depth = self.config["model"]["max_depth"]
        self.max_features = self.config["model"]["max_features"]


        # Default Attributes
        self.x_data = None
        self.y_data = None
        self.feature_names = None
        self.models = []

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

    def train(self, dataset):
        # Should always begin with this
        self.init_run()
        self.split_data(dataset)

        # setup stratified K-fold cross validation
        cv = StratifiedKFold(n_splits=self.k_folds, shuffle=True)

        with tqdm(total=self.k_folds, bar_format='{l_bar}{bar:100}{r_bar}{bar:-100b}') as progress_bar:
            for train_idx, test_idx in cv.split(self.x_data, self.y_data):

                # extract hold out test set
                train_x, test_x = self.x_data[train_idx], self.x_data[test_idx]
                train_y, test_y = self.y_data[train_idx], self.y_data[test_idx]

                # summarize train and test composition
                # train_0, train_1 = len(train_y[train_y == 0]), len(train_y[train_y == 1])
                # test_0, test_1 = len(test_y[test_y == 0]), len(test_y[test_y == 1])
                # print('>Train: 0: %d, 1: %d, Test: 0: %d, 1: %d' % (train_0, train_1, test_0, test_1))


                # Fit and Validate model then generate confusion matrix
                model = DecisionTreeClassifier(
                    criterion=self.criterion,
                    splitter=self.splitter,
                    max_depth=self.max_depth,
                    max_features=self.max_features
                )
                model.fit(train_x, train_y)
                y_preds = model.predict(test_x)

                conf_mat = confusion_matrix(test_y, y_preds)
                self.confusion_matrices.append(conf_mat)

                # Calculate matthew correlation coeffcient
                score = matthews_coefficient(conf_mat)
                self.scorer_matrix.append(score)
                self.models.append(model)

                progress_bar.update(1)

        #TODO Add learning Curve

    def log(self):
        sns.set()

        # Plot the best tree from K-fold experiments
        best_model_idx = np.argmax(self.scorer_matrix)
        best_model = self.models[best_model_idx]

        self.image_saver.save_graphviz(model=best_model,
                                       run=self.run,
                                       graph_name=f'{self.model_name}: Tree Structure',
                                       feature_names=self.feature_names,
                                       class_names=self.class_names)


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
        x_data, y_data, feature_names = dataset.provide(self.dataset,
                                         self.y_labels,
                                         shuffle=True,
                                         dtype=self.dtype)
        self.x_data = x_data
        self.y_data = y_data
        self.feature_names = feature_names


if __name__ == "__main__":
    pass