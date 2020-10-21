from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from src.library.metrics.matthews_coefficient import matthews_coefficient
from src.utils.ImageSaver import ImageSaver
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np
import wandb
from tqdm import tqdm

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
        self.k_folds = self.config["model"]["k_folds"]
        self.model_type = self.config["model"]["model_type"]

        # Default Attributes
        self.x_data = None
        self.y_data = None


        # Model Performance attributes
        self.global_score = None
        self.scorer_matrix = []
        self.y_preds = []
        self.confusion_matrices = []
        self.median_confusion_matrix = None
        self.learning_curve = []

        #Initializers

    def init_run(self):
        self.run = wandb.init(project=self.config["setup"]["project"],
                              name=self.config["setup"]["run_name"],
                              id=self.config["setup"]["id"],
                              notes=self.config["setup"]["notes"],
                              config=self.config)

    def train(self, dataset):
        self.init_run()
        self.split_data(dataset)

        elements, counts = np.unique(self.y_data, return_counts=True)
        class_selector = None
        if self.model_type == 'majority':
            # determine majority class
            max_idx = np.argmax(counts)
            class_selector = elements[max_idx]

        elif self.model_type == "minority":
            min_idx = np.argmin(counts)
            class_selector = elements[min_idx]

        # setup stratified K-fold cross validation
        cv = StratifiedKFold(n_splits=self.k_folds, shuffle=True)

        with tqdm(total=self.k_folds, bar_format='{l_bar}{bar:100}{r_bar}{bar:-100b}') as progress_bar:
            for train_idx, test_idx in cv.split(self.x_data, self.y_data):
                # extract hold out test set
                train_x, test_x = self.x_data[train_idx], self.x_data[test_idx]
                train_y, test_y = self.y_data[train_idx], self.y_data[test_idx]

                y_preds = []
                for sample in test_y:
                    if self.model_type == 'majority' or self.model_type == "minority":
                        # determine majority class
                        y_preds.append(class_selector)

                    elif self.model_type == "random":
                        y_preds.append(np.random.choice(elements))

                    else:
                        raise ValueError(f"Invalid model type {self.model_type} choice majority, minority or random")


                y_preds = np.asarray(y_preds)
                conf_mat = confusion_matrix(test_y, y_preds)
                self.confusion_matrices.append(conf_mat)

                # Calculate matthew correlation coeffcient
                #score = matthews_corrcoef(test_y, y_preds)
                score = matthews_coefficient(conf_mat)
                self.scorer_matrix.append(score)

                self.y_preds = self.y_preds + y_preds.tolist()

                progress_bar.update(1)




    def log(self):
        sns.set()
        # # log model choice counts
        plt.hist(self.y_preds, bins=2, rwidth=0.5)
        plt.title(f"Model Class Selection across {self.k_folds} folds")
        plt.xlabel("classes")
        plt.ylabel("Frequency")
        plt.xticks([0, 1])
        self.image_saver.save(plot=plt.gcf(),
                              run=self.run,
                              name='model_selection',
                              format='png')
        plt.clf()


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

    def terminate(self):
        self.run.finish()

    def split_data(self, dataset):
        x_data, y_data = dataset.provide(self.dataset,
                                         self.y_labels,
                                         shuffle=True,
                                         dtype=self.dtype)
        self.x_data = x_data
        self.y_data = y_data

