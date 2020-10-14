import os
import importlib
from src.utils.set_path import path

class Compiler:

    def __init__(self, dataset, config):

        # User defined attributes
        self.dataset = dataset
        self.config = config

        # default attributes
        self.model_chain = []

        # hidden attributes
        self.__model_dir__ = path('lib/modeling')

        # Initializer functions
        self.__init__models__()


    def execute(self):
        pass

    def __init__models__(self):

        model_files = os.listdir(self.__model_dir__)
        model_files = [file for file in model_files if file.endswith('.py')]

        user_models_files = self.config.keys()

        models = []
        for user_model in user_models_files:
            if f'{user_model}.py' in model_files:
                plugin = importlib.import_module(f'{user_model}', package=self.__model_dir__)
                plugin = plugin.Model(self.config[user_model], #TODO NEEDS X Y DATA)
                models.append(plugin)

        self.model_chain = models






if __name__ == "__main__":
    #
    config = {
        'project': 'wandb project name',
        'wandb_key': 'wandb API key',

        'dataset': {
            'src': '../../data/raw',
            'labels': 'target',
            'transforms': {
                'merge_datasets': dict(
                    name='beta-2-ag-ant',
                ),
                'change_nans': dict(
                    value=0
                )
            }
        },
        'models': {
            'random_forest': dict(
                run_name='RFC Test Run',
                model_name="Standard RandomForest 1",
                dataset="beta-2-ag-ant",
                k_folds=10,
                learning_curve=True,
                n_estimators=10,
                max_features=100
            )
        }
    }
    a = Compiler('placeholder1', config=config["models"])
