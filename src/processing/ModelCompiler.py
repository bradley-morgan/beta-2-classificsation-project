import os
import importlib
from src.utils.set_path import path
from warnings import warn

class Compiler:

    def __init__(self, dataset, config):

        # User defined attributes
        self.project = config["project"]
        self.WANDB_API_KEY = config["wandb_key"]
        self.dataset = dataset
        self.config = config["models"]

        # default attributes
        self.model_chain = []

        # hidden attributes
        self.__model_dir__ = path('library/modeling')

        # Flags
        self.models_compiled_flag = False

        # Initializer functions
        self.__init__models__()

    def execute(self):
        if self.models_compiled_flag:
            for model in self.model_chain:
                if model.log_data:
                    model.init_run()
                model.validate(self.dataset)
                model.evaluate_validation()
                if model.log_data:
                    model.log()
                    model.terminate()

    def __init__models__(self):

        os.environ["WANDB_API_KEY"] = self.WANDB_API_KEY
        model_files = os.listdir(self.__model_dir__)
        model_files = [file for file in model_files if file.endswith('.py')]

        user_models_files = self.config.keys()

        models = []
        for user_model in user_models_files:
            model_config = self.config[user_model]
            user_file = model_config["setup"]["file"]

            if f'{user_file}.py' in model_files:
                if model_config["setup"]["active"]:
                    plugin = importlib.import_module(f'{user_file}', package=self.__model_dir__)
                    model_config["setup"]["project"] = self.project
                    plugin = plugin.Model(model_config)
                    models.append(plugin)

        if len(models) == 0:
            warn('Model Compiler: No Model files have been compiled please check your config and make sure your models'
                 'are set to active-True')

        else:
            self.models_compiled_flag = True


        self.model_chain = models






if __name__ == "__main__":
    pass
