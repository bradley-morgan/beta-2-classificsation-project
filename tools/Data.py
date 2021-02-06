import yaml
from tools.set_path import path
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import importlib
from tqdm import tqdm
import pickle
import joblib
from tools.anonymousClass import Obj


class Preprocess:

    def __init__(self, config_src):
        self.src = config_src
        self.config = self.compile_config()
        self.datasets = {}
        self.execution_chain = []
        self.transform_src = path('transforms')

        self.load()
        self.compile_execution_chain()

    def compile_config(self):
        with open(self.src, mode='r') as yam_file:
            config = yaml.load(yam_file, Loader=yaml.FullLoader)
        return config

    def get_dirs(self, path):
        return  [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    def get_file_type(self, path, file_extension):
        return [file for file in os.listdir(path) if file.endswith(file_extension)]

    def load(self):
        src = path(self.config['src'])
        for sub_folder in self.get_dirs(src):
            for file in self.get_file_type(os.path.join(src, sub_folder), '.csv'):
                data = pd.read_csv(os.path.join(src, sub_folder, file))
                dataset_name = f'{sub_folder}-{file.split(".")[0]}'
                self.datasets[dataset_name] = {'data': data}

    def compile_execution_chain(self):
        """
        Checks that user-set transforms exist as files in the transform folder
        If it exists it then load and initializes the transform
        :return:
        """
        transform_files = os.listdir(self.transform_src)
        transform_files = [file for file in transform_files if file.endswith('.py')]

        user_transforms = self.config['transforms']

        transforms = []
        for transform in user_transforms.keys():
            if f'{transform}.py' in transform_files:

                plugin = importlib.import_module(f'transforms.{transform}')
                plugin = plugin.Transform(user_transforms[transform])
                transforms.append(plugin)

        self.execution_chain = transforms

    def transform(self):
        for transform in self.execution_chain:
            self.datasets = transform(self.datasets)

    def provide_raw(self, ds_name, y_name=None, shuffle=False, to_numpy=True, remove_features=None, dtype='int64'):

        dataFrame = self.datasets[ds_name]['data']
        removed_features = None
        if remove_features is not None:
            removed_features = dataFrame[remove_features]
            dataFrame.drop(remove_features, axis=1, inplace=True)

        dataFrame = dataFrame.astype(dtype)
        feature_names = dataFrame.columns.to_numpy()

        if shuffle:
            dataFrame = dataFrame.sample(frac=1)

        y = None
        if y_name is not None:
            y = dataFrame[y_name].to_numpy()
            dataFrame.drop([y_name], axis=1, inplace=True)

        if to_numpy:
            dataFrame = dataFrame.to_numpy()


        return Obj(
            data=dataFrame,
            feature_names=feature_names,
            labels=y,
            removed_features=removed_features
        )

    def provide(self, ds_name, y_name, to_numpy=True, remove_features=None,  dtype='int64'):

        dataFrame = self.datasets[ds_name]['data']

        y = dataFrame[y_name].to_numpy()
        x = dataFrame.drop(y_name, axis=1)
        feature_names = x.columns
        x = x.to_numpy()

        test_size = self.config['test_size']
        x_train, x_hold_out, y_train, y_hold_out = train_test_split(
                                                            x, y,
                                                            test_size=test_size,
                                                            shuffle=True,
                                                            stratify=y)
        removed_features = None
        if not to_numpy:
            x_train = pd.DataFrame(x_train, columns=feature_names)
            x_hold_out= pd.DataFrame(x_hold_out, columns=feature_names)

            if remove_features is not None:
                removed_features = Obj(x_train=x_train[remove_features], x_hold_out=x_hold_out[remove_features])
                x_train.drop(remove_features, axis=1, inplace=True)
                x_hold_out.drop(remove_features, axis=1, inplace=True)
                feature_names = feature_names.drop(remove_features)


        x_train = x_train.astype(dtype)
        x_hold_out = x_hold_out.astype(dtype)
        feature_names = feature_names.to_numpy()


        return Obj(
            x_train=x_train,
            x_hold_out=x_hold_out,
            y_train=y_train,
            y_hold_out=y_hold_out,
            feature_names=feature_names,
            removed_features=removed_features
        )


    def save_as_pickle(self, data, name: str, dest: str):
        if not os.path.exists(dest):
            os.mkdir(dest)

        if not name.endswith('.pickle'):
            name = f'{name}.pickle'

        with open(os.path.join(dest, name), 'wb') as out_put_file:
            pickle.dump(data, out_put_file)
            print(f'Pickle Save Successful for: {dest}')

    @staticmethod
    def load_from_local(file_path:str):

        if not os.path.exists(file_path):
            raise FileExistsError(f'File Does not exist: {file_path} try using tools.path')

        if file_path.endswith('.joblib'):
            data = joblib.load(file_path)
            print(f'Joblib Load Successful for: {file_path}')
            return data

        elif file_path.endswith('.pickle'):
            with open(file_path, 'rb') as input_put_file:
                data = pickle.load(input_put_file)
                print(f'Pickle Load Successful for: {file_path}')
            return data

        else:
            raise ValueError(f'Not supported file type use pickle or joblib: {file_path}')


if __name__ == '__main__':

    train_processor = Preprocess(config_src='../configs/train_preproessing_config.yaml')
    train_processor.transform()
    train_data = train_processor.provide(ds_name='merged-data', remove_features=['Ligand_Pose'], to_numpy=False, y_name='Action')

    zinc_processor = Preprocess(config_src='../configs/ZINC_processing_config.yaml')
    zinc_processor.transform()
    zinc_data = zinc_processor.provide_raw(ds_name='merged-data', remove_features=['Ligand_Pose2'])

    # Reduce the dataset sizes to only contain shared features between Zinc and Train datasets
    zinc_feature_set = set(zinc_data.feature_names)
    train_feature_set = set(train_data.feature_names)
    shared_features = list(train_feature_set.intersection(zinc_feature_set))

    train_data.x_train = train_data.x_train[shared_features].to_numpy()
    train_data.x_hold_out = train_data.x_hold_out[shared_features].to_numpy()
    train_data.feature_names = shared_features
    # zinc_data['data_reduced'] = zinc_data['data'][shared_features]
    # zinc_data['feature_names_reduced'] = shared_features
    train_processor.save_as_pickle(train_data, 'filter-train-processor_98.pickle', dest='../data/processed/filtered')
    zinc_processor.save_as_pickle(zinc_data, 'filter-zinc-train-processor_98.pickle', dest='../data/processed/zinc')