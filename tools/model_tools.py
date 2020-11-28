from joblib import dump, load
import os
from tools.anonymousClass import Obj


def local_load_model(file_name):
    model = load(file_name)
    return model


def local_save_model(model, file_name:str, mete_data:dict, overwrite=False, return_path=False):

    dir_path = './saved_models'
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    if not file_name.endswith('.joblib'):
        file_name = f'{file_name}.joblib'

    if not overwrite:
        if os.path.exists(os.path.join(dir_path, file_name)):
            print(f'The File {file_name} Already Exists! Creating a new version...')
            i = 2
            new_file_name = file_name
            while os.path.exists(os.path.join(dir_path, new_file_name)):
                new_file_name = f'v{i}_{file_name}'
                i += 1
            file_name = new_file_name
            print(f'New version for file: {file_name}')

    data = Obj(model=model, mete_data=mete_data)
    dump(data, os.path.join(dir_path, file_name))
    print('Model Successfully saved')

    if return_path:
        return os.path.join(dir_path, file_name)
